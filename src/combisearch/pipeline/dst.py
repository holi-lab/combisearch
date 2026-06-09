"""Dialogue-state-tracking experiment: data loading, example retrieval, prompt
generation, LLM inference, prediction parsing, normalization, and evaluation."""

from __future__ import annotations

import copy
import itertools
import json
import logging
import os
import pprint
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

from tqdm import tqdm

from combisearch.config.paths import resolve_retriever_dir, resolve_search_index_file
from combisearch.config.schema import declared_retrieval_strategy
from combisearch.domain.normalization import AbstractNormalizer
from combisearch.evaluation.dst import calc_prf, evaluate
from combisearch.evaluation.error_analysis import (
    count_prompts_from_examples,
    slot_level_f1,
)
from combisearch.pipeline.state_recorder import PreviousStateRecorder
from combisearch.representation.types import (
    CodexDecodingConfig,
    CompletionParser,
    MultiWOZDict,
    Turn,
)
from combisearch.retrieval.decoders.abstract_example_set_decoder import (
    AbstractExampleListDecoder,
)
from combisearch.retrieval.retrievers.abstract_example_retriever import ExampleRetriever
from combisearch.runtime.io import read_json_from_data_dir
from combisearch.runtime.wandb import WandbStepLogger, wandb_bar_chart


def _build_pretrained_search_index_if_known(
    search_index_filename: Path,
    index_embedding_model_name: str | None,
) -> bool:
    """Build a canonical pretrained dense index when it matches known specs."""
    if not index_embedding_model_name:
        return False

    from combisearch.config.paths import embedding_model_tag
    from combisearch.indexing.build import build_one_index
    from combisearch.indexing.encoder import load_encoder, resolve_encoder_path
    from combisearch.indexing.specs import SPECS

    for spec in SPECS:
        if spec.output_dir(index_embedding_model_name).resolve() == search_index_filename.parent.resolve():
            index_encoder = resolve_encoder_path(index_embedding_model_name, prepare_if_missing=True)
            build_one_index(
                spec,
                model=load_encoder(index_encoder),
                encoder_tag=embedding_model_tag(str(index_encoder)),
                force=False,
                dry_run=False,
                batch_size=64,
            )
            return True
    return False


class DSTExperiment(object):
    """Orchestrates one DST run: load data, build retriever/decoder/LLM, then per
    test turn retrieve examples, prompt the LLM, parse + normalize + merge the
    prediction, and evaluate."""

    train_set: List[Turn]
    test_set: List[Turn]
    use_gold: bool
    ontology_file_path: str
    format_example: Optional[Turn]

    retriever: ExampleRetriever
    demonstration_decoder: AbstractExampleListDecoder
    num_examples: int

    prompt_format: str
    completion_parser: CompletionParser

    lm_decoding_config: CodexDecodingConfig

    prediction_recorder: PreviousStateRecorder
    normalizer: AbstractNormalizer

    output_dir: str
    logger: WandbStepLogger

    def __init__(
        self, 
        runtime_config: Dict[str, Any],
        config_path: str | Path | None = None,
    ) -> None:
        """Builds run components from the config: datasets, ontology/normalizer,
        LLM client, retriever, decoder, prompt generator, completion parser,
        output dir."""

        self.runtime_config = dict(runtime_config)
        self.config_path = Path(config_path) if config_path is not None else None
        self._load_datasets()
        self._build_normalizer()
        self._build_decoding_config()
        self._build_llm_client()
        self._build_retriever_and_decoder()
        self._build_runtime_components()

    def _load_datasets(self) -> None:
        self.train_set = read_json_from_data_dir(self.runtime_config["train_fn"])
        self.test_set = read_json_from_data_dir(self.runtime_config["test_fn"])
        self.use_gold = self.runtime_config.get("use_gold", False)
        self.format_example = self.runtime_config.get("format_example")

    def _build_normalizer(self) -> None:
        from combisearch.domain.multiwoz.normalizer import DataOntologyNormalizer
        from combisearch.domain.multiwoz.ontology import Ontology

        self.ontology = Ontology.create_ontology()
        self.ontology_file_path = self.runtime_config.get("ontology_file_path", "domain/multiwoz/db/2.4/ontology.json")
        self.normalizer = DataOntologyNormalizer(
            self.ontology,
            supervised_set=self.train_set,
            counts_from_ontology_file=self.ontology_file_path,
        )

    def _build_decoding_config(self) -> None:
        self.prompt_format = self.runtime_config.get("prompt_format", "nl-prompt-chat")
        self.add_guidelines = self.runtime_config.get("add_guidelines", True)
        self.lm_decoding_config = self.runtime_config.get("lm_decoding_config", {})
        self.beam_search_config = None
        if self.lm_decoding_config is not None:
            self.beam_search_config = {
                'beam_size': self.lm_decoding_config['beam_size']
            } if self.lm_decoding_config.get('beam_size') else None

    def _build_llm_client(self) -> None:
        from combisearch.llm.client import LlamaClient, OpenAIChatClient
        from combisearch.prompting.generator import STOP_SEQUENCES

        llm_engine = self.runtime_config["llm_engine"]
        llm_kwargs = {
            "engine": llm_engine,
            "stop_sequences": STOP_SEQUENCES.get(self.prompt_format),
            "beam_search_config": self.beam_search_config,
        }
        if llm_engine.startswith('gpt'):
            self.llm_client = OpenAIChatClient(**llm_kwargs)
        elif "llama" in llm_engine.lower():
            llm_kwargs["quantization"] = self.runtime_config.get("quantization")
            self.llm_client = LlamaClient(**llm_kwargs)
        else:
            raise ValueError(f"Unsupported LLM engine: {llm_engine}")

    def _build_retriever_and_decoder(self) -> None:
        from combisearch.retrieval.registry import (
            get_decoder_by_type,
            get_retriever_by_type,
        )

        retriever_args = self.runtime_config.get("retriever_args") or {}
        sampling_method = retriever_args.get("sampling_method", "pre_assigned")
        decoder_config = dict(self.runtime_config.get("decoder_config") or {})
        decoder_config.setdefault("num_examples", self.runtime_config.get("num_examples"))

        retrieval_strategy = declared_retrieval_strategy(self.runtime_config)
        embedding_strategies = {"hybrid", "embeddingretriever", "embedding", "sbert", "sgd"}

        search_index_filename = None
        retriever_dir = ""
        if retrieval_strategy in embedding_strategies:
            search_index_filename = resolve_search_index_file(self.runtime_config)
            if search_index_filename is None:
                raise ValueError("Embedding retriever requires retriever_args.search_index_filename or retriever_dir")

            if not search_index_filename.exists():
                _build_pretrained_search_index_if_known(
                    search_index_filename,
                    self.runtime_config.get("index_embedding_model_name"),
                )

            if not search_index_filename.exists():
                raise FileNotFoundError(f"Search index file not found: {search_index_filename}")

            retriever_dir_path = resolve_retriever_dir(self.runtime_config)
            retriever_dir = str(retriever_dir_path) if retriever_dir_path is not None else ""

        self.retriever = get_retriever_by_type(
            retrieval_strategy,
            retriever_dir,
            retriever_args={
                "datasets": [self.train_set],
                "sampling_method": sampling_method,
            },
            search_index_filename=str(search_index_filename) if search_index_filename is not None else None,
            bm25_input_kwargs=retriever_args.get("bm25_input_kwargs") or {},
            dense_input_kwargs=retriever_args.get("dense_input_kwargs") or {},
        )
        self.demonstration_decoder = get_decoder_by_type(
            decoder_config=decoder_config,
            retriever=self.retriever,
        )
        self.num_examples = decoder_config.get("num_examples")

    def _build_runtime_components(self) -> None:
        from combisearch.prompting.generator import (
            CombiSearchPromptGenerator,
            get_completion_parser,
        )

        self.prediction_recorder = PreviousStateRecorder()
        self.prompt_generator = CombiSearchPromptGenerator()
        self.logger = WandbStepLogger()
        self.completion_parser = get_completion_parser(self.prompt_format)
        self.output_dir = self.runtime_config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self) -> Tuple[List[Turn], Dict[str, Any]]:
        """Runs inference over the test set turn by turn; returns
        (running_log, stats)."""
        selected_set: List[Turn] = self.test_set

        running_log: List[Turn] = []
        n_total: int = 0
        n_correct: int = 0
        total_acc: float = 0
        total_f1: float = 0

        for data_item_idx, data_item in tqdm(enumerate(selected_set)):
            n_total += 1

            examples: List[Turn] = self.get_demonstrations(data_item)

            prompt_text_dict: Final[str] = self.get_prompt_text_dict(
                data_item, examples, add_guidelines=self.add_guidelines
            )

            data_item['prompt'] = prompt_text_dict

            predicted_slot_values: MultiWOZDict = {}
            predicted_prior_context: MultiWOZDict = None
            completions: Dict[str, float] = {}
            best_completion: str = ""
            try:
                best_completion, completions, examples = self._generate_and_select_completion(
                    prompt_text_dict, data_item, examples
                )

                predicted_prior_context = self.prediction_recorder.retrieve_previous_turn_state(data_item)
                predicted_slot_values = self.completion_parser(best_completion, predicted_prior_context)

            except Exception as e:
                raise RuntimeError(
                    f"prediction failed for {data_item.get('ID')}_turn_{data_item.get('turn_id')}"
                ) from e

            all_slot_values = self._postprocess_and_merge_prediction(data_item, predicted_slot_values)

            (
                n_correct, total_acc, total_f1
            ) = self._evaluate_and_log_turn(
                data_item=data_item, all_slot_values=all_slot_values, 
                predicted_slot_values=predicted_slot_values, 
                predicted_prior_context=predicted_prior_context,
                completions=completions, best_completion=best_completion, examples=examples, 
                n_total=n_total, n_correct=n_correct, total_acc=total_acc, total_f1=total_f1
            )
    
            running_log.append(data_item)

            # Wait until we have at least 20 examples to avoid overwriting accidentally
            if data_item_idx > 20:
                with open(os.path.join(self.output_dir, "running_log.json"), 'w') as f:
                    json.dump(running_log, f, indent=2)

        stats = self._compute_final_statistics(running_log)
        
        return running_log, stats

    def get_demonstrations(self, data_item: Turn) -> List[Turn]:
        """Returns the few-shot example turns for a test turn: the fixed
        format_example (if set) plus retriever results, capped at num_examples and
        excluding the query's own dialogue."""
        examples: List[Turn] = []
        if self.format_example:
            if isinstance(self.format_example, list):
                examples.extend(self.format_example)
            else:
                examples.append(self.format_example)

        num_examples = self.num_examples if isinstance(self.num_examples, int) else sum(self.num_examples.values())

        if self.use_gold:
            examples.extend(self.retriever.item_to_best_examples(data_item, k=self.num_examples,
                                                            decoder=self.demonstration_decoder))
        elif len(examples) < num_examples:
            predicted_context = self.prediction_recorder.retrieve_previous_turn_state(data_item)

            modified_item = copy.deepcopy(data_item)
            modified_item['last_slot_values'] = predicted_context

            retrieved_examples = self.retriever.item_to_best_examples(
                modified_item, k=self.num_examples, decoder=self.demonstration_decoder)

            if isinstance(retrieved_examples, dict):
                tmp = []
                iter_num = max([len(v) for v in retrieved_examples.values()])
                # Round-robin selection from each strategy
                key_iter = itertools.islice(itertools.cycle(retrieved_examples.keys()), iter_num*len(retrieved_examples))
                while True:
                    try:
                        key = next(key_iter)
                        tmp.append(retrieved_examples[key].pop())
                    except IndexError:
                        # This strategy has no more examples
                        continue
                    except StopIteration:
                        break
                retrieved_examples = [e for e in tmp if e['ID'] != data_item['ID']][::-1]
            examples.extend(retrieved_examples)

        examples = [e for e in examples if e['ID'] != data_item['ID']]

        if len(examples) > num_examples:
            # examples are ordered worst -> best, so take the last ones
            examples = examples[-num_examples:]
        return examples

    def get_prompt_text_dict(
        self, 
        data_item: Turn, 
        examples: List[Turn], 
        add_guidelines:bool=False, 
        add_instruction_every_turn:bool=True
    ) -> Dict[str, str]:
        """Builds the prompt(s) for a turn as {name: prompt}"""
        
        if self.use_gold:
            prompt_text = self.prompt_generator.get_prompt(
                data_item, examples=examples, prompt_format=self.prompt_format, 
                add_guidelines=add_guidelines, add_instruction_every_turn=add_instruction_every_turn)
        else:
            predicted_context = self.prediction_recorder.retrieve_previous_turn_state(data_item)
            prompt_text = self.prompt_generator.get_prompt(
                data_item, examples=examples, given_context=predicted_context, prompt_format=self.prompt_format, 
                chat_format=True, add_guidelines=add_guidelines, add_instruction_every_turn=add_instruction_every_turn)
        return {f"{len(examples)}-shot":prompt_text}

    def generate_completion(
        self, 
        prompt_text: str, 
        data_item: Turn, 
        examples: List[Turn], 
    ) -> Tuple[Dict[str, float], List[Turn]]:
        """Calls the LLM for one prompt; returns (completions, examples) where
        completions maps text -> score. Retries on parse/API errors and drops an
        example on overlength, raising after 5 of either. Only greedy/beam_search
        decoding is supported; other methods raise."""
        from openai._exceptions import APIError, OpenAIError, RateLimitError

        from combisearch.llm.client import PromptOverlengthError

        complete_flag = False
        parse_error_count = 0
        other_error_cnt = 0
        completions: Dict[str, float] = {}
        last_error: BaseException | None = None

        while not complete_flag and parse_error_count < 5 and other_error_cnt < 5:
            try:
                if self.lm_decoding_config is None or self.lm_decoding_config.get("method", "greedy") in ["greedy", "beam_search"]:
                    completions = self.llm_client.greedy_lm_completion(prompt_text)

                else:
                    raise ValueError(f"Unsupported decoding arguments: {self.lm_decoding_config}")
                    
            except PromptOverlengthError as e:
                last_error = e
                logging.warning(e)
                logging.info("prompt overlength, retrying with fewer examples")
                examples = examples[1:]
                prompt_text = next(iter(self.get_prompt_text_dict(data_item=data_item, examples=examples).values()))
                other_error_cnt += 1

            except ValueError as e:
                last_error = e
                logging.exception(e)
                other_error_cnt += 1
                raise e

            except (RateLimitError, APIError, OpenAIError) as e:
                last_error = e
                logging.exception(e)
                other_error_cnt += 1

            except BaseException as e:
                last_error = e
                logging.exception(e)
                if isinstance(e, KeyboardInterrupt):
                    raise e
                other_error_cnt += 1

            else:
                try:
                    predicted_context = self.prediction_recorder.retrieve_previous_turn_state(data_item)
                    tmp_comp = max(completions, key=completions.get)
                    self.completion_parser(tmp_comp, predicted_context)
                    complete_flag = True
                except Exception as e:
                    last_error = e
                    logging.exception("completion validation failed")
                    parse_error_count += 1

        if not complete_flag:
            raise ValueError(
                f"unable to generate completion after parse_errors={parse_error_count}, "
                f"other_errors={other_error_cnt}"
            ) from last_error
        
        return completions, examples

    def _generate_and_select_completion(
        self,
        prompt_text_dict: Dict[str, str],
        data_item: Turn,
        examples: List[Turn]
    ) -> Tuple[str, Dict[str, Dict[str, float]], List[Turn]]:
        """Generates completions for each prompt variant and returns the
        highest-scoring one: (best_completion, all_completions, examples)."""
        all_completions = {}
        all_best_completions = {}

        for ids, prompt_text in prompt_text_dict.items():
            completions = {}
            completions, examples = self.generate_completion(prompt_text, data_item, examples)

            best_completion = max(completions, key=completions.get)
            best_completion = best_completion.strip().replace('agent.state.', '')

            all_completions[ids] = completions
            all_best_completions[ids] = best_completion

        best_completion = list(all_best_completions.values())[0]

        return best_completion, all_completions, examples

    def _postprocess_and_merge_prediction(
        self,
        data_item: Turn,
        predicted_slot_values: MultiWOZDict
    ) -> MultiWOZDict:
        """Normalizes the predicted turn delta, merges it onto the prior state
        (gold if use_gold else the recorded prediction), collapses multi-value
        slots, records the new state, and returns the full dialogue state."""
        # Skip normalization for SGD dataset (different ontology)
        if 'sgd' not in self.prompt_format.lower():
            predicted_slot_values = self.normalizer.normalize(predicted_slot_values)

        if self.use_gold:
            prior_dialogue_state = data_item['last_slot_values'].copy()
        else:
            prior_dialogue_state = self.prediction_recorder.retrieve_previous_turn_state(data_item).copy()

        from combisearch.representation.dialogue_state import update_dialogue_state

        all_slot_values = update_dialogue_state(prior_dialogue_state, predicted_slot_values)

        # Handle slots with multiple values (take first option)
        all_slot_values = {k: v.split('|')[0] for k, v in all_slot_values.items()}

        self.prediction_recorder.add_state(data_item, all_slot_values)

        return all_slot_values

    def _evaluate_and_log_turn(
        self,
        data_item: Turn,
        all_slot_values: MultiWOZDict,
        predicted_slot_values: MultiWOZDict,
        predicted_prior_context: MultiWOZDict,
        completions: Dict[str, float],
        best_completion: str,
        examples: List[Turn],
        n_correct: int,
        n_total: int,
        total_acc: float,
        total_f1: float,
    ) -> Tuple[int, float, float]:
        """Records the prediction fields onto data_item, evaluates the turn
        (JGA/acc/F1), logs the running JGA, and returns updated
        (n_correct, total_acc, total_f1)."""
        data_item['pred'] = all_slot_values
        data_item['pred_delta_slot_values'] = predicted_slot_values
        data_item['pred_prior_context'] = predicted_prior_context or {}
        data_item['completion'] = best_completion
        data_item['all_completions'] = completions
        data_item['num_solutions'] = len(completions)
        data_item['prompt_counts'] = count_prompts_from_examples(examples)
        data_item['examples'] = [(e['ID'], e['turn_id']) for e in examples]

        print(f"\nfew-shot completions: {completions}")
        print(f"few-shot best completion: {best_completion}")
        print(f"this is the {n_total - 1}th example. {data_item['ID']}_turn_{data_item['turn_id']}")
        print(f"system response: {data_item['dialog']['sys'][-1]}")
        print(f"user response: {data_item['dialog']['usr'][-1]}")
        print(f"pred turn change: {pprint.pformat(predicted_slot_values)}")
        print(f"gold turn change: {pprint.pformat(data_item['turn_slot_values'])}")
        print(f"pred states: {pprint.pformat(all_slot_values)}")
        print(f"gold states: {pprint.pformat(data_item['slot_values'])}")

        this_jga, this_acc, this_f1 = evaluate(prediction=all_slot_values, gold=data_item['slot_values'])
        total_acc += this_acc
        total_f1 += this_f1

        if this_jga:
            n_correct += 1
            print("\n=====================correct!=======================")
        else:
            print("\n=====================wrong!=======================")

        print(f"current jga: {n_correct / n_total}, n_correct: {n_correct}, n_total: {n_total}")
        self.logger.log({
            "current_jga": n_correct / n_total, 
            "n_correct": n_correct, 
            "n_total": n_total
        })
        self.logger.step()
        print("\n")

        return n_correct, total_acc, total_f1

    def _compute_final_statistics(self, running_log: List[Turn]) -> Dict[str, Any]:
        """Scores the full running_log (overall, per-slot, per-domain), logs to
        wandb, and returns the overall stats dict."""
        from combisearch.evaluation.score_run import (
            evaluate_on_domains,
            score_running_log,
        )

        stats = score_running_log(running_log, test_set=self.test_set)

        slot_prf = slot_level_f1(running_log, tp_means_correct=True)
        self.logger.log({f"f1/{slot_name}": f1 for slot_name, (_, f1) in slot_prf.items()})
        self.logger.log({f"precision/{slot_name}": calc_prf(f1_dict).precision for slot_name, (f1_dict, f1) in slot_prf.items()})
        self.logger.log({f"recall/{slot_name}": calc_prf(f1_dict).recall for slot_name, (f1_dict, f1) in slot_prf.items()})

        turn_accuracy_chart = wandb_bar_chart(
            [[f"{turn_id}", acc] for turn_id, acc in stats['turn_accuracies'].items()],
            columns=['turn_id', 'accuracy'],
            x="turn_id",
            y="accuracy",
            title="accuracy by turn id",
        )
        if turn_accuracy_chart is not None:
            stats['turn_accuracies'] = turn_accuracy_chart
        self.logger.log(stats)

        by_domain_stats: Dict[str, Dict[str, Any]] = evaluate_on_domains(running_log, self.test_set)
        flattened_domain_stats: Dict[str, Any] = {}
        for domain, domain_scores in by_domain_stats.items():
            for metric, value in domain_scores.items():
                flattened_domain_stats[f"{domain}-{metric}"] = value
        self.logger.log(flattened_domain_stats)

        self.logger.step()

        return stats
