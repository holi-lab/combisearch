"""Training-data generation: extends DSTExperiment to score few-shot examples by
sampling combinations and measuring each combination's per-turn JGA, then select
the top-scoring examples per turn."""

from __future__ import annotations

import copy
import itertools
import json
import os
import pprint
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from combisearch.evaluation.dst import evaluate
from combisearch.evaluation.error_analysis import count_prompts_from_examples
from combisearch.pipeline.dst import DSTExperiment
from combisearch.representation.types import MultiWOZDict, Turn


class DataGenerationExperiment(DSTExperiment):
    """Extends DSTExperiment to score example quality by sampling. Per test turn:
    retrieve a candidate pool, run several iterations partitioning the pool into
    disjoint sub-groups, score each example by the JGA of the sub-groups it
    appears in, then select the top-scoring examples for a final prediction."""

    def __init__(
        self, 
        runtime_config: Dict[str, Any],
        config_path: str | Path | None = None,
    ) -> None:
        """Reads the sampling params (num_sampling_iteration, num_samples,
        num_final_samples, score_type), then builds the base DSTExperiment
        components."""
        self.num_sampling_iteration = runtime_config.get("num_sampling_iteration", 3)
        self.num_samples = runtime_config.get("num_samples", 10)
        self.num_final_samples = runtime_config.get("num_final_samples") or self.num_samples
        self.score_type = runtime_config.get("score_type", "score_delta")

        super().__init__(
            runtime_config=runtime_config,
            config_path=config_path,
        )

    def run(self) -> Tuple[List[Turn], Dict[str, Any]]:
        """Runs sampling-based data generation over the test set turn by turn;
        returns (running_log, stats)."""
        selected_set: List[Turn] = self.test_set

        running_log: List[Turn] = []
        n_total: int = 0
        n_correct: int = 0
        total_acc: float = 0
        total_f1: float = 0

        for data_item_idx, data_item in tqdm(enumerate(selected_set)):
            n_total += 1

            skip, total_acc, total_f1, n_correct, n_total = self.replay_cached_prediction(
                data_item_idx, data_item, running_log, total_acc, total_f1, n_correct, n_total
            )

            if skip:
                continue

            (
                retrieved_examples,
                retrieved_example_ids,
                sampling_pool
            ) = self.retrieve_and_create_candidate_pool(data_item_idx, data_item)

            data_item = self.run_sampling_iterations(
                data_item, retrieved_examples, retrieved_example_ids,
                sampling_pool, n_total
            )

            examples = self.select_best_examples(
                data_item, retrieved_examples, retrieved_example_ids
            )

            (
                all_slot_values, predicted_slot_values,
                predicted_prior_context, best_completion, completions
             ) = self.make_final_prediction(data_item, examples)

            (
                n_correct, total_acc, total_f1
            ) = self._evaluate_and_log_turn(
                data_item=data_item, all_slot_values=all_slot_values, 
                predicted_slot_values=predicted_slot_values, 
                predicted_prior_context=predicted_prior_context,
                completions=completions, best_completion=best_completion, examples=retrieved_examples, 
                n_total=n_total, n_correct=n_correct, total_acc=total_acc, total_f1=total_f1
            )

            running_log.append(data_item)

            if data_item_idx > 20:
                with open(os.path.join(self.output_dir, "running_log.json"), 'w') as f:
                    json.dump(running_log, f, indent=2)

        stats = self._compute_final_statistics(running_log)

        return running_log, stats

    def replay_cached_prediction(self, data_item_idx, data_item, running_log, total_acc, total_f1, n_correct, n_total):
        if 'pred' in data_item:
            # Restore prediction to recorder so future turns have correct context
            if isinstance(data_item['pred'], list):
                data_item['pred'] = data_item['pred'][0]
            self.prediction_recorder.add_state(data_item, data_item['pred'])
            running_log.append(data_item)

            this_jga, this_acc, this_f1 = evaluate(prediction=data_item['pred'], gold=data_item['slot_values'])
            total_acc += this_acc
            total_f1 += this_f1

            if this_jga:
                n_correct += 1
                print("\n=====================correct!=======================")
            else:
                print("\n=====================wrong!=======================")

            self.logger.log({"current_jga": n_correct / n_total, "n_total": n_total})
            self.logger.step()
            print("\n")

            if data_item_idx > 20:
                with open(os.path.join(self.output_dir, "running_log.json"), 'w') as f:
                    json.dump(running_log, f)
        
            return True, total_acc, total_f1, n_correct, n_total
        return False, total_acc, total_f1, n_correct, n_total

    def retrieve_and_create_candidate_pool(self, data_item_idx, data_item):
        retrieved_examples: List[Turn] = []
        if self.use_gold:
            example_by_retriever = self.retriever.item_to_best_examples(
                data_item, k=self.num_examples, decoder=self.demonstration_decoder
            )
        else:
            predicted_context = self.prediction_recorder.retrieve_previous_turn_state(data_item)

            modified_item = copy.deepcopy(data_item)
            modified_item['last_slot_values'] = predicted_context

            example_by_retriever = self.retriever.item_to_best_examples(
                modified_item, k=self.num_examples, decoder=self.demonstration_decoder
            )[::-1] # High to Low

        # If multiple retrievers (e.g., BM25 + embedding), interleave their results
        if isinstance(example_by_retriever, dict) and len(example_by_retriever) > 1:
            key_iter = itertools.cycle(example_by_retriever.keys())
            while True:
                key = next(key_iter)
                try:
                    retrieved_examples.append(example_by_retriever[key].pop())
                except IndexError:
                    # One retriever exhausted, take remaining from others
                    retrieved_examples.extend(example_by_retriever[next(key_iter)][::-1])
                    break
        else:
            retrieved_examples = example_by_retriever if isinstance(example_by_retriever, list) else list(example_by_retriever.values())[0]

        retrieved_example_ids = [self.label_id(e) for e in retrieved_examples]

        assert len(set(retrieved_example_ids)) == len(retrieved_example_ids), \
            f"The {data_item_idx}-th data {data_item['ID']}, {data_item['turn_id']}. \
        The retrieved examples are not unique. set: {len(set(retrieved_example_ids))}, total: {len(retrieved_example_ids)}"

        sampling_pool = []
        for idx in range(self.num_sampling_iteration):
            random.seed(idx)
            sampling_pool += random.sample(retrieved_examples, len(retrieved_examples))

        return retrieved_examples, retrieved_example_ids, sampling_pool

    def run_sampling_iterations(
        self, 
        data_item: Turn, 
        retrieved_examples: List[Turn],
        retrieved_example_ids: List[str],
        sampling_pool: List[Turn],
        n_total: int
    ) -> Turn:
        """Scores candidates over num_sampling_iteration rounds: each round
        partitions a shuffle of the pool into disjoint num_samples-sized
        sub-groups, runs the LLM per sub-group, and accumulates delta/full JGA.
        Stores per-round predictions and scores under data_item['sampling_exp']
        and returns data_item."""
        num_sub_group = len(retrieved_examples) // self.num_samples

        data_item['sampling_exp'] = {
            'exp': [],
            'scores': []
        }

        sample_idx = 0

        print(f"\n======= {data_item['ID']} {data_item['turn_id']} =======")

        for iteration in range(self.num_sampling_iteration):
            print(f"\n======= iteration: {iteration+1} / {self.num_sampling_iteration} =======")

            prompt_list = []
            iter_log = {f"iter_{iteration}": []}
            examples = {}
            example_ids = {}

            for step, _ in enumerate(range(0, len(retrieved_examples), self.num_samples)):
                examples[step] = sampling_pool[sample_idx : sample_idx + self.num_samples]
                sample_idx += self.num_samples
                example_ids[step] = [self.label_id(x) for x in examples[step]]

                prompt_text_dict = self.get_prompt_text_dict(
                    data_item, examples[step], add_guidelines=self.add_guidelines
                )
                prompt_token_ids = self.llm_client.tokenizer.apply_chat_template(
                    prompt_text_dict[f'{len(examples[step])}-shot'],
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                prompt_list.append(prompt_token_ids)

            (
                batch_all_slot_values, batch_predicted_slot_values,
                predicted_prior_context, best_completion, completions
            ) = self.make_batched_prediction(data_item, prompt_list, save_prediction=False)

            batch_delta_jga = [
                self.compute_jga(pred_turn_slot_values, data_item['turn_slot_values'])
                for pred_turn_slot_values in batch_predicted_slot_values
            ]
            batch_full_jga = [
                self.compute_jga(all_slot_values, data_item['slot_values'])
                for all_slot_values in batch_all_slot_values
            ]

            iter_scores = self.aggregate_batch_scores(
                num_sub_group, retrieved_example_ids, example_ids, batch_delta_jga, batch_full_jga
            )

            for idx in range(len(batch_all_slot_values)):
                step_log = {
                    'pred': batch_all_slot_values[idx],
                    'pred_delta_slot_values': batch_predicted_slot_values[idx],
                    'pred_prior_context': predicted_prior_context or {},
                    'completion': best_completion[idx],
                    'prompt_counts': count_prompts_from_examples(examples[idx]),
                    'examples': [(e['ID'], e['turn_id']) for e in examples[idx]]
                }
                iter_log[f'iter_{iteration}'].append(step_log)

            print(f"few-shot best completions: {best_completion}")
            print(f"this is the {n_total - 1}th example. {data_item['ID']}_turn_{data_item['turn_id']}")
            print(f"system response: {data_item['dialog']['sys'][-1]}")
            print(f"user response: {data_item['dialog']['usr'][-1]}")
            print(f"gold turn change: {pprint.pformat(data_item['turn_slot_values'])}")
            print(f"pred turn change: {pprint.pformat(batch_predicted_slot_values)}")
            print()
            print(f"gold states: {pprint.pformat(data_item['slot_values'])}")
            print(f"pred states: {pprint.pformat(batch_all_slot_values)}")
            print(f"Delta JGA: {batch_delta_jga}")
            print(f"Full JGA: {batch_full_jga}")

            data_item['sampling_exp']['exp'].append(iter_log)
            data_item['sampling_exp']['scores'].append(iter_scores)
        
        return data_item

    def select_best_examples(
        self,
        data_item: Turn,
        retrieved_examples: List[Turn],
        retrieved_example_ids: List[str]
    ) -> List[Turn]:
        """Sums each example's per-round scores into data_item['final_scores'],
        sorts by score_type, and returns the top num_final_samples examples in
        prompt order (also stored at data_item['best_example'])."""
        data_item['final_scores'] = {}

        for score_idx, scores in enumerate(data_item['sampling_exp']['scores']):
            for key in scores:
                if key not in data_item['final_scores']:
                    data_item['final_scores'][key] = copy.deepcopy(scores[key])
                else:
                    for ex_id in retrieved_example_ids:
                        data_item['final_scores'][key][ex_id] += scores[key][ex_id]

        best_ex_id_score = sorted(
            data_item['final_scores'][self.score_type].items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.num_final_samples][::-1] # Top N examples, reverse for prompt order

        examples = []
        for (ex_id, score) in best_ex_id_score:
            for data in retrieved_examples:
                if self.label_id(data) == ex_id:
                    examples.append(data)
                    break

        data_item['best_example'] = examples

        return examples

    def make_final_prediction(
        self,
        data_item: Turn,
        examples: List[Turn]
    ) -> Tuple[MultiWOZDict, MultiWOZDict, MultiWOZDict | None, str, Dict]:
        """Builds a prompt from the selected examples, runs the model (recording
        the prediction), and returns (all_slot_values, predicted_slot_values,
        prior_context, best_completion, completions) for the single prompt."""
        prompt_text_dict = self.get_prompt_text_dict(
            data_item, examples, add_guidelines=self.add_guidelines
        )
        data_item['prompt'] = prompt_text_dict

        prompt_token_ids = self.llm_client.tokenizer.apply_chat_template(
            prompt_text_dict[f'{len(examples)}-shot'],
            add_generation_prompt=True,
            return_tensors="pt"
        )

        (
            all_slot_values, predicted_slot_values,
            predicted_prior_context, best_completion, completions
        ) = self.make_batched_prediction(data_item, [prompt_token_ids], save_prediction=True)

        return (
            all_slot_values[0],
            predicted_slot_values[0],
            predicted_prior_context,
            best_completion[0],
            completions[0],
        )

    def aggregate_batch_scores(self, num_sub_group: int, retrieved_example_ids: List[str], example_ids: Dict[int, List[str]], batch_delta_jga: List[float], batch_full_jga: List[float]) -> Dict[str, Dict[str, float]]:
        iter_scores = {}
        for key in ['occurence', 'score_delta', 'score_full', 'influence_delta', 'influence_full']:
            iter_scores[key] = {ids: 0 for ids in retrieved_example_ids}

        for idx, ex_id_list in example_ids.items():
            # For examples that were INCLUDED in this combination
            for ex_id in ex_id_list:
                iter_scores['occurence'][ex_id] += 1
                # delta = just this turn's changes, full = complete dialogue state
                for key in ['score_delta', 'score_full', 'influence_delta', 'influence_full']:
                    iter_scores[key][ex_id] += batch_delta_jga[idx] if 'delta' in key else batch_full_jga[idx]

            # For examples that were NOT included (negative influence)
            for neg_ex_id in set(retrieved_example_ids) - set(ex_id_list):
                iter_scores['influence_delta'][neg_ex_id] -= (1/(num_sub_group-1))*batch_delta_jga[idx]
                iter_scores['influence_full'][neg_ex_id] -= (1/(num_sub_group-1))*batch_full_jga[idx]

        return iter_scores
   
    def make_batched_prediction(self, data_item, prompt_token_ids_list, save_prediction=False):
        """Runs the model on a batch of prompts (greedy/beam_search only) and
        returns, per prompt, (batch_all_slot_values, batch_predicted_slot_values,
        predicted_prior_context, best_completion, completions). Records the first
        prediction when save_prediction is True."""
        batch_predicted_slot_values: List[MultiWOZDict] = []
        predicted_prior_context: MultiWOZDict | None = None
        completions = None

        try:
            method = (self.lm_decoding_config or {}).get("method", "greedy")
            if method not in {"greedy", "beam_search"}:
                raise ValueError(f"batch data generation does not support decoding method: {method}")

            batch_completion_fn = getattr(self.llm_client, "batch_greedy_lm_completion", None)
            if batch_completion_fn is not None:
                completions = batch_completion_fn(prompt_token_ids_list)
            else:
                completions = [
                    self.llm_client.greedy_lm_completion(prompt_ids)
                    for prompt_ids in prompt_token_ids_list
                ]

            # completions is a list of dicts mapping completion -> log_prob
            best_completion = [
                max(dic, key=dic.get).strip().replace('agent.state.', '')
                for dic in completions
            ]

            predicted_prior_context = self.prediction_recorder.retrieve_previous_turn_state(data_item)

            batch_predicted_slot_values = [self.completion_parser(comp, predicted_prior_context)
                                          for comp in best_completion]

        except Exception as e:
            raise RuntimeError(
                f"batch prediction failed for {data_item.get('ID')}_turn_{data_item.get('turn_id')}"
            ) from e

        batch_predicted_slot_values = [self.normalizer.normalize(s_v) for s_v in batch_predicted_slot_values]

        if self.use_gold:
            prior_dialogue_state = data_item['last_slot_values'].copy()
        else:
            prior_dialogue_state = self.prediction_recorder.retrieve_previous_turn_state(data_item).copy()

        from combisearch.representation.dialogue_state import update_dialogue_state

        batch_all_slot_values = [update_dialogue_state(prior_dialogue_state, predicted_slot_values)
                                for predicted_slot_values in batch_predicted_slot_values]

        # Handle slots with multiple values (take first option)
        batch_all_slot_values = [{k: v.split('|')[0] for k, v in all_slot_values.items()}
                                for all_slot_values in batch_all_slot_values]

        if save_prediction:
            self.prediction_recorder.add_state(data_item, batch_all_slot_values[0])

        return batch_all_slot_values, batch_predicted_slot_values, predicted_prior_context, best_completion, completions

    def compute_jga(self, prediction: MultiWOZDict, gold: MultiWOZDict):   
        for key in gold.keys():
            # if the gold value supports multiple ground truth values, and we predicted one, set the single-gold value to
            # the one we predicted.
            if '|' in gold[key]:
                gold_values = gold[key].split('|')
                if key in prediction and prediction[key] in gold_values:
                    gold[key] = prediction[key]

        # joint-goal can be computed with dict match
        return 1 if prediction == gold else 0

    def label_id(self, data_item: Turn) -> str:
        return data_item['ID'] + "_turn_" + str(data_item['turn_id'])
