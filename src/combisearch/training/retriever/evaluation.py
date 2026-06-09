"""SentenceTransformers evaluators.
"""

from __future__ import annotations

import logging
import pprint
from statistics import mean
from typing import Dict, List

import numpy.typing as npt
from sentence_transformers import models
from sentence_transformers.evaluation import SentenceEvaluator
from tqdm import tqdm

from combisearch.evaluation.dst import evaluate
from combisearch.evaluation.retrieval import compute_sv_sim, evaluate_retriever_on_dataset
from combisearch.retrieval.decoders.hybrid_decoder import HybridDecoder
from combisearch.retrieval.retrievers.embed_based_retriever import EmbeddingRetriever
from combisearch.retrieval.retrievers.hybrid_retriever import HybridRetriever
from combisearch.runtime.io import read_json_from_data_dir
from combisearch.runtime.wandb import wandb_log
from combisearch.representation.types import SlotName
from combisearch.training.retriever.data import ScoredExampleDataset

logger = logging.getLogger(__name__)


class RetrievalEvaluator(SentenceEvaluator):
    """Adapted from YushiHu/IC-DST."""

    train_fn: str
    dev_fn: str
    index_set: ScoredExampleDataset

    def __init__(
        self,
        train_fn: str,
        dev_fn: str,
        index_set: ScoredExampleDataset,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        string_transformation=None,
    ):
        self.train_fn = train_fn
        self.dev_fn = dev_fn
        self.index_set = index_set
        self.batch_size = batch_size
        self.string_transformation = string_transformation
        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (logging.INFO, logging.DEBUG)
        self.show_progress_bar = show_progress_bar
        self.retriever: EmbeddingRetriever | None = None

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        logger.info("Evaluating")
        scores = self.compute_metrics(model)
        wandb_log({
            "epoch": epoch,
            "step": steps,
            **{"dev_" + k: v for k, v in scores.items()},
        })
        return scores["top_5_turn_slot_value_f_score"] + scores["top_5_turn_slot_name_f_score"]

    def compute_metrics(self, model: models.Transformer) -> Dict[str, float]:
        train_set = read_json_from_data_dir(self.train_fn)
        dev_set = read_json_from_data_dir(self.dev_fn)

        embeddings: Dict[str, List[npt.NDArray]] = {
            k: v for k, v in zip(
                self.index_set.turn_labels,
                model.encode(self.index_set.turn_utts, convert_to_numpy=True),
            )
        }

        retriever = EmbeddingRetriever(
            datasets=[train_set],
            model_path="",
            model=model,
            search_embeddings=embeddings,
            sampling_method="pre_assigned",
            string_transformation=self.string_transformation,
        )

        turn_sv, turn_s, dial_sv, dial_s = evaluate_retriever_on_dataset(dev_set, retriever)
        return {
            "top_5_turn_slot_value_f_score": turn_sv,
            "top_5_turn_slot_name_f_score": turn_s,
            "top_5_hist_slot_value_f_score": dial_sv,
            "top_5_hist_slot_name_f_score": dial_s,
        }


class DstJgaEvaluator(SentenceEvaluator):
    """Adapted from refpydst's ``OptimalEvaluator``.

    Runs a vLLM Llama-3-8B-Instruct DST predictor against the dev set after
    each training step. Returns the JGA so the trainer can log it.

    Used by CombiSearch retriever training to match the original ablation
    setup. The returned JGA drives best-checkpoint selection via the
    SaveModelCallback in finetune.py (best-by-dev-evaluator).
    """
    train_fn: str
    dev_fn: str
    index_set: ScoredExampleDataset

    def __init__(
        self,
        train_fn: str,
        dev_fn: str,
        index_set: ScoredExampleDataset,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        string_transformation=None,
        vllm_enforce_eager: bool = True,
        vllm_gpu_memory_utilization: float = 0.4,
    ):
        self.train_fn = train_fn
        self.dev_fn = dev_fn
        self.index_set = index_set
        self.batch_size = batch_size
        self.string_transformation = string_transformation
        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (logging.INFO, logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        from vllm import LLM

        self.LLM = LLM(
            model='meta-llama/Meta-Llama-3-8B-Instruct',
            enforce_eager=vllm_enforce_eager,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
        )
        self.tokenizer = self.LLM.get_tokenizer()
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        self.stop_sequences = ['--', '\n', ';', '#']

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        logger.info("Evaluating")
        scores = self.compute_metrics(model)

        wandb_log({
            "epoch": epoch,
            "step": steps,
            **{"dev_" + k: v for k, v in scores.items()},
        })
        return scores['jga']
    
    def compute_metrics(self, model: models.Transformer) -> Dict[str, float]:
        train_set = read_json_from_data_dir(self.train_fn)
        dev_set = read_json_from_data_dir(self.dev_fn)

        embeddings: Dict[str, List[npt.NDArray]] = {
            k: v for k, v in zip(
                self.index_set.turn_labels,  # key is a dialogue id and turn id combined as a string
                model.encode(self.index_set.turn_utts, convert_to_numpy=True)  # value is a singleton list w/ embedding
            )
        }

        retriever_args = {
            "state_transformation": "ref_aware",
            "bm25_input_kwargs": {"input_type": "dialog_context"},
            "sbert_input_kwargs": {"input_type": "dialog_context"}
        }

        retriever = HybridRetriever(
            datasets=[train_set],
            model_path="",
            model=model,
            search_embeddings=embeddings,
            sampling_method="pre_assigned",
            string_transformation=self.string_transformation,
            **retriever_args
        )

        decoder_config = {
            "decoder_type": "hybrid",
            "operation":"multiply",
            "zscore":True,
            "decoding_logic": "multiply_top_k",
            "from_n_possible": 10
        }

        demonstration_decoder = HybridDecoder(
                retriever=retriever,
                from_n_possible=decoder_config.get('from_n_possible', 100),
                discount_factor=decoder_config.get('discount_factor',0.2),
                operation=decoder_config.get('operation', 'multiply'),
                zscore=decoder_config.get('zscore', True),
                decoding_logic=decoder_config.get('decoding_logic', 'top_k_round_robin')
        )

        
        n_correct = 0
        turn_value_sims = []
        turn_slot_sims = []
        all_value_sims = []
        all_slot_sims = []

        for idx, data_item in enumerate(tqdm(dev_set)):
            if data_item['turn_id'] == 0:
                prev_item = {}

            examples = retriever.item_to_best_examples(
                data_item, k={"BM25": 5,"SBERT": 5}, decoder=demonstration_decoder)

            prompt_text = self.get_nl_chat_prompt(
                    data_item, examples, given_context=prev_item, n_examples=None,
                    reverse_x_and_y=False, use_null_data_item=False, detailed_state_string=True
            )

            from vllm import SamplingParams

            prompt_ids = self.tokenizer.apply_chat_template(
                prompt_text, add_generation_prompt=True, return_tensors="pt"
            )
            prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=False)

            sampling_params = SamplingParams(
                n=1,
                best_of=1,
                max_tokens=120,
                temperature=0,
                stop=self.stop_sequences,
                stop_token_ids=self.terminators,
            )

            result = self.LLM.generate(prompts, sampling_params=sampling_params)
            
            completions = {result[0].outputs[0].text: 1}

            best_completion = max(completions, key=completions.get)
            best_completion = best_completion.strip().replace('agent.state.', '')
            data_item['completion'] = best_completion

            parsed_pred_delta_bs = self.parse_nl_completion(data_item['completion'])

            prev_item = data_item

            turn_value_sim, turn_slot_sim, all_value_sim, all_slot_sim = self.evaluate_single_query_ex(data_item, examples, beta=1.0)
            turn_value_sims.append(turn_value_sim)
            turn_slot_sims.append(turn_slot_sim)
            all_value_sims.append(all_value_sim)
            all_slot_sims.append(all_slot_sim)

            print(f"\nfew-shot completions: {completions}")
            print(f"few-shot best completion: {best_completion}")
            print(f"this is the {idx+1}th example. {data_item['ID']}_turn_{data_item['turn_id']}")
            print(f"system response: {data_item['dialog']['sys'][-1]}")
            print(f"user response: {data_item['dialog']['usr'][-1]}")
            print(f"pred turn change: {pprint.pformat(parsed_pred_delta_bs)}")
            print(f"gold turn change: {pprint.pformat(data_item['turn_slot_values'])}")
            
            this_jga, this_acc, this_f1 = evaluate(parsed_pred_delta_bs, data_item['turn_slot_values'])
            
            if this_jga:
                n_correct += 1
                print("\n=====================correct!=======================")
            else:
                print("\n=====================wrong!=======================")
            print("current_jga", n_correct / (idx+1), " n_correct", n_correct, " n_total", idx+1)
            print("\n")

        def _safe_mean(lst):
            return mean(lst) if lst else 0.0

        turn_sv, turn_s, dial_sv, dial_s = _safe_mean(turn_value_sims), _safe_mean(turn_slot_sims), _safe_mean(all_value_sims), _safe_mean(all_slot_sims)

        return {'jga': n_correct / len(dev_set) if len(dev_set) > 0 else 0.0, 'top_5_turn_slot_value_f_score': turn_sv, "top_5_turn_slot_name_f_score": turn_s,
                "top_5_hist_slot_value_f_score": dial_sv, "top_5_hist_slot_name_f_score": dial_s}
    
    def evaluate_single_query_ex(self, turn, examples, beta: float = 1):
        query_turn_sv = turn['turn_slot_values']
        query_sv = turn['slot_values']

        turn_value_sims = []
        turn_slot_sims = []
        all_value_sims = []
        all_slot_sims = []

        if not examples:
            return 0.0, 0.0, 0.0, 0.0

        for ex in examples:
            this_turn_sv = ex['turn_slot_values']
            this_sv = ex['slot_values']

            turn_value_sim, turn_slot_sim = compute_sv_sim(
                query_turn_sv, this_turn_sv, onescore=False, beta=beta)
            all_value_sim, all_slot_sim = compute_sv_sim(query_sv, this_sv, onescore=False, beta=beta)

            turn_value_sims.append(turn_value_sim)
            turn_slot_sims.append(turn_slot_sim)
            all_value_sims.append(all_value_sim)
            all_slot_sims.append(all_slot_sim)

        return mean(turn_value_sims), mean(turn_slot_sims), mean(all_value_sims), mean(all_slot_sims)
    
        
    def get_nl_chat_prompt(self, data_item, examples, given_context=None, n_examples: int = None,
                            reverse_x_and_y: bool = False, use_null_data_item: bool = False,
                            detailed_state_string: bool = True, add_guidelines:bool = True) -> str:
        system_msg = "**Task:** You are an expert in Dialogue State Tracking (DST) focused on managing and updating the dialogue state change based on system-user interactions. "
        system_msg += "The dialogue state represents the user's preferences and booking details across different domains: Hotel, Train, Attraction, Restaurant, and Taxi.\n\n"
        msg = [{"role": "system", "content": system_msg}]
        max_n_examples: int = n_examples is not None and n_examples or len(examples)

        # in case for zero-shot learning
        if max_n_examples > 0:
            for example_id, example in enumerate(examples[-max_n_examples:]):
                prefix_msg = f"\n**Example {example_id + 1} of Dialogue State Change Update Task:**\n"
                
                # remove multiple choice in last slot values
                last_slot_values = {s: v.split('|')[0] for s, v in example['last_slot_values'].items()}
                turn_slot_values = {s: v.split('|')[0] for s, v in example['turn_slot_values'].items()}

                last_sys_utt = example['dialog']['sys'][-1]
                if last_sys_utt == 'none':
                    last_sys_utt = ''
                
                state_string = '    **Previous Belief State (Before the Latest User Interaction):** \n'
                state_string += f"        {last_slot_values}\n\n"

                turn_msg = state_string + "    **Latest Conversation Between System and User:** \n"
                if last_sys_utt:
                    turn_msg += '        **System:** "' + last_sys_utt + '"\n'

                turn_msg += '        **User:** "' + example['dialog']['usr'][-1] + '"\n\n'

                turn_msg += '    **Instructions:**\n'
                turn_msg += '        - Based on the user\'s latest input, update the belief state by correctly identifying and filling in the relevant domain(s), slot(s) and value(s).\n'
                turn_msg += '        - Provide your output strictly in the Required Output Format below.\n\n'
                turn_msg += "    **Required Output Format:**\n"
                turn_msg += '        { "DOMAIN_1-SLOT_1": "VALUE_1", "DOMAIN_2-SLOT_2": "VALUE_2", ... }\n'
                bs_msg = f"            {turn_slot_values}\n"
                if not reverse_x_and_y:
                    msg.append({"role": "user","content": prefix_msg+turn_msg})
                    msg.append({"role": "assistant","content": bs_msg})
                else:
                    msg.append({"role": "user", "content": prefix_msg+bs_msg})
                    msg.append({"role": "assistant","content": turn_msg})

        prefix_msg = f"\n**Example {max_n_examples + 1} of Dialogue State Change Update Task:**\n"
        if given_context is None:
            last_slot_values = {s: v.split('|')[0] for s, v in data_item['last_slot_values'].items()}
        else:
            last_slot_values = given_context
        
        last_sys_utt = data_item['dialog']['sys'][-1]
        if last_sys_utt == 'none':
            last_sys_utt = ''
        
        state_string = '    **Previous Belief State (Before the Latest User Interaction):** \n'
        state_string += f"        {last_slot_values}\n\n"

        turn_msg = ''
        if not use_null_data_item:
            turn_msg = state_string + "    **Latest Conversation Between System and User:** \n"
            if last_sys_utt:
                turn_msg += '        **System:** "' + last_sys_utt + '"\n'
            turn_msg += '        **User:** "' + data_item['dialog']['usr'][-1] + '"\n\n'
            turn_msg += '    **Instructions:**\n'
            turn_msg += '        - Based on the user\'s latest input, update the belief state by correctly identifying and filling in the relevant domain(s), slot(s) and value(s).\n'
            turn_msg += '        - Provide your output strictly in the Required Output Format below.\n\n'
            turn_msg += "    **Required Output Format:**\n"
            turn_msg += '          { "DOMAIN_1-SLOT_1": "VALUE_1", "DOMAIN_2-SLOT_2": "VALUE_2", ... }\n'
        msg.append({"role": "user","content": prefix_msg+turn_msg})
        if add_guidelines:
            tmp_msg_content = msg[1]['content']
            msg[1]['content'] = '**Guidelines:**\n\n'
            msg[1]['content'] +='    1. **Hotel:**\n'
            msg[1]['content'] +='        - **Slots:** name (string), pricerange (PriceRange), type (HotelType), parking (Option), book stay (integer), book day (DayOfWeek), book people (integer), area (Area), stars (integer between 0 and 5 or "dontcare"), internet (Option).\n'
            msg[1]['content'] +='        - **Valid Values:**\n'
            msg[1]['content'] +='            - PriceRange: "dontcare", "cheap", "moderate", "expensive".\n'
            msg[1]['content'] +='            - HotelType: "hotel", "guest house", "dontcare".\n'
            msg[1]['content'] +='            - Option: "yes", "no", "dontcare".\n'
            msg[1]['content'] +='            - DayOfWeek: "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday".\n'
            msg[1]['content'] +='            - Area: "dontcare", "centre", "east", "north", "south", "west".\n\n'
            msg[1]['content'] +='    2. **Train:**\n'
            msg[1]['content'] +='        - **Slots:** destination (string), departure (string), day (DayOfWeek), book people (integer), leaveat (hh:mm or "dontcare"), arriveby (hh:mm or "dontcare").\n\n'
            msg[1]['content'] +='    3. **Attraction:**\n'
            msg[1]['content'] +='        - **Slots:** name (string), area (Area), type (AttractionType).\n'
            msg[1]['content'] +='        - **Valid Values:**\n'
            msg[1]['content'] +='            - AttractionType: "architecture", "boat", "church", "cinema", "college", "concert hall", "entertainment", "hotspot", "multiple sports", "museum", "nightclub", "park", "special", "swimming pool", "theatre", "dontcare".\n\n'
            msg[1]['content'] +='    4. **Restaurant:**\n'
            msg[1]['content'] +='        - **Slots:** name (string), food (string), pricerange (PriceRange), area (Area), book time (hh:mm or "dontcare"), book day (DayOfWeek), book people (integer).\n\n'
            msg[1]['content'] +='    5. **Taxi:**\n'
            msg[1]['content'] +='        - **Slots:** destination (string), departure (string), leaveat (hh:mm or "dontcare"), arriveby (hh:mm or "dontcare").\n\n' 
            msg[1]["content"] += tmp_msg_content
        return msg


    def parse_nl_completion(self, nl_completion: str, state=None,
                            exceptions_are_empty: bool = True, **kwargs):
        bs_dict = {}
        try:
            full_statement = nl_completion.strip()
            full_statement = full_statement.replace('{', '').replace('}', '')

            for d_s_v in full_statement.split(','):
                d_s = eval(d_s_v.split(":")[0])
                v = d_s_v.split(":")[1:]
                v = str(eval(":".join(v)))
                try:
                    assert f"{d_s}" in SlotName.__args__
                except AssertionError:
                    print(f"{d_s} not found in SlotName")
                    continue
                bs_dict[d_s] = v

            return bs_dict
        except Exception:
            return bs_dict