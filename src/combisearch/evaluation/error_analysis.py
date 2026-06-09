import json
import logging
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

from combisearch.representation.types import Turn
from tqdm import tqdm

from combisearch.runtime.artifacts import get_json_artifact_by_file_name
from combisearch.evaluation.dst import calc_prf, evaluate, F1Class


def read_running_log_file(file_name: str) -> List[Turn]:
    try:
        return get_json_artifact_by_file_name(file_name)
    except BaseException as e:
        logging.warning(e)
        with open(file_name, 'r') as f:
            return json.load(f)


def evaluate_logs(logs: List[Turn], gold_key: str = 'slot_values') -> List[Turn]:
    """Annotate each turn in a running log in place with per-turn jga/acc/turn_f1."""
    for turn in tqdm(logs, desc="evaluating turns", total=len(logs)):
        this_jga, this_acc, this_f1 = evaluate(turn['pred'], turn[gold_key])
        turn['jga'] = this_jga
        turn['acc'] = this_acc
        turn['turn_f1'] = this_f1
    return logs


def slot_level_f1(logs: List[Turn], tp_means_correct: bool = True, gold_key: str = 'slot_values') -> Dict[
    str, Tuple[Counter[F1Class], float]]:
    slot_scores: Dict[str, Counter[F1Class]] = defaultdict(Counter)
    for turn in tqdm(logs, desc="calculating slot-level F1", total=len(logs)):
        for gold_slot, gold_value in turn[gold_key].items():
            # if the slot is present in prediction, whether its a TP or FN depends on our tp_means_correct flag
            if gold_slot in turn['pred'] and (not tp_means_correct or turn['pred'][gold_slot] == gold_value):
                slot_scores[gold_slot][F1Class.TP] += 1
            else:
                slot_scores[gold_slot][F1Class.FN] += 1
        for pred_slot, pred_value in turn['pred'].items():
            if pred_slot not in turn[gold_key]:
                slot_scores[pred_slot][F1Class.FP] += 1
    return {k: (v, calc_prf(v).f1) for k, v in slot_scores.items()}


def count_prompts_from_examples(examples: List[Turn]) -> Counter[str]:
    """Count, over example turns used in a prompt, how many demonstrate each slot (by normalized slot name)."""
    prompt_counter: Counter = Counter()
    for demo_turn in examples:
        for slot_name in demo_turn['turn_slot_values']:
            prompt_counter[slot_name] += 1
    return prompt_counter
