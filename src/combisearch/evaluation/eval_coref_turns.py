import logging
import os
import sys
from collections import defaultdict
from typing import Any, List, Tuple, Callable, Dict, Set

import numpy as np
from tqdm import tqdm

from combisearch.domain.multiwoz.ontology import normalize
from combisearch.representation.types import Turn, SlotName, SlotValue
from combisearch.runtime.io import read_json, read_json_from_data_dir
from combisearch.runtime.artifacts import get_running_logs_by_group, read_run_artifact_logs
from combisearch.runtime.env import WANDB_ENTITY, WANDB_PROJECT
from combisearch.runtime.wandb import wandb_init, wandb_log
from combisearch.evaluation.error_analysis import evaluate_logs

MWOZ_23_DATA_FILE: str = "raw/mw23/data.json"


def _normalize_slot_name(slot_name: str) -> SlotName:
    fixes: Dict[str, SlotName] = {
        'booking-day': "hotel-area",
        'hotel-day': "hotel-book day",
        'hotel-people': "hotel-book people",
        'hotel-price': "hotel-pricerange",
        'restaurant-day': "restaurant-book day",
        'restaurant-people': "restaurant-book people",
        'restaurant-price': "restaurant-pricerange",
        'restaurant-time': "restaurant-book time",
        'taxi-arrive': "taxi-arriveby",
        'taxi-depart': "taxi-departure",
        'taxi-dest': "taxi-destination",
        'taxi-leave': "taxi-leaveat",
        'train-dest': "train-destination",
        'train-people': "train-book people"
    }
    # default to current value
    return fixes.get(slot_name, slot_name)


def get_coreference_annotations(mwoz_23_data_file: str = MWOZ_23_DATA_FILE) -> Dict[
        str, Dict[int, Dict[SlotName, Dict[str, str]]]]:
    result: Dict[str, Dict[int, Any]] = defaultdict(lambda: defaultdict(dict))

    data = read_json_from_data_dir(mwoz_23_data_file)
    for dial_id, dialogue_struct in tqdm(data.items(), desc="reading MultiWOZ 2.3 co-reference annotations"):
        for i, log in enumerate(dialogue_struct['log']):
            # [(0,), (1, 2), (3, 4) ...]
            turn_id: int = (i + 1) // 2
            if 'coreference' in log:
                for domain_intent, coreferences in log['coreference'].items():
                    for coreference in coreferences:
                        domain, _ = domain_intent.lower().split('-')
                        # e.g. Day, 'same day', 'saturday', (int), (str)
                        slot_str, coref_phrase, slot_value, _, _ = coreference
                        full_slot_name: SlotName = _normalize_slot_name(f"{domain}-{slot_str.lower()}")
                        result[dial_id][turn_id][full_slot_name] = {slot_value: coref_phrase}
    return result


def eval_on_given_turns(runs: List[List[Turn]]) -> Tuple[float, float]:
    jgas: List[float] = []
    for logs in runs:
        logs = evaluate_logs(logs)
        jgas.append(np.mean([t['jga'] for t in logs]).item())
    return np.mean(jgas).item(), np.std(jgas).item()


def eval_just_coreference_slots(runs: List[List[Turn]], coreferences) -> Tuple[float, float, float]:
    accs: List[float] = []
    total_coref_slots: int = 0
    for logs in runs:
        n_correct, n_total = 0, 0
        for i, log in enumerate(logs):
            dial_id, turn_id = log['ID'], log['turn_id']
            if dial_id in coreferences and turn_id in coreferences[dial_id]:
                coreferred_slots: Dict[SlotName, Tuple[SlotValue, str]] = coreferences[dial_id][turn_id]
                for slot_name, co_ref_dict in coreferred_slots.items():
                    if slot_name not in log['slot_values']:
                        logging.info(f"coreference annotation on {dial_id}-{turn_id} is not in gold state, "
                                     f"possible annotation correction")
                        continue
                    n_total += 1
                    # if we have a prediction for the slot, check correctness, otherwise wrong
                    if slot_name in log['pred']:
                        predicted_value: SlotValue = log['pred'][slot_name]
                        gold_value = log['slot_values'][slot_name]
                        if predicted_value == gold_value:
                            n_correct += 1
                            mw23_slot_value = normalize(list(co_ref_dict.keys())[0])
                            if not predicted_value == mw23_slot_value:
                                logging.warning(f"dataset mismatch: {dial_id}-{turn_id}, {slot_name}, "
                                                f"{predicted_value}, {mw23_slot_value}")
        accs.append(n_correct / n_total)
        total_coref_slots += n_total
    return np.mean(accs).item(), np.std(accs).item(), (total_coref_slots/len(runs))


def filter_each_run(runs: List[List[Turn]], filter: Callable[[List[Turn]], List[Turn]]) -> List[List[Turn]]:
    filtered = []
    for run in runs:
        filtered.append(filter(run))
    return filtered


def only_dialogues_with_coreference(turns: List[Turn], coreferences) -> List[Turn]:
    coref_dial_ids: Set[str] = set()
    for turn in turns:
        dial_id, turn_id = turn['ID'], turn['turn_id']
        # a dialogue is co-refferent if annotated as such AND the coref slots haven't been removed
        # in subsequent annotation cleanup (e.g. MultiWOZ 2.4)
        if dial_id in coreferences and any(
            slot_name in turn['slot_values'] for slot_name in coreferences[dial_id][turn_id]
        ):
            coref_dial_ids.add(dial_id)
    return [turn for turn in turns if turn['ID'] in coref_dial_ids]


def only_turns_with_coreference(turns: List[Turn], coreferences) -> List[Turn]:
    coref_turns: List[Turn] = []
    for turn in turns:
        dial_id, turn_id = turn['ID'], turn['turn_id']
        # a dialogue is co-refferent if annotated as such AND the coref slots haven't been removed
        # in subsequent annotation cleanup (e.g. MultiWOZ 2.4)
        if dial_id in coreferences and turn_id in coreferences[dial_id] and any(
            slot_name in turn['slot_values'] for slot_name in coreferences[dial_id][turn_id]
        ):
            coref_turns.append(turn)
    return coref_turns


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise ValueError(
            "specify running_log.json path(s), a wandb group id, or comma-separated wandb run ids"
        )
    args = sys.argv[1:]
    coreferences = get_coreference_annotations()

    # Local mode (offline-reproducible): arguments are running_log.json files on disk.
    # Otherwise argv[1] is a wandb group id, or comma-separated wandb run ids.
    label: str = args[0]
    if label.endswith(".json"):
        runs: List[List[Turn]] = [read_json(path) for path in args]
    elif "," in label:
        runs = [read_run_artifact_logs(run_id) for run_id in label.split(",")]
    else:
        runs = get_running_logs_by_group(group_id=label)

    print(f"Evaluating coreference for {label} ({len(runs)} run(s))")
    full_mean, full_std = eval_on_given_turns(runs)
    print(f"Full run performance => JGA:{full_mean:.2%}, (std={full_std:.2%})")
    coref_slots_mean, coref_slots_std, n_total = eval_just_coreference_slots(runs, coreferences)
    print(f"Coreference slot performance => Acc:{coref_slots_mean:.2%}, (std={coref_slots_std:.2%})")

    wandb_init(
        project=os.environ.get(WANDB_PROJECT, "combisearch"),
        entity=os.environ.get(WANDB_ENTITY),
        name=f"coreference_result_{os.path.basename(label)}",
        notes="eval_coref_turns.py",
        group="coreference_results",
    )
    wandb_log({
        "full_mean": full_mean,
        "full_std": full_std,
        "coref_slots_mean": coref_slots_mean,
        "coref_slots_std": coref_slots_std,
        "total_coref_slots": n_total,
    })

