# data.json from extracted files in MultiWOZ 2.3 dataset repo:
# https://github.com/lexmen318/MultiWOZ-coref/raw/main/MultiWOZ2_3.zip

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal

from tqdm import tqdm


# Resolve all data paths relative to this file (data/code/build_coref_only_dataset.py)
# so the script runs regardless of the caller's cwd. Layout:
#   data/code/<this file>
#   data/raw/mw23/data.json           ← MultiWOZ 2.3 coref annotations (input)
#   data/mw24_100p_dev.json ← MultiWOZ 2.4 dev set         (input)
#   data/mw24_coref_only_dials_dev.json                    (output)
_DATA_DIR = Path(__file__).resolve().parent.parent
MWOZ_23_DATA_FILE: Path = _DATA_DIR / "raw" / "mw23" / "data.json"
_MW24_DEV_INPUT: Path = _DATA_DIR / "mw24_100p_dev.json"
_COREF_OUTPUT: Path = _DATA_DIR / "mw24_coref_only_dials_dev.json"

# SlotNames are complete names including a domain and a slot, separated by a dash. e.g. "hotel-area"
SlotName = Literal["attraction-area", "attraction-name", "attraction-type", "bus-day", "bus-departure",
                   "bus-destination", "bus-leaveat", "hospital-department", "hotel-area", "hotel-book day",
                   "hotel-book people", "hotel-book stay", "hotel-internet", "hotel-name", "hotel-parking",
                   "hotel-pricerange", "hotel-stars", "hotel-type", "restaurant-area", "restaurant-book day",
                   "restaurant-book people", "restaurant-book time", "restaurant-food", "restaurant-name",
                   "restaurant-pricerange", "taxi-arriveby", "taxi-departure", "taxi-destination", "taxi-leaveat",
                   "train-arriveby", "train-book people", "train-day", "train-departure", "train-destination",
                   "train-leaveat"]

def _normalize_slot_name(slot_name: str) -> str:
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


def get_coreference_annotations(mwoz_23_data_file: Path | str = MWOZ_23_DATA_FILE) -> Dict[
    str, Dict[int, Dict[str, Dict[str, str]]]]:
    result: Dict[str, Dict[int, Any]] = defaultdict(lambda: defaultdict(dict))

    with open(mwoz_23_data_file, 'r') as f:
        data = json.load(f)
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
                        full_slot_name: str = _normalize_slot_name(f"{domain}-{slot_str.lower()}")
                        result[dial_id][turn_id][full_slot_name] = {slot_value: coref_phrase}
    return result


def group_by_dial_id_and_turn(turns: List[Any]) -> Dict[str, List[Any]]:
    result = defaultdict(dict)
    for turn in turns:
        result[turn['ID']][turn['turn_id']] = turn
    return {dial_id: [turn for index, turn in sorted(turns_dict.items(), key=lambda item: item[0])]
            for dial_id, turns_dict in result.items()}


if __name__ == '__main__':
    with _MW24_DEV_INPUT.open('r') as f:
        dev_set = json.load(f)

    coreferences: Dict[str, Any] = get_coreference_annotations()
    only_coref_turns = [dial for dial in dev_set if dial['ID'] in coreferences]
    dev_by_dial_ids = group_by_dial_id_and_turn(dev_set)
    coref_only_by_dial_ids = group_by_dial_id_and_turn(only_coref_turns)
    for k in coref_only_by_dial_ids:
        assert len(coref_only_by_dial_ids[k]) == len(dev_by_dial_ids[k])
    with _COREF_OUTPUT.open("w") as f:
        json.dump(only_coref_turns, f)
