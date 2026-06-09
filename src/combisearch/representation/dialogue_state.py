import copy
from collections import defaultdict
from typing import Dict, List

import dictdiffer
from combisearch.representation.types import MultiWOZDict, Turn


def compute_delta(prev_dst: MultiWOZDict, dst: MultiWOZDict) -> MultiWOZDict:
    """Compute the difference between two dialogue states in flattened form."""
    delta: MultiWOZDict = {}
    for diff in dictdiffer.diff(prev_dst, dst):
        # treat changes as adds, since we're starting from an empty dict (UPSERT-like behavior)
        if diff[0] == 'change':
            diff = ('add', '', [(diff[1], diff[2][1])])
        if diff[0] == 'add':
            delta = dictdiffer.patch([diff], delta)
        elif diff[0] == 'remove':
            assert diff[1] == ''  # code won't work otherwise, but should be true of our representation
            for slot_name, slot_value in diff[2]:
                delta[slot_name] = '[DELETE]'
    return delta


def update_dialogue_state(context: MultiWOZDict, normalized_turn_parse: MultiWOZDict) -> MultiWOZDict:
    """Apply a predicted turn-delta onto the previous complete dialogue state."""
    new_dialogue_state: MultiWOZDict = copy.deepcopy(context)
    for slot_name, slot_value in normalized_turn_parse.items():
        if slot_name in new_dialogue_state and slot_value == "[DELETE]":
            del new_dialogue_state[slot_name]
        elif slot_value != "[DELETE]":
            new_dialogue_state[slot_name] = slot_value
    return new_dialogue_state


def group_by_dial_id_and_turn(turns: List[Turn]) -> Dict[str, List[Turn]]:
    result = defaultdict(dict)
    for turn in turns:
        result[turn['ID']][turn['turn_id']] = turn
    return {dial_id: [turn for index, turn in sorted(turns_dict.items(), key=lambda item: item[0])]
            for dial_id, turns_dict in result.items()}
