"""Stores each turn's predicted dialogue state, retrieved as prior context for later turns."""
from collections import defaultdict
from typing import Dict

from combisearch.representation.types import MultiWOZDict, Turn


class PreviousStateRecorder:
    """
    Records predictions for use in subsequent turns
    """
    states: Dict[str, Dict[int, MultiWOZDict]]

    def __init__(self):
        self.states = defaultdict(dict)

    def add_state(self, turn: Turn, slot_values: MultiWOZDict) -> None:
        """
        Store the slot values for a given turn, indexable by dialogue and turn_id
        :param turn: 
        :param slot_values: 
        :return: 
        """
        dialog_id: str = turn['ID']
        turn_id: int = turn['turn_id']
        self.states[dialog_id][turn_id] = slot_values

    def retrieve_previous_turn_state(self, turn: Turn) -> MultiWOZDict:
        """
        retrieve a recorded prediction for the turn prior to this one. Raises an exception if missing, implying we did 
        not process turns in order.
        
        :param turn: 
        :return: 
        """
        dialog_id = turn['ID']
        turn_id = turn['turn_id']
        if turn_id == 0:
            return {}
        else:
            return self.states[dialog_id][turn_id - 1]