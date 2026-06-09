from typing import Callable

import numpy as np
from tqdm import tqdm

from combisearch.representation.types import Turn
from combisearch.representation.turn_text import (
    data_item_to_string, DOMAINS, default_state_transform, StateTransformationFunction
)
from combisearch.runtime.io import read_json_from_data_dir
from combisearch.evaluation.retrieval import compute_sv_sim


class ScoredExampleDataset:

    def __init__(self, mw_json_fn: str, string_transformation: Callable[[Turn], str] = data_item_to_string,
                 state_transformation: StateTransformationFunction = default_state_transform,
                 filter_domains: bool = True,
                 use_raw_turn_state: bool = False):

        data = read_json_from_data_dir(mw_json_fn)

        self.turn_labels = []
        self.turn_utts = []
        self.turn_states = []
        self.pool = []
        self.string_transformation = string_transformation
        self.state_transformation = state_transformation

        for turn in data:
            if filter_domains and not set(turn["domains"]).issubset(set(DOMAINS)):
                continue

            history = self.string_transformation(turn)
            current_state = turn["turn_slot_values"] if use_raw_turn_state else self.state_transformation(turn)

            self.turn_labels.append(f"{turn['ID']}_turn_{turn['turn_id']}")
            self.turn_utts.append(history)
            self.turn_states.append(current_state)
            if "final_scores" not in turn:
                raise ValueError(
                    f"ScoredExampleDataset requires final_scores, but {mw_json_fn} contains raw turns. "
                    "Use CombiSearch-generated data or the legacy index-based SGD finetuning path."
                )
            self.pool.append(turn['final_scores'])

        self.n_turns = len(self.turn_labels)
        if self.n_turns == 0:
            raise ValueError(
                f"No turns loaded from {mw_json_fn}. "
                f"filter_domains={filter_domains}, use_raw_turn_state={use_raw_turn_state}"
            )
        print(f"there are {self.n_turns} turns in this dataset")


class MWDataset:
    """Turn dataset with a full pairwise state-similarity (F1) matrix, used by the
    cl retriever training (hard-negative mining).

    ``filter_domains`` and ``use_raw_turn_state`` default to MultiWOZ behavior; SGD
    training passes ``filter_domains=False`` (its services aren't in ``DOMAINS``) and
    ``use_raw_turn_state=True`` (the reference-aware transforms are MultiWOZ-only).
    """

    def __init__(self, mw_json_fn: str, just_embed_all=False, beta: float = 1.0,
                 string_transformation: Callable[[Turn], str] = data_item_to_string,
                 state_transformation: StateTransformationFunction = default_state_transform,
                 filter_domains: bool = True, use_raw_turn_state: bool = False):

        data = read_json_from_data_dir(mw_json_fn)

        self.turn_labels = []
        self.turn_utts = []
        self.turn_states = []
        self.string_transformation = string_transformation
        self.state_transformation = state_transformation

        for turn in data:
            if filter_domains and not set(turn["domains"]).issubset(set(DOMAINS)):
                continue

            history = self.string_transformation(turn)
            current_state = turn["turn_slot_values"] if use_raw_turn_state else self.state_transformation(turn)

            self.turn_labels.append(f"{turn['ID']}_turn_{turn['turn_id']}")
            self.turn_utts.append(history)
            self.turn_states.append(current_state)

        self.n_turns = len(self.turn_labels)
        print(f"there are {self.n_turns} turns in this dataset")

        if not just_embed_all:
            self.similarity_matrix = np.zeros((self.n_turns, self.n_turns))
            for i in tqdm(range(self.n_turns)):
                self.similarity_matrix[i, i] = 1
                for j in range(i, self.n_turns):
                    self.similarity_matrix[i, j] = compute_sv_sim(self.turn_states[i],
                                                                  self.turn_states[j],
                                                                  beta=beta)
                    self.similarity_matrix[j, i] = self.similarity_matrix[i, j]


def save_embeddings(model, dataset: MWDataset, output_filename: str) -> None:
    """Encode every turn utterance and save a {turn_label: embedding} npy index."""
    embeddings = model.encode(dataset.turn_utts, convert_to_numpy=True)
    output = {}
    for i in tqdm(range(len(embeddings))):
        output[dataset.turn_labels[i]] = embeddings[i:i + 1]
    np.save(output_filename, output)
