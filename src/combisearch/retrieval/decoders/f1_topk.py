from typing import Iterator, Tuple, List

import numpy as np

from combisearch.representation.types import Turn

from combisearch.evaluation.retrieval import compute_sv_sim

from combisearch.retrieval.decoders.abstract_example_set_decoder import AbstractExampleListDecoder
from combisearch.retrieval.retrievers.embed_based_retriever import EmbeddingRetriever
from combisearch.representation.turn_text import reference_aware_state_transform


class F1TopKDecoder(AbstractExampleListDecoder):
    """
    An example selection decoder which simply takes the k highest f1 overlaps with query examples.
    """

    from_n_possible: int
    retriever: EmbeddingRetriever

    def __init__(self, retriever: EmbeddingRetriever, from_n_possible: int =200, **kwargs) -> None:
        self.retriever = retriever
        self.from_n_possible = from_n_possible

    def select_k(self, examples: Iterator[Tuple[Turn, float]], k: int, query) -> List[Turn]:        
        all_considered_examples = [turn for _, (turn, score) in zip(range(self.from_n_possible), examples)][::-1]

        similarity_to_query = np.zeros(len(all_considered_examples))
        for idx, turn in enumerate(all_considered_examples):
            similarity_to_query[idx] = compute_sv_sim(reference_aware_state_transform(turn),
                                                      reference_aware_state_transform(query),
                                                      beta=1.0)
        
        sorted_indices = np.argsort(similarity_to_query) # ascending order (low to high)
        topk = sorted_indices[-k:]

        return [all_considered_examples[i] for i in topk]
