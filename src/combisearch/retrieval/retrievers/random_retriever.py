"""Random-sample retriever (baseline)."""
import random
from typing import List

from combisearch.representation.types import Turn

from combisearch.retrieval.retrievers.abstract_example_retriever import ExampleRetriever
from combisearch.retrieval.decoders.abstract_example_set_decoder import AbstractExampleListDecoder


class RandomExampleRetriever(ExampleRetriever):
    def __init__(self, datasets: List[List[Turn]], **kwargs) -> None:
        self.data_items: List[Turn] = []
        for dataset in datasets:
            self.data_items.extend(dataset)

    def item_to_best_examples(self, turn: Turn, k: int = 10, decoder: AbstractExampleListDecoder = None) -> \
            List[Turn]:
        return random.choices(self.data_items, k=k)
