import abc
from typing import List

from numpy.typing import NDArray

from combisearch.retrieval.turn_labels import TurnLabel, label_to_data_item, turn_label
from combisearch.representation.types import Turn

from combisearch.retrieval.decoders.abstract_example_set_decoder import AbstractExampleListDecoder


class ExampleRetriever(abc.ABC):

    data_items: List[Turn]

    @abc.abstractmethod
    def __init__(self, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def item_to_best_examples(self, turn: Turn, k: int = 10, decoder: AbstractExampleListDecoder = None) \
            -> List[Turn]:
        pass

    def data_item_to_label(self, turn: Turn) -> TurnLabel:
        return turn_label(turn)

    def label_to_data_item(self, label: TurnLabel) -> Turn:
        return label_to_data_item(self.data_items, label)

    def label_to_search_embedding(self, label: TurnLabel) -> NDArray:
        if label not in self.retriever.label_to_idx:
            raise KeyError(f"{label} not in search index")
        return self.retriever.emb_values[self.retriever.label_to_idx[label]]


class MockRetriever(ExampleRetriever):
    def __init__(self, **kwargs) -> None:
        pass

    def item_to_best_examples(self, turn: Turn, k: int = 10, decoder: AbstractExampleListDecoder = None) -> \
    List[Turn]:
        raise NotImplementedError("misconfigured experiment: not expecting this method to be called!")
