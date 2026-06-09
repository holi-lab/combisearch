from typing import Iterator, Tuple, List

from combisearch.representation.types import Turn

from combisearch.retrieval.decoders.abstract_example_set_decoder import AbstractExampleListDecoder


class TopKDecoder(AbstractExampleListDecoder):
    def __init__(self, **kwargs) -> None:
        pass

    def select_k(self, examples: Iterator[Tuple[Turn, float]], k: int, **kwargs) -> List[Turn]:
        # first 10 in iterator are highest scoring, reverse so highest is last
        return [turn for _, (turn, score) in zip(range(k), examples)][::-1]