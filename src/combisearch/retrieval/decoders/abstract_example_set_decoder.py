import abc
from collections.abc import Iterator

from combisearch.representation.types import Turn


class AbstractExampleListDecoder(abc.ABC):
    """Selects the best K examples from an iterator of (example, score) pairs in descending score order.

    Returns them in ascending order, so the highest-scoring example is last (nearest the inference turn in the prompt).
    """

    @abc.abstractmethod
    def __init__(self, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def select_k(self, examples: Iterator[Turn, float], k: int):
        pass
