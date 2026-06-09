import abc

from combisearch.representation.types import MultiWOZDict


class AbstractNormalizer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def normalize(self, raw_parse: MultiWOZDict) -> MultiWOZDict:
        """
        Map a raw parse (e.g. with typos) to a normalized parse ready for system use/auto eval.
        Interface for different normalization approaches; called after parse(completion).
        """
        pass
