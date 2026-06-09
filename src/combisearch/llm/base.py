import abc
from typing import List, Dict, Callable


class AbstractLMClient(metaclass=abc.ABCMeta):
    """Interface for the generative LM used in a DST experiment."""

    @abc.abstractmethod
    def __init__(self, stop_sequences: List[str] = None, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def greedy_lm_completion(self, prompt_text: str) -> Dict[str, float]:
        """Greedy completion for the prompt. Returns {completion_text: logprob}, excluding prompt tokens."""
        pass

    @abc.abstractmethod
    def top_p_lm_completion(self, prompt_text: str, top_p: float = 0.9, n: int = 5, best_of: int = 10,
                            max_tokens: int = 120, **kwargs) -> Dict[str, float]:
        """Top-p sampled completions for the prompt. Returns {completion_text: logprob}, excluding prompt tokens."""
        pass

    @abc.abstractmethod
    def get_completion_log_probabilities(self, prompt_text: str, completion: str,
                                         token_log_probs_telemetry_hook: Callable[[List[float]], None] = None) -> List[
        float]:
        pass
