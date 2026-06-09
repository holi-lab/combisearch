from collections.abc import Iterable
from typing import Any, Callable

from combisearch.representation.types import Turn
from combisearch.representation.turn_text import (
    data_item_to_string,
    get_string_transformation_by_type,
)
from combisearch.retrieval.sampling import (
    pre_assigned,
    random_by_dialog,
    random_by_turn,
    sgd_pre_assigned,
)


def flatten_datasets(datasets: Iterable[list[Turn]]) -> list[Turn]:
    data_items: list[Turn] = []
    for dataset in datasets:
        data_items.extend(dataset)
    return data_items


def input_kwargs_from_config(config: dict[str, Any], *, default_input_type: str) -> dict[str, Any]:
    return {
        "full_history": config.get("full_history", False),
        "input_type": config.get("input_type", default_input_type),
        "only_slot": config.get("only_slot", False),
        "gt_type": config.get("gt_type", None),
    }


def make_string_transformation(
    string_transformation: str | Callable[[Turn], str] | None,
    input_kwargs: dict[str, Any],
) -> Callable[[Turn], str]:
    if isinstance(string_transformation, str):
        return get_string_transformation_by_type(string_transformation)
    if string_transformation is not None:
        return string_transformation

    def default_transformation(turn: Turn) -> str:
        return data_item_to_string(turn, **input_kwargs)

    return default_transformation


def apply_sampling_method(
    search_embeddings: dict,
    data_items: list[Turn],
    *,
    sampling_method: str,
    ratio: float,
    pre_assigned_empty_if_none: bool = False,
    allow_sgd_pre_assigned: bool = False,
) -> dict:
    """Dispatch the configured embedding-sampling strategy and return the resulting dict.

    Why: three retrievers had byte-identical if/elif chains over the same five sampling
    methods, differing only in whether ``pre_assigned`` defaulted ``empty_if_none`` to
    ``True`` and whether ``sgd_pre_assigned`` was reachable.
    """
    if sampling_method == "none":
        return search_embeddings
    if sampling_method == "random_by_dialog":
        return random_by_dialog(search_embeddings, ratio=ratio)
    if sampling_method == "random_by_turn":
        return random_by_turn(search_embeddings, ratio=ratio)
    if sampling_method == "pre_assigned":
        return pre_assigned(search_embeddings, data_items, empty_if_none=pre_assigned_empty_if_none)
    if allow_sgd_pre_assigned and sampling_method == "sgd_pre_assigned":
        return sgd_pre_assigned(search_embeddings, data_items)
    raise ValueError("selection method not supported")
