"""Build (label, utterance) datasets from MultiWOZ / SGD splits."""

from __future__ import annotations

from dataclasses import dataclass

from combisearch.representation.turn_text import DOMAINS, data_item_to_string
from combisearch.runtime.io import read_json_from_data_dir


MW_DOMAINS: frozenset[str] = frozenset(DOMAINS)

_DOMAIN_FILTERS: dict[str, frozenset[str] | None] = {
    "mw": MW_DOMAINS, "multiwoz": MW_DOMAINS, "mw21": MW_DOMAINS, "mw24": MW_DOMAINS,
    "sgd": None,
}


@dataclass(frozen=True)
class TurnDataset:
    """Parallel arrays — what `save_embeddings` consumes."""
    turn_labels: list[str]
    turn_utts: list[str]

    def __len__(self) -> int:
        return len(self.turn_labels)

    def __bool__(self) -> bool:
        return bool(self.turn_labels)


def domain_filter_for(dataset: str) -> frozenset[str] | None:
    key = dataset.lower()
    if key not in _DOMAIN_FILTERS:
        raise ValueError(f"Unsupported index dataset: {dataset}")
    return _DOMAIN_FILTERS[key]


def load_turn_dataset(
    train_fn: str,
    *,
    input_type: str,
    full_history: bool = False,
    only_slot: bool = False,
    gt_type: str | None = None,
    domain_filter: frozenset[str] | None = MW_DOMAINS,
) -> TurnDataset:
    raw = read_json_from_data_dir(train_fn)

    kwargs: dict[str, object] = {
        "input_type": input_type,
        "full_history": full_history,
        "only_slot": only_slot,
    }
    if gt_type is not None:
        kwargs["gt_type"] = gt_type

    labels: list[str] = []
    utts: list[str] = []
    for turn in raw:
        if domain_filter is not None and not set(turn["domains"]).issubset(domain_filter):
            continue
        labels.append(f"{turn['ID']}_turn_{turn['turn_id']}")
        utts.append(data_item_to_string(turn, **kwargs))

    return TurnDataset(turn_labels=labels, turn_utts=utts)
