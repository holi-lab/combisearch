"""Declarative list of indexes to build."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from combisearch.config.paths import (
    MW21_100P_TRAIN,
    SGD_5P_TRAIN,
    embedding_model_tag,
    resolve_canonical_path,
)


@dataclass(frozen=True)
class IndexSpec:
    sub_dir: str
    dataset: str
    train_fn: str
    input_type: str
    gt_type: str | None = None

    def output_dir(self, model_name: str) -> Path:
        return self.output_file(embedding_model_tag(model_name)).parent

    def output_file(self, encoder_tag: str) -> Path:
        return resolve_canonical_path(
            f"outputs/indexes/{encoder_tag}/{self.sub_dir}/train_index.npy"
        )


SPECS: tuple[IndexSpec, ...] = (
    IndexSpec("gt_delta_bs",        "mw",  MW21_100P_TRAIN, "gt_delta_bs"),
    IndexSpec("gt_delta_slot",      "mw",  MW21_100P_TRAIN, "gt_delta_slot"),
    IndexSpec("dialog_context_bs",  "mw",  MW21_100P_TRAIN, "dialog_context", gt_type="gt_delta_bs"),
    IndexSpec("dialog_context",     "mw",  MW21_100P_TRAIN, "dialog_context"),
    IndexSpec("dialog",             "mw",  MW21_100P_TRAIN, "dialog"),
    IndexSpec("sgd/5p/gt_delta_bs", "sgd", SGD_5P_TRAIN,    "gt_delta_bs"),
)


def select_specs(only: list[str]) -> list[IndexSpec]:
    if not only:
        return list(SPECS)
    return [
        spec for spec in SPECS
        if any(spec.sub_dir == term or spec.sub_dir.startswith(f"{term}/") for term in only)
    ]
