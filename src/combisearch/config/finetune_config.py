"""Algorithm settings extracted from a retriever-training runtime config."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class FinetuneConfig:

    base_embedding_model_name: str
    train_file_name: str = ""
    dev_set_path: str = ""
    test_file_name: str = ""
    num_epochs: int = 15
    batch_size: int = 16
    neg_size: int = 31
    top_k: int = 10
    top_range: int = 100
    state_transformation_type: str = "ref_aware_nl"
    score_type: str = "score_delta"
    loss_type: str = "info_nce_med"
    pooling_mode: str | None = None
    filter_domains: bool = True
    use_raw_turn_state: bool = False

    @classmethod
    def from_runtime_config(cls, config: dict[str, Any]) -> "FinetuneConfig":
        base = config.get("base_embedding_model_name")
        if not base:
            raise ValueError("retriever training config requires base_embedding_model_name")

        allowed = {f.name for f in fields(cls)}
        source: dict[str, Any] = {k: v for k, v in config.items() if k in allowed}
        source["base_embedding_model_name"] = base
        source["train_file_name"] = config.get("train_fn", source.get("train_file_name", ""))
        source["dev_set_path"] = config.get("dev_fn", source.get("dev_set_path", ""))
        source["test_file_name"] = config.get("test_fn", source.get("test_file_name", ""))
        return cls(**source)
