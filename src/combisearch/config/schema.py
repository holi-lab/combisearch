"""Flat experiment config schema.

Validation policy (intentional):

- **Required keys** per stage are checked (fail fast on missing inputs).
- **Discriminator values** (``retriever_type``, ``decoder_type``, ``prompt_format``,
  ``loss_type``) are validated against the registry — typos here would
  silently route to the wrong code path.
- **Extension keys** (anything else, including ``retriever_args.*``,
  ``decoder_config.*``, ``lm_decoding_config.*``) are *open*: unknown keys
  pass through. Each consumer pops only what it needs.

Add a key by using it in code; add a new retriever/decoder type by
registering it in the registry.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from combisearch.config.paths import apply_runtime_output_path


STAGE_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "data_generation": (
        "train_fn", "test_fn", "llm_engine", "prompt_format",
        "retriever_type", "retriever_dir", "retriever_args",
        "num_examples",
        "score_type", "num_sampling_iteration", "num_samples",
    ),
    "dst_run": (
        "train_fn", "test_fn", "llm_engine", "prompt_format",
        "retriever_type", "retriever_dir", "retriever_args", "num_examples",
    ),
    "retriever_training": (
        "train_fn", "dev_fn", "test_fn",
        "base_embedding_model_name",
        "num_epochs", "batch_size", "neg_size", "top_k", "top_range",
        "state_transformation_type", "score_type", "loss_type",
    ),
}


KNOWN_RETRIEVER_TYPES = frozenset({
    "embeddingretriever", "embedding", "sbert",
    "bm25", "hybrid", "sgd",
    "random", "no_retriever",
})

KNOWN_DECODER_TYPES = frozenset({
    "top_k", "max_emb_distance", "bm25_max_emb_distance",
    "hybrid", "sampling_top_k", "f1_top_k",
})

KNOWN_PROMPT_FORMATS = frozenset({
    "python-prompt", "python-prompt-NULL",
    "nl-prompt-chat",
    "sgd_plain_text", "SGD",
})

KNOWN_LOSS_TYPES = frozenset({
    "info_nce_med", "info_nce_hard", "info_nce_easy", "contrastive",
    "cl",
})


def _check_discriminator(name: str, value: Any, allowed: frozenset[str]) -> None:
    if value is None:
        return
    text = str(value)
    if text not in allowed and text.lower() not in {a.lower() for a in allowed}:
        raise ValueError(
            f"unknown {name}: {value!r}. Known: {sorted(allowed)}. "
            f"To add a new {name}, register it in combisearch.retrieval.registry "
            f"(or the appropriate dispatch) and add it to KNOWN_{name.upper()}S."
        )


def to_plain_dict(value: Any) -> Any:
    """Convert dataclasses / nested dicts into plain Python containers."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_dict(v) for v in value]
    return copy.deepcopy(value)


def normalize_config(config: Any) -> dict[str, Any]:
    """Coerce a config-like object into a flat ``dict`` and validate discriminators.

    Unknown top-level keys pass through — this is intentional. Add a key by
    using it in code; the validation here is just for things that route
    silently (discriminator values).
    """
    flat = to_plain_dict(config or {})
    if not isinstance(flat, dict):
        raise TypeError(f"config must resolve to a dict, got {type(config).__name__}")

    for nested_key in ("retriever_args", "decoder_config", "lm_decoding_config"):
        value = flat.get(nested_key)
        if value is None:
            flat[nested_key] = {}
        elif not isinstance(value, dict):
            raise TypeError(f"{nested_key} must be a dictionary, got {type(value).__name__}")

    _check_discriminator("retriever_type", flat.get("retriever_type"), KNOWN_RETRIEVER_TYPES)
    _check_discriminator("decoder_type", flat["decoder_config"].get("decoder_type"), KNOWN_DECODER_TYPES)
    _check_discriminator("prompt_format", flat.get("prompt_format"), KNOWN_PROMPT_FORMATS)
    _check_discriminator("loss_type", flat.get("loss_type"), KNOWN_LOSS_TYPES)

    num_examples = flat.get("num_examples")
    if num_examples is not None and not isinstance(num_examples, (int, dict)):
        raise TypeError(
            f"num_examples must be int or dict (e.g. {{'BM25': 5, 'SBERT': 5}}), "
            f"got {type(num_examples).__name__}"
        )

    return flat


def validate_stage_config(config: dict[str, Any], stage: str | None) -> None:
    """Fail early if the config is missing keys required by its stage."""
    if stage is None:
        return
    required = STAGE_REQUIRED_KEYS.get(stage)
    if required is None:
        raise ValueError(f"unknown config stage: {stage}")
    missing = [k for k in required if k not in config or config.get(k) in (None, "")]
    if missing:
        raise ValueError(f"{stage} config is missing required key(s): {missing}")


def prepare_runtime_config(
    config: Any,
    *,
    config_path: str | Path | None,
    stage: str,
) -> dict[str, Any]:
    """Normalize the config and inject the canonical ``output_dir``."""
    flat = normalize_config(config)
    validate_stage_config(flat, stage)

    if config_path is None:
        if not flat.get("output_dir"):
            raise ValueError("config_path or output_dir is required to prepare runtime config")
        return flat

    flat["_config_path"] = str(config_path)
    flat = apply_runtime_output_path(flat, config_path, stage=stage)
    return flat


_NO_RETRIEVER_TYPES = {"no_retriever", "random"}
_DENSE_RETRIEVER_TYPES = {"hybrid", "embeddingretriever", "embedding", "sbert", "sgd"}


def _num_examples_count(config: dict[str, Any], key: str) -> int | None:
    value = config.get("num_examples")
    if not isinstance(value, dict):
        return None
    for candidate in (key, key.upper(), key.lower()):
        if candidate in value:
            try:
                return int(value[candidate])
            except (TypeError, ValueError):
                return None
    return None


def is_bm25_only_config(config: dict[str, Any]) -> bool:
    bm25 = _num_examples_count(config, "BM25")
    sbert = _num_examples_count(config, "SBERT")
    return bm25 is not None and bm25 > 0 and sbert == 0


def declared_retrieval_strategy(config: dict[str, Any]) -> str:
    if is_bm25_only_config(config):
        return "bm25"
    return str(config.get("retriever_type") or "").strip().lower()


def requires_declared_dense_retriever(config: dict[str, Any]) -> bool:
    """Whether runtime should validate dense retriever model/index paths."""
    retriever_type = str(config.get("retriever_type") or "").strip().lower()
    retriever_dir = str(config.get("retriever_dir") or "").strip().lower()

    if retriever_type in _NO_RETRIEVER_TYPES or retriever_dir in {"bm25", "bm25/"}:
        return False
    if is_bm25_only_config(config):
        return False
    if retriever_type in _DENSE_RETRIEVER_TYPES:
        return True
    args = config.get("retriever_args") if isinstance(config.get("retriever_args"), dict) else {}
    return bool(config.get("retriever_dir")) or bool(args.get("search_index_filename"))
