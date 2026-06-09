"""Runtime preflight: validate config + verify declared inputs exist."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from combisearch.config.paths import (
    DEFAULT_CONFIG_ROOT,
    canonical_output_path,
    classify_config,
    resolve_canonical_path,
    resolve_data_path,
    resolve_output_path,
    resolve_retriever_dir,
    resolve_search_index_file,
)
from combisearch.config.schema import (
    prepare_runtime_config,
    requires_declared_dense_retriever,
)


def is_openai_engine(engine: Any) -> bool:
    return str(engine or "").lower().startswith(("gpt-", "o1-", "o3-", "o4-"))


def is_local_model_reference(engine: Any) -> bool:
    text = str(engine or "").strip()
    if not text:
        return False
    return Path(text).is_absolute() or text.startswith((".", "..", "outputs/"))


def require_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def require_hf_model_dir(path: Path, label: str) -> None:
    """An HF/SentenceTransformer model dir must have weights and a config."""
    require_dir(path, label)
    has_weights = any(
        candidate.exists()
        for candidate in (
            path / "model.safetensors", path / "pytorch_model.bin",
            *path.glob("*/model.safetensors"), *path.glob("*/pytorch_model.bin"),
        )
    )
    if not has_weights:
        raise FileNotFoundError(f"{label} has no model weights: {path}")
    has_config = (
        (path / "config.json").exists()
        or (path / "modules.json").exists()
        or any(path.glob("*/config.json"))
    )
    if not has_config:
        raise FileNotFoundError(f"{label} has no model config: {path}")


def require_llm_ready(config: dict[str, Any]) -> None:
    engine = config.get("llm_engine")
    if is_openai_engine(engine):
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY_JLAB_ORG")):
            raise EnvironmentError("OpenAI engine requested but OPENAI_API_KEY is not set")
        return
    if is_local_model_reference(engine):
        require_hf_model_dir(resolve_output_path(engine) or Path(str(engine)), "local LLM")


def _index_outside_retriever(config: dict[str, Any]) -> bool:
    retriever_dir = resolve_retriever_dir(config)
    search_index = resolve_search_index_file(config)
    if retriever_dir is None or search_index is None:
        return False
    try:
        search_index.resolve().relative_to(retriever_dir.resolve())
        return False
    except ValueError:
        return True


def _require_canonical_output(
    raw_config: dict[str, Any],
    runtime_config: dict[str, Any],
    *,
    stage: str,
    config_path: str | Path | None,
) -> None:
    if config_path is None:
        return
    cfg_path = Path(config_path).resolve()
    cfg_root = DEFAULT_CONFIG_ROOT
    try:
        cfg_path.relative_to(cfg_root)
    except ValueError:
        return

    classified = classify_config(cfg_path, cfg_root, raw_config)
    if classified != stage:
        raise RuntimeError(
            f"config stage mismatch: {cfg_path.relative_to(cfg_root)} is {classified!r}, "
            f"but this entrypoint requires {stage!r}"
        )

    expected = resolve_canonical_path(canonical_output_path(cfg_path, stage, cfg_root)).resolve()
    actual = Path(str(runtime_config.get("output_dir", ""))).resolve()
    if actual != expected:
        raise RuntimeError(
            f"runtime output_dir is not canonical: actual={actual}, expected={expected}"
        )


def _require_dense_retriever(runtime_config: dict[str, Any]) -> None:
    retriever_dir = resolve_retriever_dir(runtime_config)
    search_index = resolve_search_index_file(runtime_config)
    if retriever_dir is None:
        raise ValueError("dense retriever config requires retriever_dir")
    if search_index is None:
        raise ValueError("dense retriever config requires retriever_args.search_index_filename or retriever_dir")

    if _index_outside_retriever(runtime_config):
        if not runtime_config.get("index_embedding_model_name"):
            raise ValueError(
                "dense retriever config uses a search index outside retriever_dir; "
                "set index_embedding_model_name to the encoder used to build that index"
            )
        args = runtime_config.get("retriever_args") or {}
        if not (args.get("dense_input_kwargs") or {}):
            raise ValueError(
                "dense retriever config uses a search index outside retriever_dir; "
                "set retriever_args.dense_input_kwargs to the runtime input representation"
            )

    index_model = runtime_config.get("index_embedding_model_name")
    if is_local_model_reference(index_model):
        require_hf_model_dir(
            resolve_output_path(index_model) or Path(str(index_model)),
            "index embedding model",
        )
    require_hf_model_dir(retriever_dir, "retriever")
    require_file(search_index, "search index")


def check_runtime_config(
    config: dict[str, Any],
    *,
    stage: str,
    config_path: str | Path | None = None,
    require_retriever: bool = True,
    require_llm: bool = True,
) -> dict[str, Any]:
    """Normalize the config, inject canonical output_dir, and verify declared inputs exist."""
    runtime_config = prepare_runtime_config(config, config_path=config_path, stage=stage)
    _require_canonical_output(config, runtime_config, stage=stage, config_path=config_path)

    for key, label in (("train_fn", "train data"), ("test_fn", "test data"), ("dev_fn", "dev data")):
        value = runtime_config.get(key)
        if value:
            require_file(resolve_data_path(value), label)

    if stage == "retriever_training":
        base = runtime_config.get("base_embedding_model_name")
        if is_local_model_reference(base):
            require_hf_model_dir(
                resolve_output_path(base) or Path(str(base)),
                "base embedding model",
            )

    if require_retriever and requires_declared_dense_retriever(runtime_config):
        _require_dense_retriever(runtime_config)

    if require_llm:
        require_llm_ready(runtime_config)

    return runtime_config
