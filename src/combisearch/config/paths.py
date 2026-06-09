"""Path roots, canonical output paths, and config-declared artifact resolution.

A CombiSearch experiment is identified by its config file. The runtime output
directory mirrors the config's path under a stage-specific artifact root:

    configs/table_2/8B/combisearch/1p/split_v1.json
        -> outputs/runs/table_2/8B/combisearch/1p/split_v1/
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[3]

COMBISEARCH_DATA_DIR = "COMBISEARCH_DATA_DIR"
COMBISEARCH_OUTPUTS_DIR = "COMBISEARCH_OUTPUTS_DIR"
COMBISEARCH_CONFIG_ROOT = "COMBISEARCH_CONFIG_ROOT"

DEFAULT_DATA_ROOT: Path = Path(os.environ.get(COMBISEARCH_DATA_DIR) or ROOT / "data").resolve()
DEFAULT_OUTPUTS_ROOT: Path = Path(os.environ.get(COMBISEARCH_OUTPUTS_DIR) or ROOT / "outputs").resolve()
DEFAULT_CONFIG_ROOT: Path = Path(os.environ.get(COMBISEARCH_CONFIG_ROOT, ROOT / "configs")).resolve()

MW21_100P_TRAIN = "data/mw21_100p_train.json"
SGD_5P_TRAIN = "data/sgd_5p_train.json"


# JSON files inside configs/ that are not experiment configs (HF model
# sidecars, trainer state, etc.). Classification skips these.
EXCLUDED_JSON_NAMES = {
    "running_log.json", "analyzed_log.json", "trainer_state.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "config_sentence_transformers.json",
    "sentence_bert_config.json", "modules.json",
}

_STAGE_OUTPUT_ROOTS = {
    "data_generation": Path("data/generated"),
    "retriever_training": Path("outputs/retrievers"),
    "dst_run": Path("outputs/runs"),
}


def resolve_canonical_path(canonical_path: str | Path) -> Path:
    """Resolve a path starting with ``outputs/`` or ``data/`` to its root."""
    path = Path(canonical_path)
    if path.is_absolute():
        return path
    if not path.parts:
        return ROOT
    if path.parts[0] == "outputs":
        return DEFAULT_OUTPUTS_ROOT.joinpath(*path.parts[1:])
    if path.parts[0] == "data":
        return DEFAULT_DATA_ROOT.joinpath(*path.parts[1:])
    return ROOT / path


def resolve_data_path(file_name: str | Path) -> Path:
    """Resolve a data file path through the configured data/outputs roots."""
    path = Path(file_name)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in {"data", "outputs"}:
        return resolve_canonical_path(path)
    if path.parts and path.parts[0] in {"indexes", "models", "retrievers", "runs"}:
        return DEFAULT_OUTPUTS_ROOT / path
    return path if path.exists() else DEFAULT_DATA_ROOT / path


def is_empty_value(value: Any) -> bool:
    """True for None or strings whose stripped lowercase form is empty/'none'/'null'."""
    return value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"})


def resolve_output_path(value: Any) -> Path | None:
    """Resolve a config-declared output artifact path."""
    if is_empty_value(value):
        return None
    path = Path(str(value).strip())
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "outputs":
        return resolve_canonical_path(path)
    return DEFAULT_OUTPUTS_ROOT / path


def resolve_retriever_dir(config: dict[str, Any]) -> Path | None:
    raw = config.get("retriever_dir")
    if is_empty_value(raw) or str(raw).strip().lower() in {"bm25", "bm25/"}:
        return None
    return resolve_output_path(raw)


def resolve_search_index_file(config: dict[str, Any]) -> Path | None:
    """Resolve ``retriever_args.search_index_filename`` to a ``.npy`` path."""
    args = config.get("retriever_args") or {}
    raw = args.get("search_index_filename") if isinstance(args, dict) else None
    if is_empty_value(raw):
        retriever_dir = resolve_retriever_dir(config)
        return retriever_dir / "train_index.npy" if retriever_dir else None
    path = resolve_output_path(raw)
    if path is None:
        return None
    return path if path.suffix == ".npy" else path / "train_index.npy"


def slug(value: Any, default: str = "none") -> str:
    """File-system-safe slug of an arbitrary value."""
    if value is None:
        return default
    text = str(value).strip().replace("%", "p")
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text.lower() or default


def embedding_model_tag(value: Any) -> str:
    """Compact tag derived from an HF model name or local encoder path."""
    text = str(value or "sentence-transformers/all-mpnet-base-v2").strip()
    parts = [p for p in Path(text).parts if p not in ("", "/")]
    if len(parts) >= 3 and parts[-1] == "pretrained" and parts[-3] == "retrievers":
        return slug(parts[-2])
    if len(parts) >= 2 and parts[0] == "sentence-transformers":
        return slug("_".join(parts[:2]))
    return slug(text.replace("/", "_"))


def is_model_sidecar(path: Path, obj: dict[str, Any]) -> bool:
    """True when a JSON inside configs/ is an HF model sidecar, not an experiment."""
    if path.name in EXCLUDED_JSON_NAMES:
        return True
    model_keys = {"architectures", "model_type", "hidden_size", "num_attention_heads"}
    experiment_keys = {"train_fn", "test_fn", "dev_fn", "llm_engine", "prompt_format"}
    return bool(model_keys & set(obj)) and not bool(experiment_keys & set(obj))


def classify_config(path: Path, config_root: Path, obj: dict[str, Any]) -> str | None:
    """Return the stage of an experiment config, or None if it isn't one."""
    if is_model_sidecar(path, obj):
        return None
    if not any(key in obj for key in ("train_fn", "test_fn", "dev_fn")):
        return None
    parts = path.relative_to(config_root).parts
    if parts[0] == "finetuning" or "retriever_training" in parts:
        if "num_epochs" in obj or "loss_type" in obj or "top_k" in obj:
            return "retriever_training"
        return None
    # num_sampling_iteration marks a combinatorial-scoring (data-generation) run wherever it lives
    if parts[0] == "data_generation" or "data_generation" in parts or "num_sampling_iteration" in obj:
        return "data_generation"
    if "llm_engine" in obj or "prompt_format" in obj:
        return "dst_run"
    return None


def iter_experiment_configs(config_root: str | Path) -> Iterator[tuple[Path, dict[str, Any], str]]:
    """Yield ``(path, config, stage)`` for every experiment config under the root."""
    root = Path(config_root).resolve()
    for path in sorted(root.rglob("*.json")):
        if path.name in EXCLUDED_JSON_NAMES or "checkpoint" in path.as_posix():
            continue
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        stage = classify_config(path, root, obj)
        if stage is not None:
            yield path, obj, stage


def canonical_output_path(
    config_path: str | Path,
    stage: str,
    config_root: str | Path | None = None,
) -> str:
    """The output directory that mirrors this config under the stage's artifact root."""
    try:
        stage_root = _STAGE_OUTPUT_ROOTS[stage]
    except KeyError as exc:
        raise ValueError(f"unknown config stage: {stage}") from exc
    cfg_path = Path(config_path).resolve()
    root = Path(config_root).resolve() if config_root else DEFAULT_CONFIG_ROOT
    rel = cfg_path.relative_to(root)
    return (stage_root / rel.parent / cfg_path.stem).as_posix()


def apply_runtime_output_path(
    args: dict[str, Any],
    config_path: str | Path,
    stage: str,
    *,
    config_root: str | Path | None = None,
) -> dict[str, Any]:
    """Inject the canonical ``output_dir`` derived from the config file path."""
    resolved = dict(args)
    resolved["output_dir"] = str(
        resolve_canonical_path(canonical_output_path(config_path, stage, config_root))
    )
    return resolved


def experiment_output_records(config_root: str | Path) -> list[dict[str, str]]:
    """For every experiment config, the (source, stage, canonical output) triple."""
    root = Path(config_root).resolve()
    records: list[dict[str, str]] = []
    for cfg_path, _cfg, stage in iter_experiment_configs(root):
        try:
            source = cfg_path.relative_to(ROOT).as_posix()
        except ValueError:
            source = cfg_path.as_posix()
        records.append({
            "source_config": source,
            "stage": stage,
            "canonical_path": canonical_output_path(cfg_path, stage, root),
        })
    return records


def output_path_collisions(records: list[dict[str, str]]) -> list[dict[str, object]]:
    """Canonical output paths claimed by more than one config."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[record["canonical_path"]].append(record["source_config"])
    return [
        {"canonical_path": canonical, "source_configs": sources}
        for canonical, sources in sorted(grouped.items())
        if len(sources) > 1
    ]
