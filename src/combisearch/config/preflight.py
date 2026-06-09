"""Preflight scan of config-declared input artifacts.

Walks the ``configs/`` tree and resolves every declared input path (train/
test/dev data, retriever dir, search index, declared embedding models, and
any local LLM checkpoint) to verify it exists on disk. Read-only — never
modifies configs or path values.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from combisearch.config.paths import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    apply_runtime_output_path,
    embedding_model_tag,
    is_empty_value,
    iter_experiment_configs,
    resolve_canonical_path,
)
from combisearch.config.schema import requires_declared_dense_retriever


def _is_hf_identifier(name: str) -> bool:
    """HF model ids look like ``org/model`` with no path-like prefix."""
    return (
        bool(name)
        and not name.startswith(("/", ".", "~", "outputs/", "data/", "models/"))
        and "/" in name
    )


def _is_openai_engine(name: str) -> bool:
    return bool(name) and name.lower().startswith(("gpt-", "text-", "code-", "o1-", "o3-"))


def _resolve(value: Any, *, root: Path) -> Path | None:
    """Resolve a config-declared path under either ``data/`` or ``outputs/`` root."""
    text = str(value)
    if os.path.isabs(text):
        return Path(text)
    if text.startswith("outputs/"):
        return resolve_canonical_path(text)
    return root / text


def _check_data(value: Any) -> Path | None:
    if is_empty_value(value):
        return None
    return _resolve(str(value).removeprefix("data/"), root=DEFAULT_DATA_ROOT)


def _check_retriever(value: Any) -> Path | None:
    if is_empty_value(value) or str(value).lower() in {"bm25", "bm25/"}:
        return None
    return _resolve(value, root=DEFAULT_OUTPUTS_ROOT)


def _check_index(value: Any) -> Path | None:
    base = _check_retriever(value)
    if base is None:
        return None
    return base if base.suffix == ".npy" else base / "train_index.npy"


def _check_model(value: Any) -> Path | None:
    if is_empty_value(value):
        return None
    text = str(value)
    if _is_openai_engine(text) or _is_hf_identifier(text):
        return None
    return _resolve(text, root=DEFAULT_OUTPUTS_ROOT)


_PRETRAINED_INDEX_MARKER = "outputs/indexes/"


def _pretrained_index_error(config: dict[str, Any]) -> str | None:
    """Check that the declared index encoder tag matches the declared model name."""
    args = config.get("retriever_args") if isinstance(config.get("retriever_args"), dict) else {}
    search_index = str(args.get("search_index_filename") or "")
    if _PRETRAINED_INDEX_MARKER not in search_index:
        return None
    index_model = config.get("index_embedding_model_name")
    if not index_model:
        return "pretrained index declared, but index_embedding_model_name is missing"

    rest = search_index.split(_PRETRAINED_INDEX_MARKER, 1)[1].strip("/")
    index_tag = rest.split("/", 1)[0] if rest else None
    declared_tag = embedding_model_tag(index_model)
    if index_tag and index_tag != declared_tag:
        return (
            f"pretrained index model tag `{index_tag}` does not match "
            f"index_embedding_model_name `{declared_tag}`"
        )
    return None


# Most-specific first; the empty marker `""` is the per-category fallback.
_PRODUCERS: list[tuple[str, str, str]] = [
    ("data", "/generated/", "data generation: CombiSearch sampling stage"),
    ("data", "", "create or copy the exact config-declared data file"),
    ("index", "/outputs/retrievers/", "retriever training: python -m combisearch.cli.train_retriever produces train_index.npy"),
    ("index", "/outputs/indexes/", "python -m combisearch.cli.build_indexes"),
    ("index", "", "create the exact config-declared index path or update the config"),
    ("retriever", "/pretrained", "python -m combisearch.cli.build_indexes (saves declared dense encoder)"),
    ("retriever", "/outputs/retrievers/finetuning/", "retriever training: python -m combisearch.cli.train_retriever produces this directory"),
    ("retriever", "", "create the exact config-declared retriever dir or update the config"),
    ("index_embedding_model", "", "create the declared local encoder used to build the search index"),
    ("base_embedding_model", "", "create the declared local encoder used to initialize retriever training"),
    ("model", "", "set llm_engine to an existing local path or a supported remote model id"),
]

CATEGORY_ORDER: tuple[str, ...] = (
    "data", "index", "retriever", "index_embedding_model", "base_embedding_model", "model",
)


def producer_hint(category: str, target: Path) -> str:
    """Best-guess command/stage that should have produced ``target``."""
    s = target.as_posix()
    for cat, marker, hint in _PRODUCERS:
        if cat == category and (not marker or marker in s):
            return hint
    return "unknown"


def _gather_candidates(stage: str, resolved: dict[str, Any]) -> list[tuple[str, Path | None]]:
    cands: list[tuple[str, Path | None]] = [
        ("data", _check_data(resolved.get(k))) for k in ("train_fn", "test_fn", "dev_fn")
    ]
    if requires_declared_dense_retriever(resolved):
        ra = resolved.get("retriever_args") or {}
        cands.append(("retriever", _check_retriever(resolved.get("retriever_dir"))))
        cands.append(("index", _check_index(ra.get("search_index_filename") if isinstance(ra, dict) else None)))
        cands.append(("index_embedding_model", _check_model(resolved.get("index_embedding_model_name"))))
    if stage == "retriever_training":
        cands.append(("base_embedding_model", _check_model(resolved.get("base_embedding_model_name"))))
    cands.append(("model", _check_model(resolved.get("llm_engine"))))
    return cands


@dataclass
class ArtifactScan:
    """Result of scanning a config tree for missing declared artifacts.

    ``missing`` maps ``(category, resolved_path)`` to the list of configs
    that reference it; ``semantic_errors`` holds ``(config, message)`` pairs.
    """

    n_configs: int = 0
    n_missing: int = 0
    missing: dict[tuple[str, str], list[str]] = field(default_factory=lambda: defaultdict(list))
    semantic_errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.semantic_errors


def scan_config_artifacts(config_root: Path) -> ArtifactScan:
    """Scan every experiment config under ``config_root`` for missing inputs."""
    scan = ArtifactScan()
    for cfg_path, raw, stage in iter_experiment_configs(config_root):
        scan.n_configs += 1
        resolved = apply_runtime_output_path(raw, str(cfg_path), stage=stage, config_root=config_root)
        rel_cfg = cfg_path.relative_to(config_root).as_posix()

        if requires_declared_dense_retriever(resolved):
            err = _pretrained_index_error(resolved)
            if err:
                scan.semantic_errors.append((rel_cfg, err))

        for category, target in _gather_candidates(stage, resolved):
            if target is None or target.exists():
                continue
            scan.missing[(category, target.as_posix())].append(rel_cfg)
            scan.n_missing += 1
    return scan
