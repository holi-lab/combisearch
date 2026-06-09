"""Optional Weights & Biases helpers.
"""

from __future__ import annotations

import logging
from os import PathLike
from typing import Any, Dict, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

try:
    import wandb as wandb
except Exception:  # pragma: no cover - depends on the caller's environment
    wandb = None


def wandb_run() -> Any | None:
    return getattr(wandb, "run", None) if wandb is not None else None


def wandb_init(**kwargs: Any) -> Any | None:
    init_fn = getattr(wandb, "init", None) if wandb is not None else None
    if not callable(init_fn):
        logger.warning("wandb.init is unavailable; continuing without W&B logging")
        return None
    return init_fn(**kwargs)


def wandb_log(items: Mapping[str, Any]) -> bool:
    if wandb_run() is None:
        return False
    log_fn = getattr(wandb, "log", None)
    if not callable(log_fn):
        return False
    log_fn(dict(items))
    return True


def wandb_config_update(items: Mapping[str, Any]) -> bool:
    if wandb_run() is None:
        return False
    config = getattr(wandb, "config", None)
    update_fn = getattr(config, "update", None)
    if not callable(update_fn):
        return False
    update_fn(dict(items))
    return True


def wandb_artifact_dir(
    artifact_name: str,
    artifact_type: str,
    directory: str | PathLike[str],
    metadata: Mapping[str, Any] | None = None,
) -> bool:
    run = wandb_run()
    if run is None:
        return False
    artifact_cls = getattr(wandb, "Artifact", None)
    if not callable(artifact_cls):
        return False
    artifact = artifact_cls(artifact_name, type=artifact_type, metadata=dict(metadata or {}))
    artifact.add_dir(str(directory))
    log_artifact_fn = getattr(wandb, "log_artifact", None) or getattr(run, "log_artifact", None)
    if not callable(log_artifact_fn):
        return False
    log_artifact_fn(artifact)
    return True


def wandb_add_run_tag(tag: str) -> bool:
    run = wandb_run()
    if run is None:
        return False
    tags = set(getattr(run, "tags", ()) or ())
    tags.add(tag)
    run.tags = tuple(tags)
    return True


def wandb_bar_chart(
    data: Iterable[Sequence[Any]],
    columns: Sequence[str],
    x: str,
    y: str,
    title: str,
) -> Any | None:
    if wandb_run() is None:
        return None
    table_cls = getattr(wandb, "Table", None)
    plot = getattr(wandb, "plot", None)
    bar_fn = getattr(plot, "bar", None)
    if not callable(table_cls) or not callable(bar_fn):
        return None
    table = table_cls(data=list(data), columns=list(columns))
    return bar_fn(table, x, y, title=title)


class WandbStepLogger:
    """Buffer per-step metrics and flush them on ``step()`` calls."""

    def __init__(self) -> None:
        self.current_step: int = 0
        self.log_for_step: Dict[str, Any] = {}

    def log(self, items: Dict[str, Any]) -> None:
        self.log_for_step.update(items)

    def step(self, increment: int = 1) -> None:
        wandb_log({"current_step": self.current_step, **self.log_for_step})
        self.current_step += increment
        self.log_for_step = {}
