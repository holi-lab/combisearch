"""Shared helpers for the CLI entry points."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

from combisearch.runtime.artifacts import output_dir_to_run_or_artifact_name
from combisearch.runtime.checks import check_runtime_config
from combisearch.runtime.env import WANDB_ENTITY, WANDB_PROJECT
from combisearch.runtime.io import read_json
from combisearch.runtime.seed import resolve_seed, set_global_seed
from combisearch.runtime.wandb import (
    wandb_add_run_tag,
    wandb_artifact_dir,
    wandb_init,
    wandb_run,
)

ROOT = Path(__file__).resolve().parents[3]
_EXCLUDED_RUN_ARTIFACTS = {"running_log.json", "exp_config.json"}


def configure_default_env() -> None:
    """Apply the same defaults as the shell wrappers."""
    os.environ.setdefault("COMBISEARCH_DATA_DIR", str(ROOT / "data"))
    os.environ.setdefault("COMBISEARCH_OUTPUTS_DIR", str(ROOT / "outputs"))
    os.environ.setdefault("COMBISEARCH_CONFIG_ROOT", str(ROOT / "configs"))
    os.environ.setdefault("WANDB_MODE", "offline")


def apply_runtime_seed(runtime_config: dict[str, Any] | None = None) -> int:
    """Pin every CombiSearch entrypoint's randomness to a single seed.

    Reads ``runtime_config['seed']``, then ``$COMBISEARCH_SEED``, falling
    back to the package default. Safe to call multiple times (each call
    re-applies the resolved seed).
    """
    seed = resolve_seed(runtime_config)
    return set_global_seed(seed)


class _TeeStream:
    """Write to both a console stream and a log file."""

    def __init__(self, console: TextIO, log_file: TextIO) -> None:
        self.console = console
        self.log_file = log_file
        self.encoding = getattr(console, "encoding", None)
        self.errors = getattr(console, "errors", None)

    def write(self, data: str) -> int:
        self.console.write(data)
        self.log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self.console.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.console.isatty()


@contextmanager
def tee_output_to_log(log_path: str | Path):
    """Mirror stdout/stderr into a log file while keeping the console output."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = _TeeStream(original_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = _TeeStream(original_stderr, log_file)  # type: ignore[assignment]
        try:
            yield log_path
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def default_run_log_path(runtime_config: dict[str, Any]) -> Path:
    return Path(runtime_config["output_dir"]) / "run.log"


def _collect_targets(patterns: Iterable[str]) -> list[Path]:
    """Expand file paths and globs into runnable config files."""
    targets: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_file():
            if path.name not in _EXCLUDED_RUN_ARTIFACTS:
                targets.append(path)
            continue
        runnable = sorted(
            Path(m) for m in glob.glob(pattern, recursive=True)
            if Path(m).is_file() and Path(m).name not in _EXCLUDED_RUN_ARTIFACTS
        )
        if not runnable:
            raise FileNotFoundError(f"no runnable config files matched: {pattern}")
        targets.extend(runnable)
    if not targets:
        raise ValueError("no runnable config files provided")
    return targets


def init_wandb(runtime_config: dict[str, Any]):
    name = runtime_config.get("run_name") or output_dir_to_run_or_artifact_name(runtime_config["output_dir"])
    group = runtime_config.get("run_group") or name.rsplit("-", maxsplit=1)[0]
    return wandb_init(
        config=runtime_config,
        project=os.environ.get(WANDB_PROJECT, "combisearch"),
        entity=os.environ.get(WANDB_ENTITY),
        name=name,
        notes=runtime_config.get("run_notes"),
        group=group,
        tags=runtime_config.get("run_tags"),
    )


def load_json_config(config_path: str | Path) -> dict[str, Any]:
    config = read_json(config_path)
    if not isinstance(config, dict):
        raise TypeError(f"config must be a JSON object: {config_path}")
    return config


def write_runtime_config(runtime_config: dict[str, Any]) -> None:
    output_dir = Path(runtime_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "exp_config.json").write_text(json.dumps(runtime_config, indent=2))


def run_config_targets(
    argv: list[str] | None,
    *,
    description: str | None,
    stage_label: str,
    run_config: Callable[[Path], Any],
) -> int:
    """Parse CLI args, expand globs, then call ``run_config`` for each target."""
    configure_default_env()
    apply_runtime_seed()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("configs", nargs="+", help="config file path or glob")
    parser.add_argument("--dry-run", action="store_true", help="print matched configs without running")
    args = parser.parse_args(argv)

    targets = _collect_targets(args.configs)
    is_dry_run = args.dry_run or os.environ.get("DRY_RUN", "0").lower() in {"1", "true", "yes"}
    if is_dry_run:
        print(f"[dry-run] {stage_label}: {len(targets)} config(s)")
        for t in targets:
            print(f"  - {t}")
        return 0

    for target in targets:
        run_config(target)
    return 0


def run_experiment_config(
    config_path: str | Path,
    *,
    stage: str,
    experiment_factory: Callable[[dict[str, Any], Path], Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Common runner for DSTExperiment-style configs."""
    config_path = Path(config_path)
    raw_config = load_json_config(config_path)
    runtime_config = check_runtime_config(raw_config, stage=stage, config_path=config_path)
    runtime_config.setdefault("run_name", output_dir_to_run_or_artifact_name(runtime_config["output_dir"]))
    runtime_config.setdefault("log_file", str(default_run_log_path(runtime_config)))
    apply_runtime_seed(runtime_config)
    write_runtime_config(runtime_config)

    with tee_output_to_log(runtime_config["log_file"]):
        print(f"Writing console log to {runtime_config['log_file']}")
        init_wandb(runtime_config)
        experiment = experiment_factory(runtime_config, config_path)
        running_log, stats = experiment.run()
        (Path(experiment.output_dir) / "running_log.json").write_text(json.dumps(running_log, indent=2))

        run = wandb_run()
        if run is not None:
            wandb_artifact_dir(
                getattr(run, "name", runtime_config["run_name"]),
                "run_output",
                experiment.output_dir,
            )
            if len(running_log) == len(experiment.test_set):
                wandb_add_run_tag("complete_run")
        return running_log, stats
