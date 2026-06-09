"""Run CombiSearch DST inference configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from combisearch.cli.runner import (
    run_config_targets,
    run_experiment_config,
)


def _make_experiment(runtime_config: dict[str, Any], config_path: Path):
    from combisearch.pipeline.dst import DSTExperiment

    return DSTExperiment(
        runtime_config=runtime_config,
        config_path=config_path,
    )


def run_config(config_path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return run_experiment_config(
        config_path,
        stage="dst_run",
        experiment_factory=_make_experiment,
    )


def main(argv: list[str] | None = None) -> int:
    return run_config_targets(
        argv,
        description=__doc__,
        stage_label="DST inference",
        run_config=run_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
