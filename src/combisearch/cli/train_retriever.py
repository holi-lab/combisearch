"""Run CombiSearch retriever-finetuning configs.

Works in both single-process (``python``) and distributed
(``accelerate launch`` / ``torchrun``) modes; rank-0 gating collapses to
no-ops in single-process mode.
"""

from __future__ import annotations

from pathlib import Path

from combisearch.runtime.artifacts import output_dir_to_run_or_artifact_name
from combisearch.cli.runner import (
    apply_runtime_seed,
    default_run_log_path,
    init_wandb,
    load_json_config,
    run_config_targets,
    tee_output_to_log,
)
from combisearch.runtime.checks import check_runtime_config
from combisearch.runtime.dist import get_local_rank, is_main_process


def _all_ranks_tee(log_path: str | Path):
    """Tee stdout/stderr for every rank.

    Rank 0  → ``log_path`` (e.g. run.log)
    Rank N  → ``<dir>/run_rankN.log``

    Writing to separate files avoids concurrent-write races while still
    capturing errors from non-main ranks.
    """
    log_path = Path(log_path)
    if is_main_process():
        return tee_output_to_log(log_path)
    rank = get_local_rank()
    rank_log = log_path.with_name(f"{log_path.stem}_rank{rank}{log_path.suffix}")
    return tee_output_to_log(rank_log)


def run_config(config_path: str | Path):
    config_path = Path(config_path)
    raw_config = load_json_config(config_path)
    runtime_config = check_runtime_config(
        raw_config,
        stage="retriever_training",
        config_path=config_path,
        require_retriever=False,
        require_llm=False,
    )
    runtime_config.setdefault("run_name", output_dir_to_run_or_artifact_name(runtime_config["output_dir"]))
    runtime_config.setdefault("log_file", str(default_run_log_path(runtime_config)))
    # Every rank re-applies the same seed so deterministic dataloader
    # materialization (loaders.py random.randint) lines up across processes.
    apply_runtime_seed(runtime_config)

    with _all_ranks_tee(runtime_config["log_file"]):
        if is_main_process():
            print(f"Writing console log to {runtime_config['log_file']}")
            init_wandb(runtime_config)
        from combisearch.training.retriever.finetune import run_finetuning

        return run_finetuning(runtime_config)


def main(argv: list[str] | None = None) -> int:
    return run_config_targets(
        argv,
        description=__doc__,
        stage_label="retriever training",
        run_config=run_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
