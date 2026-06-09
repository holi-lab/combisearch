from __future__ import annotations

import os
from typing import Any, List

from combisearch.config.paths import COMBISEARCH_OUTPUTS_DIR
from combisearch.representation.types import Turn
from combisearch.runtime.env import WANDB_ENTITY, WANDB_PROJECT
from combisearch.runtime.wandb import wandb


def _require_wandb_attr(name: str) -> Any:
    attr = getattr(wandb, name, None) if wandb is not None else None
    if attr is None:
        raise RuntimeError(f"W&B {name} API is unavailable; install/configure wandb or use local artifacts")
    return attr


def output_dir_to_run_or_artifact_name(output_dir: str) -> str:
    parent_dir = (
        os.environ.get(COMBISEARCH_OUTPUTS_DIR)
        or "outputs/"
    )
    return output_dir.replace(parent_dir, "").replace('/', '-')


def read_json_artifact(artifact_name: str, file_path: str,
                       alias: str = 'latest', project: str | None = None,
                       entity: str | None = None) -> Any:
    import json
    entity = entity or os.environ.get(WANDB_ENTITY)
    project = project or os.environ.get(WANDB_PROJECT, "combisearch")
    if not entity:
        raise RuntimeError(
            f"set the {WANDB_ENTITY} environment variable (or pass entity=) to "
            "load a W&B artifact"
        )
    api_cls = _require_wandb_attr("Api")
    api = api_cls()
    artifact = api.artifact(f'{entity}/{project}/{artifact_name}:{alias}')
    download_path: str = artifact.download()
    with open(os.path.join(download_path, file_path), "r") as f:
        return json.load(f)


def get_json_artifact_by_file_name(expected_file_path: str) -> Any:
    output_dir, file_name = expected_file_path.rsplit("/", maxsplit=1)
    artifact_name: str = output_dir_to_run_or_artifact_name(output_dir)
    return read_json_artifact(artifact_name, file_name)


def read_run_artifact_logs(run_id: str, project: str | None = None,
                           entity: str | None = None) -> List[Turn] | None:
    """Return the running log a W&B run logged as an artifact, if any."""
    entity = entity or os.environ.get(WANDB_ENTITY)
    project = project or os.environ.get(WANDB_PROJECT, "combisearch")
    if not entity:
        raise RuntimeError(
            f"set the {WANDB_ENTITY} environment variable (or pass entity=) to read run artifacts"
        )
    api = _require_wandb_attr("Api")()
    run = api.run(f"{entity}/{project}/{run_id}")
    for artifact in run.logged_artifacts():
        if artifact.type in ("run_output", "running_log"):
            return read_json_artifact(
                artifact.name.split(":")[0], file_path="running_log.json", alias=artifact.version
            )
    return None


def get_running_logs_by_group(group_id: str, tags_in: List[str] | None = None,
                              tags_not_in: List[str] | None = None) -> List[List[Turn]]:
    """Return the running logs of every W&B run in a group, subject to tag filters."""
    tags_in = tags_in or ["complete_run"]
    tags_not_in = tags_not_in or ["outdated"]
    entity = os.environ.get(WANDB_ENTITY)
    project = os.environ.get(WANDB_PROJECT, "combisearch")
    if not entity:
        raise RuntimeError(
            f"set the {WANDB_ENTITY} environment variable to read run artifacts by group"
        )
    api = _require_wandb_attr("Api")()
    runs = api.runs(
        path=f"{entity}/{project}",
        filters={"group": group_id, "tags": {"$in": tags_in, "$nin": tags_not_in}},
    )
    return [read_run_artifact_logs(run.id) for run in runs]
