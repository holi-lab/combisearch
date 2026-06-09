"""Experiment config schema and path resolution.

- ``combisearch.config.schema``  : normalize/validate config dicts, discriminator checks
- ``combisearch.config.paths``   : canonical output paths, declared-artifact path resolution
- ``combisearch.config.finetune_config``: dataclass for retriever-training settings
"""

from combisearch.config.finetune_config import FinetuneConfig
from combisearch.config.paths import (
    apply_runtime_output_path,
    canonical_output_path,
    classify_config,
    embedding_model_tag,
    experiment_output_records,
    iter_experiment_configs,
    output_path_collisions,
    resolve_canonical_path,
    resolve_data_path,
    resolve_output_path,
    resolve_retriever_dir,
    resolve_search_index_file,
)
from combisearch.config.schema import (
    KNOWN_DECODER_TYPES,
    KNOWN_LOSS_TYPES,
    KNOWN_PROMPT_FORMATS,
    KNOWN_RETRIEVER_TYPES,
    STAGE_REQUIRED_KEYS,
    declared_retrieval_strategy,
    is_bm25_only_config,
    normalize_config,
    prepare_runtime_config,
    requires_declared_dense_retriever,
    validate_stage_config,
)

__all__ = [
    "FinetuneConfig",
    "KNOWN_DECODER_TYPES",
    "KNOWN_LOSS_TYPES",
    "KNOWN_PROMPT_FORMATS",
    "KNOWN_RETRIEVER_TYPES",
    "STAGE_REQUIRED_KEYS",
    "apply_runtime_output_path",
    "canonical_output_path",
    "classify_config",
    "declared_retrieval_strategy",
    "embedding_model_tag",
    "experiment_output_records",
    "is_bm25_only_config",
    "iter_experiment_configs",
    "normalize_config",
    "output_path_collisions",
    "prepare_runtime_config",
    "requires_declared_dense_retriever",
    "resolve_canonical_path",
    "resolve_data_path",
    "resolve_output_path",
    "resolve_retriever_dir",
    "resolve_search_index_file",
    "validate_stage_config",
]
