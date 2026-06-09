"""RefPyDST retriever fine-tuning.

The original RefPyDST recipe:

* Read raw training turns and compute a slot-value F1 similarity matrix.
* Mine hard positives/negatives from a pretrained SBERT index over the same
  training subset.
* Train a SentenceTransformer with :class:`OnlineContrastiveLoss` (margin-based)
  rather than the CombiSearch InfoNCE objective.
* Save the resulting encoder + ``train_index.npy`` at the canonical output
  directory (so inference configs can load it via the same retriever_dir).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import torch
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.losses import OnlineContrastiveLoss
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from combisearch.config.finetune_config import FinetuneConfig
from combisearch.config.paths import resolve_output_path
from combisearch.representation.turn_text import (
    get_state_transformation_by_type,
    get_string_transformation_by_type,
)
from combisearch.runtime.dist import device_for_local_rank, is_main_process
from combisearch.training.retriever.evaluation import RetrievalEvaluator

from combisearch.training.retriever.data import (
    MWDataset,
    save_embeddings as refpydst_save_embeddings,
)
from combisearch.retrieval.retrievers.index_based_retriever import IndexRetriever
from combisearch.retrieval.pretrained_index import (
    embed_single_sentence,
    read_MW_with_string_transformation,
    store_embed,
)
from combisearch.training.retriever.loaders import MWContrastiveDataloader
from combisearch.runtime.io import read_json_from_data_dir as refpydst_read_json

logger = logging.getLogger(__name__)

# The pretrained index for hard-negative mining adds [BS], but the finetuned
# retriever does not -- [BS] never appears in the retriever input.
INDEX_SPECIAL_TOKENS = ("[USER]", "[SYS]", "[CONTEXT]", "[BS]")
FINETUNE_SPECIAL_TOKENS = ("[USER]", "[SYS]", "[CONTEXT]")
PRETRAINED_INDEX_FILENAME = "pretrained_train_index.npy"
TRAIN_BATCH_SIZE_DEFAULT = 24
WARMUP_STEPS_DEFAULT = 100


def _build_pretrained_train_index(
    base_model_name: str,
    train_fn: str,
    output_dir: Path,
    filter_domains: bool = True,
) -> Path:
    """Materialize a SBERT embedding file for hard-negative mining over ``train_fn``.

    Mirrors :func:`refpydst.retriever.code.pretrained_embed_index.embed_everything`
    but parameterized on ``train_fn`` (the original hard-codes ``mw21_100p_train``).
    """
    index_path = output_dir / PRETRAINED_INDEX_FILENAME
    if index_path.exists():
        logger.info("Reusing pretrained train index at %s", index_path)
        return index_path

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModel.from_pretrained(base_model_name)
    tokenizer.add_tokens(list(INDEX_SPECIAL_TOKENS), special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    turns = read_MW_with_string_transformation(train_fn, filter_domains=filter_domains)

    def _embed(sentence: str):
        return embed_single_sentence(sentence, model=model, tokenizer=tokenizer, cls=False)

    store_embed(turns, str(index_path), _embed)
    logger.info("Built pretrained train index at %s (n=%d turns)", index_path, len(turns))
    return index_path


def run_refpydst_finetuning(runtime_config: Dict[str, Any]) -> None:
    """End-to-end RefPyDST retriever training from a runtime config dict.

    The runtime config is the same flat dict that
    :func:`combisearch.training.retriever.finetune.run_finetuning` consumes; this
    function is dispatched when ``loss_type == "cl"``.
    """
    cfg = FinetuneConfig.from_runtime_config(runtime_config)

    output_dir = resolve_output_path(runtime_config.get("output_dir"))
    if output_dir is None:
        raise ValueError(
            "canonical path generation did not resolve a retriever output directory"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = device_for_local_rank()
    f_beta = float(runtime_config.get("f_beta", 1.0))
    batch_size = int(runtime_config.get("batch_size", TRAIN_BATCH_SIZE_DEFAULT))
    warmup_steps = int(runtime_config.get("warmup_steps", WARMUP_STEPS_DEFAULT))

    string_transformation = get_string_transformation_by_type("default")
    state_transformation = get_state_transformation_by_type(cfg.state_transformation_type)

    # SGD passes filter_domains=False (its services aren't MultiWOZ domains) and
    # use_raw_turn_state=True (the reference-aware transforms are MultiWOZ-only).
    f1_train_set = MWDataset(
        cfg.train_file_name,
        beta=f_beta,
        string_transformation=string_transformation,
        state_transformation=state_transformation,
        filter_domains=bool(runtime_config.get("filter_domains", True)),
        use_raw_turn_state=bool(runtime_config.get("use_raw_turn_state", False)),
    )

    pretrained_index = _build_pretrained_train_index(
        base_model_name=cfg.base_embedding_model_name,
        train_fn=cfg.train_file_name,
        output_dir=output_dir,
        filter_domains=bool(runtime_config.get("filter_domains", True)),
    )

    train_set_raw = refpydst_read_json(cfg.train_file_name)
    pretrained_retriever = IndexRetriever(
        datasets=[train_set_raw],
        embedding_filenames=[str(pretrained_index)],
        search_index_filename=str(pretrained_index),
        # MultiWOZ uses "pre_assigned"; SGD needs "sgd_pre_assigned" (ids contain '_').
        sampling_method=runtime_config.get("sampling_method", "pre_assigned"),
    )

    loader = MWContrastiveDataloader(f1_train_set, pretrained_retriever)
    examples = loader.generate_train_examples(topk=cfg.top_k, top_range=cfg.top_range)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    logger.info("RefPyDST training pairs: %d (batch_size=%d)", len(examples), batch_size)

    word_embedding = models.Transformer(cfg.base_embedding_model_name, max_seq_length=512)
    pooling = models.Pooling(
        word_embedding_dimension=word_embedding.get_word_embedding_dimension(),
        pooling_mode=cfg.pooling_mode,
    )
    word_embedding.tokenizer.add_tokens(list(FINETUNE_SPECIAL_TOKENS), special_tokens=True)
    word_embedding.auto_model.resize_token_embeddings(len(word_embedding.tokenizer))

    model = SentenceTransformer(modules=[word_embedding, pooling], device=str(device))

    train_loss = OnlineContrastiveLoss(model=model)

    evaluator = RetrievalEvaluator(
        train_fn=cfg.train_file_name,
        dev_fn=cfg.dev_set_path,
        index_set=f1_train_set,
        string_transformation=string_transformation,
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=cfg.num_epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=(len(train_dataloader) // 300 + 1) * 100,
        output_path=str(output_dir),
    )

    if is_main_process():
        model = SentenceTransformer(str(output_dir), device=str(device))
        refpydst_save_embeddings(
            model, f1_train_set, os.path.join(str(output_dir), "train_index.npy")
        )
        logger.info("RefPyDST retriever saved at %s", output_dir)
