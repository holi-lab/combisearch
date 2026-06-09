"""Fine-tune a SentenceTransformer retriever on CombiSearch-scored data.

Single entrypoint for both single-process (``python``) and distributed
(``accelerate launch`` / ``torchrun``) runs. Rank-0 gating is performed via
:mod:`combisearch.runtime.dist`; in single-process mode every gate is a no-op
(``is_main_process()`` is True, ``barrier()`` is skipped).
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict

import transformers
from packaging import version

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from sentence_transformers import models
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.fit_mixin import EvaluatorCallback, SaveModelCallback
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import (
    BatchSamplers,
    MultiDatasetBatchSamplers,
    SentenceTransformerTrainingArguments,
)

from datasets import Dataset, DatasetDict

from combisearch.config.finetune_config import FinetuneConfig
from combisearch.config.paths import resolve_output_path
from combisearch.representation.turn_text import (
    get_state_transformation_by_type,
    get_string_transformation_by_type,
)
from combisearch.indexing.embed import save_embeddings
from combisearch.runtime.dist import (
    barrier,
    device_for_local_rank,
    is_main_process,
)
from combisearch.runtime.wandb import wandb_artifact_dir, wandb_config_update, wandb_run
from combisearch.training.retriever.data import ScoredExampleDataset
from combisearch.training.retriever.evaluation import DstJgaEvaluator, RetrievalEvaluator
from combisearch.training.retriever.loaders import InfoNCEDataloader
from combisearch.training.retriever.loss import InfoNCELoss


logger = logging.getLogger(__name__)

DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WARMUP_STEPS = 100
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_EVAL_BATCH_SIZE = 32
SPECIAL_TOKENS = ["[USER]", "[SYS]", "[CONTEXT]"]
NO_DECAY_PARAMS = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
CHECKPOINT_DIR = "checkpoints"

DEFAULT_EVALUATOR_TYPE = "dst_jga"
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.4
DEFAULT_VLLM_ENFORCE_EAGER = True


def _build_evaluator(
    evaluator_type: str,
    *,
    finetune_config: FinetuneConfig,
    train_dataset: ScoredExampleDataset,
    eval_batch_size: int,
    string_transformation,
    vllm_gpu_memory_utilization: float,
    vllm_enforce_eager: bool,
):
    """Construct the requested evaluator on rank 0 only.

    HuggingFace ``Trainer`` already runs evaluation on the main process, so
    building the evaluator solely on rank 0 keeps every non-main rank from
    paying the (large) vLLM init cost or competing for GPU memory.
    """
    if not is_main_process():
        return None

    common_kwargs = dict(
        train_fn=finetune_config.train_file_name,
        dev_fn=finetune_config.dev_set_path,
        index_set=train_dataset,
        batch_size=eval_batch_size,
        string_transformation=string_transformation,
    )

    evaluator_type = (evaluator_type or DEFAULT_EVALUATOR_TYPE).lower()
    if evaluator_type == "retrieval":
        return RetrievalEvaluator(**common_kwargs)
    if evaluator_type != "dst_jga":
        raise ValueError(
            f"unknown evaluator_type {evaluator_type!r}; expected 'dst_jga' or 'retrieval'"
        )

    _DDP_ENV_KEYS = [
        "MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK",
        "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_USE_AGENT_STORE",
    ]
    _saved = {k: os.environ.pop(k, None) for k in _DDP_ENV_KEYS}
    try:
        evaluator = DstJgaEvaluator(
            **common_kwargs,
            vllm_enforce_eager=vllm_enforce_eager,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        )
    finally:
        for k, v in _saved.items():
            if v is not None:
                os.environ[k] = v
    return evaluator


def build_model_and_data(
    finetune_config: FinetuneConfig,
    batch_size: int,
    eval_batch_size: int,
    max_seq_length: int,
    device: str,
    str_transformation_type: str = "default",
    evaluator_type: str = DEFAULT_EVALUATOR_TYPE,
    vllm_gpu_memory_utilization: float = DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
    vllm_enforce_eager: bool = DEFAULT_VLLM_ENFORCE_EAGER,
) -> tuple:
    """Build the SentenceTransformer model, training DataLoader, and evaluator.

    The evaluator is created on rank 0 only; non-main ranks receive ``None``
    (HF Trainer runs evaluation on the main process anyway).
    """
    pretrained = finetune_config.base_embedding_model_name

    word_embedding_model = models.Transformer(pretrained, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(
        word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
        pooling_mode=finetune_config.pooling_mode,
    )
    word_embedding_model.tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
    model.max_seq_length = max_seq_length

    logger.info(
        f"Model {pretrained} loaded on {device}. "
        f"embedding_dim={model.get_sentence_embedding_dimension()}, "
        f"trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    string_transformation = get_string_transformation_by_type(str_transformation_type)
    state_transformation = get_state_transformation_by_type(finetune_config.state_transformation_type)

    train_dataset = ScoredExampleDataset(
        finetune_config.train_file_name,
        string_transformation=string_transformation,
        state_transformation=state_transformation,
        filter_domains=finetune_config.filter_domains,
        use_raw_turn_state=finetune_config.use_raw_turn_state,
    )

    neg_size_kwargs = {"neg_size": finetune_config.neg_size} if finetune_config.neg_size else {}
    train_loader = InfoNCEDataloader(
        train_dataset, score_type=finetune_config.score_type, **neg_size_kwargs
    )
    train_samples = train_loader.generate_train_examples(
        topk=finetune_config.top_k, top_range=finetune_config.top_range
    )
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
    logger.info(f"Training batches: {len(train_dataloader)}")

    evaluator = _build_evaluator(
        evaluator_type,
        finetune_config=finetune_config,
        train_dataset=train_dataset,
        eval_batch_size=eval_batch_size,
        string_transformation=string_transformation,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_enforce_eager=vllm_enforce_eager,
    )

    barrier()

    return model, train_dataloader, evaluator, train_dataset


def _build_train_dataset_dict(train_dataloader: DataLoader) -> tuple[DatasetDict, dict]:
    """Materialize the dataloader into a single-entry DatasetDict expected by the Trainer."""
    train_dataloader.collate_fn = lambda batch: batch

    texts: list = []
    labels: list = []
    for batch in train_dataloader:
        batch_texts, batch_labels = zip(*[(ex.texts, ex.label) for ex in batch])
        texts += batch_texts
        labels += batch_labels

    dataset = Dataset.from_dict(
        {f"sentence_{idx}": col for idx, col in enumerate(zip(*texts))}
    )
    # If every example has label 0 (the InputExample default), we don't carry a
    # label column. Wrap in try/except so unhashable labels don't crash.
    try:
        has_real_labels = set(labels) != {0}
    except TypeError:
        has_real_labels = True
    if has_real_labels:
        dataset = dataset.add_column("label", labels)

    # Key "train" → HF Trainer produces metric "eval_train_loss", which must
    # match metric_for_best_model in prepare_training_components.
    return DatasetDict({"train": dataset})


def prepare_training_components(
    model: SentenceTransformer,
    train_dataloader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    use_amp: bool,
    embedder_save_dir: str,
) -> tuple:
    """Build loss, optimizer, scheduler, and TrainingArguments."""
    train_loss = InfoNCELoss(model=model)
    train_dataset_dict = _build_train_dataset_dict(train_dataloader)
    loss_fn_dict = {"train": train_loss}

    steps_per_epoch = len(train_dataloader)
    num_train_steps = steps_per_epoch * num_epochs
    batch_size = train_dataloader.batch_size or 1

    eval_strategy_key = (
        "eval_strategy"
        if version.parse(transformers.__version__) >= version.parse("4.41.0")
        else "evaluation_strategy"
    )
    training_args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(embedder_save_dir, CHECKPOINT_DIR),
        batch_sampler=BatchSamplers.BATCH_SAMPLER,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.ROUND_ROBIN,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        max_steps=-1,
        **{eval_strategy_key: "steps"},
        eval_steps=steps_per_epoch,
        max_grad_norm=DEFAULT_MAX_GRAD_NORM,
        fp16=use_amp,
        disable_tqdm=not is_main_process(),
        save_strategy="steps",
        save_steps=steps_per_epoch,
        save_total_limit=num_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_train_loss",
        greater_is_better=False,
        ddp_find_unused_parameters=True,
        report_to=["wandb"] if (is_main_process() and wandb_run() is not None) else [],
    )

    param_optimizer = list(model.named_parameters())
    grouped_params = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in NO_DECAY_PARAMS)],
            "weight_decay": DEFAULT_WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in NO_DECAY_PARAMS)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(grouped_params, lr=learning_rate)

    def lr_lambda(current_step: int) -> float:
        if current_step < DEFAULT_WARMUP_STEPS:
            return float(current_step) / float(max(1, DEFAULT_WARMUP_STEPS))
        return max(
            0.0,
            float(num_train_steps - current_step)
            / float(max(1, num_train_steps - DEFAULT_WARMUP_STEPS)),
        )

    scheduler = LambdaLR(optimizer, lr_lambda)

    logger.info(f"Training for {num_epochs} epochs ({num_train_steps} steps)")
    return train_dataset_dict, loss_fn_dict, training_args, optimizer, scheduler


def train_and_save_model(
    model: SentenceTransformer,
    training_args: SentenceTransformerTrainingArguments,
    train_dataset_dict: DatasetDict,
    loss_fn_dict: Dict,
    evaluator,
    optimizer: AdamW,
    scheduler: LambdaLR,
    embedder_save_dir: str,
    train_dataset: ScoredExampleDataset,
    device: str,
    max_seq_length: int,
) -> SentenceTransformer:
    """Run the Trainer, reload the best checkpoint, save model + train index.

    Evaluator-related callbacks and post-training filesystem writes are
    gated to rank 0 (no-op in single-process mode).
    """
    callbacks = []
    if is_main_process() and evaluator is not None:
        callbacks.append(EvaluatorCallback(evaluator, embedder_save_dir))

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_dict,
        eval_dataset=train_dataset_dict,
        loss=loss_fn_dict,
        evaluator=evaluator,
        optimizers=(optimizer, scheduler),
        callbacks=callbacks,
    )
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, EvaluatorCallback):
            callback.trainer = trainer
    if is_main_process() and embedder_save_dir is not None:
        trainer.add_callback(SaveModelCallback(embedder_save_dir, evaluator, True))

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=False)
    logger.info("Training completed.")

    barrier()

    if not is_main_process():
        return model

    model = SentenceTransformer(str(embedder_save_dir), device=device)
    model.max_seq_length = max_seq_length
    model.save(embedder_save_dir)
    logger.info(f"Model saved to {embedder_save_dir}")

    train_index_path = Path(embedder_save_dir) / "train_index.npy"
    save_embeddings(model, train_dataset, train_index_path)
    logger.info(f"Train index saved to {train_index_path}")

    run = wandb_run()
    if run is not None:
        wandb_artifact_dir(getattr(run, "name", embedder_save_dir.name), "model", embedder_save_dir)

    return model


def run_finetuning(config: Dict):
    """End-to-end retriever fine-tuning from a flat runtime config dict.

    Works in both single-process and DDP launches: rank-0 gates collapse to
    no-ops in single-process mode, and the device is picked from
    ``LOCAL_RANK`` (defaulting to ``cuda:0`` when unset).

    When ``loss_type == "cl"`` the call is dispatched to the RefPyDST recipe
    (OnlineContrastiveLoss over raw turns + F1 similarity-based hard-neg
    mining); otherwise the CombiSearch InfoNCE path below is used.
    """
    runtime_config = dict(config)
    if str(runtime_config.get("loss_type", "")).lower() == "cl":
        from combisearch.training.retriever.refpydst_pipeline import (
            run_refpydst_finetuning,
        )
        return run_refpydst_finetuning(runtime_config)

    finetuning_config = FinetuneConfig.from_runtime_config(runtime_config)
    logger.info("Resolved training data path: %s", finetuning_config.train_file_name)

    embedder_save_dir = resolve_output_path(runtime_config.get("output_dir"))
    if embedder_save_dir is None:
        raise ValueError("canonical path generation did not resolve a retriever output directory")

    device = device_for_local_rank()
    batch_size = finetuning_config.batch_size
    eval_batch_size = min(DEFAULT_EVAL_BATCH_SIZE, batch_size)
    # Effective batch size = 16 when batch_size < 16, else no accumulation.
    gradient_accumulation_steps = max(1, 16 // batch_size if batch_size < 16 else 1)

    evaluator_type = runtime_config.get("evaluator_type", DEFAULT_EVALUATOR_TYPE)
    vllm_mem = float(
        runtime_config.get(
            "vllm_gpu_memory_utilization", DEFAULT_VLLM_GPU_MEMORY_UTILIZATION
        )
    )
    vllm_enforce_eager = bool(
        runtime_config.get("vllm_enforce_eager", DEFAULT_VLLM_ENFORCE_EAGER)
    )

    if is_main_process():
        wandb_config_update(finetuning_config.__dict__)
        os.makedirs(embedder_save_dir, exist_ok=True)
        with open(os.path.join(embedder_save_dir, "exp_config.json"), "w") as f:
            json.dump(runtime_config, f, indent=4)
    barrier()

    model, train_dataloader, evaluator, train_dataset = build_model_and_data(
        finetune_config=finetuning_config,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
        device=device,
        str_transformation_type="default",
        evaluator_type=evaluator_type,
        vllm_gpu_memory_utilization=vllm_mem,
        vllm_enforce_eager=vllm_enforce_eager,
    )

    (
        train_dataset_dict, loss_fn_dict,
        training_args, optimizer, scheduler,
    ) = prepare_training_components(
        model=model,
        train_dataloader=train_dataloader,
        num_epochs=finetuning_config.num_epochs,
        learning_rate=DEFAULT_LEARNING_RATE,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_amp=True,
        embedder_save_dir=embedder_save_dir,
    )

    model = train_and_save_model(
        model=model,
        training_args=training_args,
        train_dataset_dict=train_dataset_dict,
        loss_fn_dict=loss_fn_dict,
        evaluator=evaluator,
        optimizer=optimizer,
        scheduler=scheduler,
        embedder_save_dir=embedder_save_dir,
        train_dataset=train_dataset,
        device=device,
        max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
    )

    logger.info("Finetuning pipeline completed.")
    return model
