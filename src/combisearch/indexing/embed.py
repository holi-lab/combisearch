"""Encode a TurnDataset and persist the result."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .datasets import TurnDataset


def save_embeddings(
    model,
    dataset: TurnDataset,
    output_file: Path,
    *,
    batch_size: int = 64,
    normalize: bool = True,
    show_progress: bool = True,
) -> None:
    if not dataset:
        raise ValueError(f"cannot save embeddings for an empty dataset: {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    embeddings = encode_turns(
        model,
        dataset.turn_utts,
        batch_size=batch_size,
        normalize=normalize,
        show_progress=show_progress,
    )

    output = {label: embeddings[i:i + 1] for i, label in enumerate(dataset.turn_labels)}
    np.save(output_file, output)

    keys_file = output_file.with_name(f"{output_file.stem}_keys.txt")
    with keys_file.open("w") as f:
        json.dump(
            [{label: utt} for label, utt in zip(dataset.turn_labels, dataset.turn_utts)],
            f,
            indent=2,
        )


def encode_turns(
    model,
    texts: list[str],
    *,
    batch_size: int,
    normalize: bool,
    show_progress: bool,
) -> np.ndarray:
    """Encode text with either SentenceTransformer or AutoModel/tokenizer pair."""
    if hasattr(model, "encode"):
        return model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
        )

    tokenizer, auto_model = model
    return _encode_with_auto_model(
        tokenizer,
        auto_model,
        texts,
        batch_size=batch_size,
        normalize=normalize,
        show_progress=show_progress,
    )


def _encode_with_auto_model(
    tokenizer,
    model,
    texts: list[str],
    *,
    batch_size: int,
    normalize: bool,
    show_progress: bool,
) -> np.ndarray:
    device = next(model.parameters()).device
    outputs = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=(len(texts) + batch_size - 1) // batch_size)

    with torch.no_grad():
        for start in iterator:
            batch = texts[start:start + batch_size]
            encoded_input = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            input_ids = encoded_input["input_ids"].to(device)
            attention_mask = encoded_input["attention_mask"].to(device)
            model_output = model(input_ids, attention_mask)
            embeddings = mean_pooling(model_output, attention_mask)
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            outputs.append(embeddings.detach().cpu())

    return torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9,
    )
