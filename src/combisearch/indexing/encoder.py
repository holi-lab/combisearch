"""Resolve and load the encoder used for dense index building."""

from __future__ import annotations

from pathlib import Path

from combisearch.config.paths import embedding_model_tag, resolve_canonical_path

DEFAULT_INDEX_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
SPECIAL_TOKENS: tuple[str, ...] = ("[USER]", "[SYS]", "[CONTEXT]", "[BS]")


def _looks_like_path(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return Path(text).is_absolute() or text.startswith((".", "..", "outputs/"))


def _pretrained_dir(model_name: str) -> Path:
    return resolve_canonical_path(
        f"outputs/retrievers/{embedding_model_tag(model_name)}/pretrained"
    )


def _has_weights(path: Path) -> bool:
    return (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()

def resolve_encoder_path(model_name: str, *, prepare_if_missing: bool = True) -> Path:
    """Return a local directory that holds the AutoModel encoder."""
    if _looks_like_path(model_name):
        return resolve_canonical_path(model_name)

    target = _pretrained_dir(model_name)
    if _has_weights(target) or not prepare_if_missing:
        return target
    return _download_and_extend(model_name, target)


def _download_and_extend(model_name: str, target: Path) -> Path:
    """Download the AutoModel encoder, add DST special tokens, and save."""
    from transformers import AutoModel, AutoTokenizer
    target.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokenizer.add_tokens(list(SPECIAL_TOKENS), special_tokens=True)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))
    return target


def load_encoder(encoder_path: Path):
    """Load the tokenizer/model pair used for pretrained dense indexes."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(encoder_path))
    model = AutoModel.from_pretrained(str(encoder_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model
