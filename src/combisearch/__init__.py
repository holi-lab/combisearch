"""CombiSearch — combinatorial in-context example scoring for dialogue state tracking.

ACL 2025 long paper:
    "Improving Dialogue State Tracking through Combinatorial Search for In-Context Examples"
    https://aclanthology.org/2025.acl-long.1393/

Public modules
--------------
- combisearch.pipeline   : DST inference (DSTExperiment) and the CombiSearch
                           data-generation algorithm (DataGenerationExperiment, paper §3.1)
- combisearch.retrieval  : retrievers (BM25, embedding, hybrid, SGD) and decoders
- combisearch.training   : retriever fine-tuning (paper §3.2)
- combisearch.evaluation : DST metrics + retriever evaluation + error analysis
- combisearch.prompting  : prompt construction and completion parsing
- combisearch.indexing   : building dense search indexes
- combisearch.llm        : OpenAI / Llama clients
- combisearch.config     : config schema, path resolution
- combisearch.cli        : command-line entry points

The convenience symbols below are lazily imported so ``import combisearch``
stays cheap.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

_LAZY: dict[str, str] = {
    "DSTExperiment": "combisearch.pipeline.dst",
    "DataGenerationExperiment": "combisearch.pipeline.data_generation",
    "HybridRetriever": "combisearch.retrieval.retrievers.hybrid_retriever",
    "BM25Retriever": "combisearch.retrieval.retrievers.bm25_retriever",
    "EmbeddingRetriever": "combisearch.retrieval.retrievers.embed_based_retriever",
    "HybridDecoder": "combisearch.retrieval.decoders.hybrid_decoder",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'combisearch' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*_LAZY, "__version__"})


__all__ = [*_LAZY.keys(), "__version__"]
