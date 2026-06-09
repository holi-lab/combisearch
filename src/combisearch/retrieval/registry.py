import os
import logging
import pprint
from typing import Any, Dict

from combisearch.retrieval.decoders.abstract_example_set_decoder import AbstractExampleListDecoder
from combisearch.retrieval.decoders.top_k import TopKDecoder
from combisearch.retrieval.decoders.hybrid_decoder import HybridDecoder
from combisearch.retrieval.decoders.f1_topk import F1TopKDecoder
from combisearch.retrieval.decoders.maximize_embedding_distinctness import MaximizeEmbeddingDistinctness
from combisearch.retrieval.retrievers.abstract_example_retriever import ExampleRetriever, MockRetriever
from combisearch.retrieval.retrievers.random_retriever import RandomExampleRetriever
from combisearch.retrieval.retrievers.bm25_retriever import BM25Retriever
from combisearch.retrieval.retrievers.hybrid_retriever import HybridRetriever
from combisearch.retrieval.retrievers.sgd_retriever import SGDRetriever
from combisearch.retrieval.retrievers.embed_based_retriever import EmbeddingRetriever


def _as_dict(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    raise TypeError(f"expected a dict-like config, got {type(config)}")


def get_retriever_by_type(
    retrieval_strategy: str,
    retriever_dir: str,
    retriever_args: Dict[str, Any],
    **kwargs
) -> ExampleRetriever:
    search_index_filename = kwargs.get("search_index_filename", None)
    bm25_input_kwargs = _as_dict(kwargs.get("bm25_input_kwargs", {}))
    dense_input_kwargs = _as_dict(kwargs.get("dense_input_kwargs", {}))
    strategy = retrieval_strategy.lower()
    retriever_full_path = str(retriever_dir) if retriever_dir else None

    if strategy in {"embeddingretriever", "embedding", "sbert"}:
        if retriever_full_path:
            if not os.path.exists(retriever_full_path):
                raise ValueError(f"retriever path does not exist: {retriever_full_path}")
            logging.info(f"loading retriever from {retriever_full_path}")
        return EmbeddingRetriever(**{
            "model_path": retriever_full_path,
            "search_index_filename": search_index_filename,
            **retriever_args,
            **dense_input_kwargs,
        })

    elif strategy == "no_retriever":
        return MockRetriever()

    elif strategy == "random":
        return RandomExampleRetriever(**retriever_args)

    elif strategy == "bm25":
        return BM25Retriever(**{**retriever_args, **bm25_input_kwargs})

    elif strategy == "hybrid":
        return HybridRetriever(**{
            "model_path": retriever_full_path,
            "search_index_filename": search_index_filename,
            **retriever_args,
            "sbert_input_kwargs": dense_input_kwargs,
            "bm25_input_kwargs": bm25_input_kwargs,
        })

    elif strategy == "sgd":
        return SGDRetriever(**{
            "model_path": retriever_full_path,
            "search_index_filename": search_index_filename,
            **retriever_args,
            **dense_input_kwargs,
        })

    else:
        raise ValueError(f"unknown retriever type: {retrieval_strategy}")


def get_decoder_by_type(decoder_config: Any, retriever: ExampleRetriever) -> AbstractExampleListDecoder:
    decoder_config_dict = _as_dict(decoder_config) if decoder_config else None

    if not decoder_config_dict or decoder_config_dict.get('decoder_type', 'top_k') == 'top_k':
        return TopKDecoder()

    elif decoder_config_dict['decoder_type'] == 'max_emb_distance':
        if not isinstance(retriever, EmbeddingRetriever):
            raise ValueError(f"cannot maximize embedding distance with a retriever of type: {type(retriever)}")
        return MaximizeEmbeddingDistinctness(
            retriever=retriever,
            from_n_possible=decoder_config_dict['from_n_possible'],
            discount_factor=decoder_config_dict['discount_factor'],
        )

    elif decoder_config_dict['decoder_type'] == 'hybrid':
        if not isinstance(retriever, HybridRetriever):
            raise ValueError(
                "decoder_type='hybrid' requires a HybridRetriever (SGD is dense-only). "
                f"Got retriever type: {type(retriever)}"
            )
        pprint.pprint(f"Retriever type: {type(retriever)}")
        return HybridDecoder(
            retriever=retriever,
            from_n_possible=decoder_config_dict.pop('from_n_possible', 100),
            discount_factor=decoder_config_dict.pop('discount_factor', 0.2),
            operation=decoder_config_dict.pop('operation', 'multiply'),
            zscore=decoder_config_dict.pop('zscore', True),
            decoding_logic=decoder_config_dict.pop('decoding_logic', 'top_k_round_robin'),
            **decoder_config_dict,
        )

    elif decoder_config_dict['decoder_type'] == 'f1_top_k':
        if not isinstance(retriever, EmbeddingRetriever):
            raise ValueError(f"cannot maximize embedding distance with a retriever of type: {type(retriever)}")
        return F1TopKDecoder(retriever=retriever)

    else:
        raise ValueError(f"Unable to construct decoder: {pprint.pformat(decoder_config)}")
