from pathlib import Path
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from combisearch.representation.types import Turn
from combisearch.retrieval.decoders.abstract_example_set_decoder import (
    AbstractExampleListDecoder,
)
from combisearch.retrieval.decoders.top_k import TopKDecoder
from combisearch.retrieval.embedding_store import load_search_embeddings
from combisearch.retrieval.retriever_setup import (
    apply_sampling_method,
    flatten_datasets,
    input_kwargs_from_config,
    make_string_transformation,
)
from combisearch.retrieval.retrievers.abstract_example_retriever import ExampleRetriever
from combisearch.retrieval.search.hybrid import HybridSearch
from combisearch.retrieval.turn_labels import TurnLabel


class HybridRetriever(ExampleRetriever):

    def __init__(
        self, 
        datasets, 
        model_path, 
        search_index_filename:str = None, 
        sampling_method="none", 
        ratio=1.0,
        model=None, 
        search_embeddings=None,
        string_transformation: Union[str, Callable[[Turn], str]] = None,
        **kwargs
    ):

        bm25_input_kwargs = kwargs.get("bm25_input_kwargs") or input_kwargs_from_config(
            kwargs, default_input_type="dialog_context"
        )
        sbert_input_kwargs = kwargs.get("sbert_input_kwargs") or input_kwargs_from_config(
            kwargs, default_input_type="dialog_context"
        )

        if isinstance(string_transformation, str) or string_transformation is not None:
            shared_transformation = make_string_transformation(string_transformation, sbert_input_kwargs)
            self.bm25_string_transformation = shared_transformation
            self.sbert_string_transformation = shared_transformation
        else:
            self.bm25_string_transformation = make_string_transformation(None, bm25_input_kwargs)
            self.sbert_string_transformation = make_string_transformation(None, sbert_input_kwargs)

        self.data_items = flatten_datasets(datasets)

        self.model = model

        if model is None and model_path is not None:
            if isinstance(model_path, Path):
                model_path = str(model_path)
            self.model = SentenceTransformer(str(model_path))
        
        self.search_embeddings = load_search_embeddings(search_embeddings, search_index_filename)

        emb_dict = apply_sampling_method(
            self.search_embeddings,
            self.data_items,
            sampling_method=sampling_method,
            ratio=ratio,
            pre_assigned_empty_if_none=True,
            allow_sgd_pre_assigned=True,
        )
        
        self.bm25_embedding = list(map(self.data_item_to_bm25_embedding, self.data_items))

        self.bm25_emb_dict = {}
        for idx, turn in enumerate(self.data_items):
            id_turn_label = self.data_item_to_label(turn)
            self.bm25_emb_dict.update({id_turn_label: self.bm25_embedding[idx]})

        print('-----------------------------------------------------------------------------------')
        print(f" Load {search_index_filename} ")
        print(f" Number of search embeddings: {len(self.search_embeddings)} ")
        print(f" Number of data items: {len(self.data_items)} ")
        example_item = self.data_items[min(3, len(self.data_items) - 1)]
        if kwargs.get('bm25_input_kwargs') is not None:
            print(f" BM25 string Example: {self.bm25_string_transformation(example_item)} ")
        if kwargs.get('sbert_input_kwargs') is not None:
            print(f" SBERT string Example: {self.sbert_string_transformation(example_item)} ")
        print('-----------------------------------------------------------------------------------')

        self.retriever = HybridSearch(emb_dict, self.bm25_emb_dict)
        
    def data_item_to_bm25_embedding(self, data_item):
        if isinstance(data_item, list):
            return data_item
        string_query = self.bm25_string_transformation(data_item)
        embed = string_query.split()
        return embed

    def data_item_to_embedding(self, data_item):
        with torch.no_grad():
            embed = self.model.encode(self.sbert_string_transformation(
                data_item), convert_to_numpy=True).reshape(1, -1)
        return embed

    def item_to_best_examples(self, data_item, k, decoder: AbstractExampleListDecoder = TopKDecoder()):
        # the nearest neighbor is at the end
        query = self.data_item_to_embedding(data_item) if self.model is not None else None
        bm25_query = self.data_item_to_bm25_embedding(data_item)
        if isinstance(k, dict):
            k_sbert = k.get('SBERT', 0)
            k_bm25 = k.get('BM25', 0)
        else:
            k_sbert, k_bm25 = k, k
        try:
            example_generator = (
                (turn_label, score) 
                for turn_label, score in self.retriever.iterate_nearest_dialogs(query, k=k_sbert))
            bm25_example_generator = (
                (turn_label, score) 
                for turn_label, score in self.retriever.bm25_iterate_nearest_dialogs(bm25_query, k=k_bm25))
            return decoder.select_k(k=k, examples=example_generator, bm25_examples=bm25_example_generator, query_id=self.data_item_to_label(data_item))
        except StopIteration as e:
            print("ran out of examples! unable to decode")
            raise e
    
    def label_to_bm25_search_embedding(self, label: TurnLabel) -> NDArray:
        if label not in self.retriever.label_to_idx:
            raise KeyError(f"{label} not in search index")
        return self.bm25_emb_dict[label]

    def get_bm25_scores_for_query(self, query:List, all_considered_examples:List[Tuple[Turn, float]]):
        if isinstance(all_considered_examples[0], tuple):
            example_idx = [self.retriever.label_to_idx[turn_label] for turn_label, score in all_considered_examples]
            scores = self.retriever.bm25.get_scores(query)
            exmaple_score = [scores[idx] for idx in example_idx]
            return np.array(exmaple_score)
        elif isinstance(all_considered_examples[0], list):
            bm25 = BM25Okapi(corpus=all_considered_examples)
            scores = bm25.get_scores(query)  
            return np.array(scores)
