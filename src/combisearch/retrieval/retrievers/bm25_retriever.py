from typing import Callable, Union

from numpy.typing import NDArray
from combisearch.representation.types import Turn

from combisearch.retrieval.retrievers.abstract_example_retriever import ExampleRetriever
from combisearch.retrieval.decoders.abstract_example_set_decoder import AbstractExampleListDecoder
from combisearch.retrieval.retriever_setup import (
    flatten_datasets,
    input_kwargs_from_config,
    make_string_transformation,
)
from combisearch.retrieval.turn_labels import TurnLabel
from combisearch.retrieval.search.bm25 import BM25Search
from combisearch.retrieval.decoders.top_k import TopKDecoder


class BM25Retriever(ExampleRetriever):

    def __init__(
        self, 
        datasets, 
        sampling_method="none", 
        ratio=1.0,
        string_transformation: Union[str, Callable[[Turn], str]] = None,
        **kwargs
    ):

        input_kwargs = input_kwargs_from_config(kwargs, default_input_type="dialog_context")
        self.string_transformation = make_string_transformation(string_transformation, input_kwargs)
        self.data_items = flatten_datasets(datasets)
        
        self.embedding = [self.data_item_to_embedding(turn) for turn in self.data_items]

        self.search_embeddings = {}
        self.label_to_idx = {}
        for idx, turn in enumerate(self.data_items):
            id_turn_label = self.data_item_to_label(turn)
            self.search_embeddings.update({id_turn_label: self.embedding[idx]})
            self.label_to_idx.update({id_turn_label: idx})

        self.retriever = BM25Search(self.search_embeddings)
        
    def data_item_to_embedding(self, data_item) -> NDArray:
        if isinstance(data_item, list):
            return data_item
        string_query = self.string_transformation(data_item)
        embed = string_query.split()
        return embed

    def item_to_best_examples(self, data_item, k=5, decoder: AbstractExampleListDecoder = TopKDecoder()):
        # the nearest neighbor is at the end
        query = self.data_item_to_embedding(data_item)
        try:
            return decoder.select_k(k=k, examples=((self.label_to_data_item(turn_label), score) 
                                                   for turn_label, score in self.retriever.iterate_nearest_dialogs(query, k=k)))
        except StopIteration as e:
            print("ran out of examples! unable to decode")
            raise e
    
    def label_to_search_embedding(self, label: TurnLabel) -> NDArray:
        if label not in self.label_to_idx:
            raise KeyError(f"{label} not in search index")
        return self.search_embeddings[label]

