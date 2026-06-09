from typing import Callable, Union

import torch
from numpy.typing import NDArray
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
from combisearch.retrieval.search.dense import DenseSearch


class EmbeddingRetriever(ExampleRetriever):

    def __init__(
        self, 
        datasets, 
        model_path, 
        search_index_filename: str = None,
        sampling_method="none", 
        ratio=1.0,
        model=None,
        search_embeddings=None,
        full_history=False,
        string_transformation: Union[str, Callable[[Turn], str]] = None,
        **kwargs
    ):

        input_kwargs = input_kwargs_from_config(kwargs, default_input_type="dialog_context")
        self.string_transformation = make_string_transformation(string_transformation, input_kwargs)
        self.data_items = flatten_datasets(datasets)

        self.model = model

        if model is None:
            self.model = SentenceTransformer(model_path)

        self.search_embeddings = load_search_embeddings(search_embeddings, search_index_filename)

        emb_dict = apply_sampling_method(
            self.search_embeddings,
            self.data_items,
            sampling_method=sampling_method,
            ratio=ratio,
        )
        self.retriever = DenseSearch(emb_dict)

    def data_item_to_embedding(self, data_item, **kwargs) -> NDArray:
        with torch.no_grad():
            embed = self.model.encode(self.string_transformation(
                data_item), convert_to_numpy=True).reshape(1, -1)
        return embed

    def item_to_best_examples(self, data_item, k=5, decoder: AbstractExampleListDecoder = TopKDecoder()):
        # the nearest neighbor is at the end
        query = self.data_item_to_embedding(data_item)
        try:
            return decoder.select_k(k=k, examples=((self.label_to_data_item(turn_label), score) 
                                                for turn_label, score in self.retriever.iterate_nearest_dialogs(query, k=k)), query=data_item)
        except StopIteration as e:
            print("ran out of examples! unable to decode")
            raise e
