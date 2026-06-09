import numpy as np


def load_search_embeddings(search_embeddings, search_index_filename):
    if search_embeddings is not None:
        return search_embeddings
    if search_index_filename:
        return np.load(search_index_filename, allow_pickle=True).item()
    raise ValueError("unable to instantiate a retreiver without embeddings. Supply pre-loaded search_embeddings"
                     " or a search_index_filename")
