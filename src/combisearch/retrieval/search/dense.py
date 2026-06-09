from typing import Dict, Iterator, List, Tuple

import numpy as np
from scipy.spatial import KDTree


class DenseSearch:
    """Cosine-similarity dense search over a KDTree (normalized embeddings, p=2 euclidean)."""

    label_to_idx: Dict[str, int]

    def normalize(self, emb):
        return emb / np.linalg.norm(emb, axis=-1, keepdims=True)

    def __init__(self, emb_dict):
        self.emb_keys: List[str] = list(emb_dict.keys())
        self.label_to_idx = {k: i for i, k in enumerate(self.emb_keys)}
        emb_dim = emb_dict[self.emb_keys[0]].shape[-1]

        self.emb_values = np.zeros((len(self.emb_keys), emb_dim))
        for i, k in enumerate(self.emb_keys):
            self.emb_values[i] = emb_dict[k]

        # normalize for cosine distance (kdtree only support euclidean when p=2)
        self.emb_values = self.normalize(self.emb_values)
        self.kdtree = KDTree(self.emb_values)

    def iterate_nearest_dialogs(self, query_emb, k=5) -> Iterator[Tuple[str, float]]:
        query_emb = self.normalize(query_emb)
        i = 0
        fetch_size: int = k
        while i < len(self.emb_keys):
            scores, query_result = self.kdtree.query(query_emb, k=fetch_size, p=2)
            if query_result.shape == (1,):
                i += 1
                yield self.emb_keys[query_result.item()], scores.item()
            else:
                for item, score_item in zip(query_result.squeeze(0)[i:], scores.squeeze(0)[i:]):
                    i += 1
                    if item.item() >= len(self.emb_keys):
                        return
                    yield self.emb_keys[item.item()], score_item.item()
            fetch_size = min(2 * fetch_size, len(self.emb_keys))
