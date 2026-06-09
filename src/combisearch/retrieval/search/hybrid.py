from typing import Dict, Iterator, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from scipy.spatial import KDTree


class HybridSearch:
    """Dense (KDTree, cosine) + sparse (BM25Okapi) search over the same key set.

    Used by both MultiWOZ (HybridRetriever).
    """

    label_to_idx: Dict[str, int]

    def normalize(self, emb):
        return emb / np.linalg.norm(emb, axis=-1, keepdims=True)

    def __init__(self, emb_dict, bm25_emb_dict):
        self.emb_keys: List[str] = list(emb_dict.keys())
        self.label_to_idx = {k: i for i, k in enumerate(self.emb_keys)}

        emb_dim = emb_dict[self.emb_keys[0]].shape[-1] if emb_dict else 0
        self.emb_values = np.zeros((len(self.emb_keys), emb_dim))
        for i, k in enumerate(self.emb_keys):
            self.emb_values[i] = emb_dict[k]

        # normalize for cosine distance (KDTree only supports euclidean with p=2)
        self.emb_values = self.normalize(self.emb_values)
        self.kdtree = KDTree(self.emb_values)
        self.bm25 = BM25Okapi(corpus=list(bm25_emb_dict.values()))

    def bm25_iterate_nearest_dialogs(self, query, k=5) -> Iterator[Tuple[str, float]]:
        scores = self.bm25.get_scores(query)
        idx_score = [(i, score) for i, score in enumerate(scores)]
        sorted_idx_score = sorted(idx_score, key=lambda x: x[1], reverse=True)
        for idx, score in sorted_idx_score:
            if idx >= len(self.emb_keys):
                return
            yield self.emb_keys[idx], score

    def iterate_nearest_dialogs(self, query_emb, k=5) -> Iterator[Tuple[str, float]]:
        if query_emb is None:
            return
        query_emb = self.normalize(query_emb)
        scores, query_result = self.kdtree.query(query_emb, k=len(self.emb_keys), p=2)

        # KDTree.query returns results sorted by distance ascending; break
        # ties deterministically by index so identical inputs yield in the
        # same order across runs.
        order = np.lexsort((query_result, scores)).squeeze()
        query_result = query_result[:, order]
        scores = scores[:, order]

        if query_result.shape == (1,):
            yield self.emb_keys[query_result.item()], scores.item()
            return
        for idx, score in zip(query_result.squeeze(0), scores.squeeze(0)):
            if idx.item() >= len(self.emb_keys):
                return
            yield self.emb_keys[idx.item()], score.item()
