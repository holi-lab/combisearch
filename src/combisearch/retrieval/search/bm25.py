from typing import Dict, Iterator, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


class BM25Search:
    label_to_idx: Dict[str, int]

    def normalize(self, emb):
        return emb

    def __init__(self, emb_dict):
        self.bm25 = BM25Okapi(corpus=list(emb_dict.values()))
        self.emb_keys: List[str] = list(emb_dict.keys())

    def iterate_nearest_dialogs(self, query_emb, k=5) -> Iterator[Tuple[str, float]]:
        query_emb = self.normalize(query_emb)
        i = 0
        scores = self.bm25.get_scores(query_emb)
        score_idx_dict = [[i,score] for i, score in enumerate(scores)]
        sorted_scores = sorted(score_idx_dict, key=lambda x: x[1], reverse=True)
        query_result = np.array([[i for (i, score) in sorted_scores]])
        sorted_scores = np.array([[score for (i, score) in sorted_scores]])
        while i < len(self.emb_keys):
            if query_result.shape == (1,):
                i += 1
                yield self.emb_keys[query_result.item()], sorted_scores.item()
                if i >= len(self.emb_keys):
                    break
            else:
                for item, score_item in zip(query_result.squeeze(0)[i:], sorted_scores.squeeze(0)[i:]):
                    i += 1
                    if item.item() >= len(self.emb_keys):
                        return
                    yield self.emb_keys[item.item()], score_item.item()
                if i >= len(self.emb_keys):
                    break
