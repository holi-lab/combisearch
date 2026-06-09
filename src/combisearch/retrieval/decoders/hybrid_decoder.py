from typing import Tuple, Iterator, List, Dict

import numpy as np
from numpy.typing import NDArray
from combisearch.representation.types import Turn
from sklearn.metrics.pairwise import cosine_similarity

from combisearch.retrieval.decoders.abstract_example_set_decoder import AbstractExampleListDecoder
from combisearch.retrieval.retrievers.embed_based_retriever import EmbeddingRetriever

class HybridDecoder(AbstractExampleListDecoder):
    """Fuses BM25 and SBERT results; fusion strategy selected by self.decoding_logic in select_k."""
    from_n_possible: int
    discount_factor: float
    retriever: EmbeddingRetriever

    def __init__(self, retriever: EmbeddingRetriever, from_n_possible: int, discount_factor: float = 0.1, **kwargs) -> None:
        self.from_n_possible = from_n_possible
        self.discount_factor = discount_factor
        self.retriever = retriever
        self.zscore = kwargs.get('zscore', False)
        self.decoding_logic = kwargs.get('decoding_logic', 'top_k_round_robin')
        self.alpha = kwargs.get('alpha', 0.5)
        self.constant = kwargs.get('constant', 60)
        self.is_inference=kwargs.get('is_inference', False)

    def select_k(self, examples: Iterator[Tuple[Turn, float]],
                 bm25_examples:Iterator[Tuple[Turn, float]], k: int, **kwargs) -> List[Turn]:
        if isinstance(k, dict):
            self.bm25_k = k.get('BM25', 0)
            self.sbert_k = k.get('SBERT', 0)
        else:
            self.bm25_k = k
            self.sbert_k = k

        if self.is_inference:
            self.sbert_pool_size = self.sbert_k*2
            self.bm25_pool_size = self.bm25_k*2
        else:
            self.sbert_pool_size = self.from_n_possible
            self.bm25_pool_size = self.from_n_possible
        
        query_id = kwargs.get('query_id', None)
        query_doc_id = query_id.split('_turn')[0]
        
        # Select up to "from N possible" in the iterator
        sbert_examples: List[Tuple[Turn, float]] = [(turn_label, score) for turn_label, score in examples if turn_label.split('_turn')[0] != query_doc_id]
        bm25_examples: Dict[Tuple[Turn, float]] = {turn_label: score for turn_label, score in bm25_examples if turn_label.split('_turn')[0] != query_doc_id}
        
        # scores from retriever are euclidean distances, of unit vectors (range 0-2). cos(y, z) = 1-.5*euc(y, z)^2
        # We initialize example_scores with the cosine similarity of each example e to x, the input turn (implicit from
        #  order of argument examples to select_k).
        sbert_score = np.array([(1-0.5*(score**2)) for _, score in sbert_examples])
        bm25_score = np.array(list(bm25_examples.values()))
        
        if self.decoding_logic == 'div_topk_round_robin':
            sbert_examples, bm25_examples = self.union_examples_update_score(sbert_examples, bm25_examples, sbert_score, bm25_score)
            return self.div_topk_round_robin(sbert_examples, bm25_examples)

        elif self.decoding_logic == 'multiply_div_top_k':
            sbert_score, bm25_score = self.transform_distribution(sbert_score, bm25_score)
            sbert_examples, bm25_examples = self.union_examples_update_score(sbert_examples, bm25_examples, sbert_score, bm25_score)
            return self.multiply_div_top_k(sbert_examples, bm25_examples)
        
        elif self.decoding_logic == 'multiply_top_k':
            sbert_score, bm25_score = self.transform_distribution(sbert_score, bm25_score)
            sbert_examples, bm25_examples = self.union_examples_update_score(sbert_examples, bm25_examples, sbert_score, bm25_score)
            return self.multiply_top_k(sbert_examples, bm25_examples)       

        elif self.decoding_logic == 'round_robin_div_top_k':
            sbert_score, bm25_score = self.transform_distribution(sbert_score, bm25_score)
            sbert_examples, bm25_examples = self.union_examples_update_score(sbert_examples, bm25_examples, sbert_score, bm25_score)
            return self.round_robin_div_top_k(sbert_examples, bm25_examples)
        
        elif self.decoding_logic == 'top_k_round_robin':
            sbert_examples, bm25_examples = self.union_examples_update_score(sbert_examples, bm25_examples, sbert_score, bm25_score)
            return self.top_k_round_robin(sbert_examples, bm25_examples) 

        elif self.decoding_logic == 'rrf':
            return self.reciprocal_rank_fusion(sbert_examples, bm25_examples, self.bm25_k+self.sbert_k, self.constant) 
    
        elif self.decoding_logic == 'weighted_sum':
            sbert_score, bm25_score = self.transform_distribution(sbert_score, bm25_score)
            sbert_examples, bm25_examples = self.union_examples_update_score(sbert_examples, bm25_examples, sbert_score, bm25_score)
            return self.weighted_sum(sbert_examples, bm25_examples, self.alpha)
        
    def weighted_sum(
        self,
        sbert_examples: List[Tuple[Turn, float]],
        bm25_examples: List[Tuple[Turn, float]],
        alpha: float = 0.5
    ) -> List[Turn]:
        """Combine normalized scores as alpha*sbert + (1-alpha)*bm25, return top-k Turns low to high.

        Assumes every turn_label in the union exists in both dicts; raises KeyError otherwise.
        """
        sbert_examples = dict(sbert_examples)
        bm25_examples = dict(bm25_examples)

        union_set = set(sbert_examples).union(set(bm25_examples))

        total_examples = {}
        for turn_label in union_set:
            total_examples[turn_label] = alpha * sbert_examples[turn_label] + (1-alpha) * bm25_examples[turn_label]
        
        total_examples = sorted(total_examples.items(), key=lambda item: item[1], reverse=True) # [higt to low]
        scores = [score for _, score in total_examples]
        assert np.all(np.diff(np.array(scores)) <= 0)  # verifies they are decreasing as expected
        
        result = [self.retriever.label_to_data_item(turn_label) for turn_label, _ in total_examples[:self.bm25_k+self.sbert_k]]  # top k

        return result[::-1] # return [low to high]

    def reciprocal_rank_fusion(
        self,
        sbert_examples: List[Tuple[Turn, float]],
        bm25_examples: Dict[Turn, float],
        k: int=10,
        constant:int = 60
    ) -> List[Turn]:
        """Reciprocal Rank Fusion: score each turn_label as sum over rankers of 1/(constant + rank).

        Ranks are 0-indexed. Returns top-k Turns ordered low to high.
        """
        sbert_rank = {turn_label: idx for idx, (turn_label, _) in enumerate(sbert_examples)}
        bm25_rank = {turn_label: idx for idx, (turn_label, _) in enumerate(bm25_examples.items())}

        rrf_rank = {}
        for turn_label in sbert_rank:
            rrf_rank[turn_label] = 1/(sbert_rank[turn_label]+constant)
        for turn_label in bm25_rank:
            rrf_rank[turn_label] += 1/(bm25_rank[turn_label]+constant)

        rrf_rank = sorted(rrf_rank.items(), key=lambda x: x[1], reverse=True) # [high to low]
        result = [self.retriever.label_to_data_item(turn_label) for turn_label, _ in rrf_rank[:k]]

        return result[::-1] # return [low to high]
    
    def div_topk_round_robin(
        self,
        sbert_examples: List[Tuple[Turn, float]],
        bm25_examples: Dict[Turn, float]
    ) -> List[Turn]:
        """Diversity-decode SBERT and BM25 separately, then round-robin merge (deduped). Returns Turns low to high."""
        sbert_examples = self.sbert_div(sbert_examples, self.sbert_pool_size)[::-1]
        bm25_examples = self.bm25_div(bm25_examples, self.bm25_pool_size)[::-1]
    
        # Round-robin of the two lists
        result = {}
        while len(result) < self.bm25_k+self.sbert_k:
            while True:
                try:
                    turn_label = sbert_examples.pop(0)
                    if turn_label not in result:
                        result.update({turn_label: 0})
                        break
                except IndexError:
                    break
            while True:
                try:
                    turn_label = bm25_examples.pop(0)
                    if turn_label not in result:
                        result.update({turn_label: 0})
                        break
                except IndexError:
                    break
        
        return [self.retriever.label_to_data_item(turn_label) for turn_label in result][::-1] # [low to high]

    def multiply_div_top_k(
        self,
        sbert_examples: List[Tuple[Turn, float]],
        bm25_examples: List[Tuple[Turn, float]]
    ) -> List[Turn]:
        """Multiply normalized SBERT*BM25 scores, then diversity-decode.

        Discount combines both SBERT (cosine) and BM25 similarity of the selected example to the rest.
        Returns top-k Turns low to high.
        """

        # storing these as a list of indices so they can be interpreted: 0-(k-1) would mean our discount factor had no
        # impact on the score.
        result : List[int] = []
        sbert_examples = dict(sbert_examples)
        bm25_examples = dict(bm25_examples)

        union_set = set(sbert_examples).union(set(bm25_examples))

        total_examples = {}
        for turn_label in union_set:
                total_examples[turn_label] = sbert_examples[turn_label] * bm25_examples[turn_label]
        
        total_examples = sorted(total_examples.items(), key=lambda x: x[1], reverse=True)   # [high to low]

        sbert_embeddings: NDArray = np.asarray([
            self.retriever.label_to_search_embedding(turn_label) for turn_label in sbert_examples
        ])
        bm25_embeddings: NDArray = [
            self.retriever.label_to_bm25_search_embedding(turn_label) for turn_label in bm25_examples
        ]

        example_scores = []
        idx_label_dict = {}
        for idx, (turn_label, score) in enumerate(total_examples):
            idx_label_dict.update({idx: turn_label})
            example_scores.append(score)
        
        example_scores = np.array(example_scores)
        assert np.all(np.diff(example_scores) <= 0)  # verifies they are decreasing as expected

        while len(result) < self.bm25_k+self.sbert_k:
            # find the current best scoring example
            best_idx: int = np.argmax(example_scores).item()
            example_scores[best_idx] = -np.inf
            result.append(best_idx)
            # Update the scores. The worst-case decrease in score is defined by discount_factor.
            best_emb: NDArray = sbert_embeddings[best_idx]
            best_bm25_emb:NDArray = bm25_embeddings[best_idx]

            discount = self.standardize(bm25_score=self.retriever.get_bm25_scores_for_query(best_bm25_emb, bm25_embeddings))
            discount *= self.standardize(sbert_score=cosine_similarity(best_emb[None, :], sbert_embeddings).squeeze(0))
            
            discount *= self.discount_factor 
            example_scores = example_scores - discount

        return [self.retriever.label_to_data_item(idx_label_dict[i]) for i in result][::-1] # [low to high]
    
    def multiply_top_k(
        self,
        sbert_examples: List[Tuple[Turn, float]],
        bm25_examples: List[Tuple[Turn, float]]
    ) -> List[Turn]:
        """Multiply normalized SBERT*BM25 scores, sort, return top-k Turns low to high (no diversity).

        Assumes every turn_label in the union exists in both dicts; raises KeyError otherwise.
        """
        sbert_examples = dict(sbert_examples)
        bm25_examples = dict(bm25_examples)

        union_set = set(sbert_examples).union(set(bm25_examples))

        total_examples = {}
        for turn_label in union_set:
            total_examples[turn_label] = sbert_examples[turn_label] * bm25_examples[turn_label]

        total_examples = sorted(total_examples.items(), key=lambda item: item[1], reverse=True) # [higt to low]
        scores = [score for _, score in total_examples]
        assert np.all(np.diff(np.array(scores)) <= 0)  # verifies they are decreasing as expected
        
        result = [self.retriever.label_to_data_item(turn_label) for turn_label, _ in total_examples[:self.bm25_k+self.sbert_k]]  # top k

        return result[::-1] # return [low to high]

    def round_robin_div_top_k(
        self,
        sbert_examples: List[Tuple[Turn, float]],
        bm25_examples: List[Tuple[Turn, float]]
    ) -> List[Turn]:
        """Round-robin merge SBERT and BM25, then diversity-decode.

        Discount uses each example's own embedding type (SBERT ndarray -> cosine, BM25 list -> bm25 score).
        Returns top-k Turns low to high.
        """
        total_length = len(sbert_examples)
        
        # shape: (example_idx, embedding size)
        sbert_all_embeddings: NDArray = np.asarray([
            self.retriever.label_to_search_embedding(turn_label) for (turn_label, _) in sbert_examples
        ])
        bm25_all_embeddings = [
            self.retriever.label_to_bm25_search_embedding(turn_label) for (turn_label, _) in bm25_examples
        ]

        # Round-robin of the two lists
        example_scores_embedding = {}
        s_idx, b_idx = 0, 0
        while len(example_scores_embedding) < total_length:
            while s_idx < len(sbert_examples):
                turn_label, s_score = sbert_examples[s_idx]
                if turn_label not in example_scores_embedding:
                    example_scores_embedding.update({turn_label: (s_score, sbert_all_embeddings[s_idx])})
                    s_idx += 1
                    break
                s_idx += 1
                
            while b_idx < len(bm25_examples):
                turn_label, b_score = bm25_examples[b_idx]
                if turn_label not in example_scores_embedding:
                    example_scores_embedding.update({turn_label: (b_score, bm25_all_embeddings[b_idx])})
                    b_idx += 1
                    break
                b_idx += 1
                
        example_scores_embedding = sorted(example_scores_embedding.items(), key=lambda x: x[1][0], reverse=True) # [high to low]
        sbert_all_embeddings: NDArray = np.asarray([
            self.retriever.label_to_search_embedding(turn_label) for (turn_label, _) in example_scores_embedding
        ])
        bm25_all_embeddings = [
            self.retriever.label_to_bm25_search_embedding(turn_label) for (turn_label, _) in example_scores_embedding
        ]

        idx_label_dict = {}
        example_scores = []
        embeddings = []
        for idx, (turn_label, (score, emb)) in enumerate(example_scores_embedding):
            idx_label_dict.update({idx: turn_label})
            example_scores.append(score)
            embeddings.append(emb)
        
        example_scores = np.array(example_scores)
        assert np.all(np.diff(example_scores) <= 0)  # verifies they are decreasing as expected

        # storing these as a list of indices so they can be interpreted: 0-(k-1) would mean our discount factor had no
        # impact on the score.
        result : List[int] = []
        while len(result) < self.bm25_k+self.sbert_k:
            # find the current best scoring example
            best_idx: int = np.argmax(example_scores).item()
            example_scores[best_idx] = -np.inf
            result.append(best_idx)
            # Update the scores. The worst-case decrease in score is defined by discount_factor.
            best_emb = embeddings[best_idx]
            
            if isinstance(best_emb, np.ndarray):
                div_score = cosine_similarity(best_emb[None, :], sbert_all_embeddings).squeeze(0)
                div_score = self.standardize(sbert_score=div_score)
                discount = self.discount_factor * div_score
            else:
                div_score = self.retriever.get_bm25_scores_for_query(best_emb, bm25_all_embeddings)
                div_score = self.standardize(bm25_score=div_score)
                discount = self.discount_factor * div_score
            
            example_scores = example_scores - discount

        return [self.retriever.label_to_data_item(idx_label_dict[i]) for i in result][::-1] # [low to high]
    
    def top_k_round_robin(
        self,
        sbert_examples: List[Tuple[Turn, float]],
        bm25_examples: List[Tuple[Turn, float]]
    ) -> List[Turn]:
        """Round-robin merge SBERT and BM25, skipping duplicates, until k collected. Returns Turns low to high."""
        # Round-robin of the two lists
        result = {}
        s_idx, b_idx = 0, 0
        while len(result) < self.bm25_k+self.sbert_k:
            while s_idx < len(sbert_examples):
                turn_label, _ = sbert_examples[s_idx]
                if turn_label not in result:
                    result.update({turn_label: 0})
                    s_idx += 1
                    break
                s_idx += 1
            while b_idx < len(bm25_examples):
                turn_label, _ = bm25_examples[b_idx]
                if turn_label not in result:
                    result.update({turn_label: 0})
                    b_idx += 1
                    break
                b_idx += 1

        return [self.retriever.label_to_data_item(turn_label) for turn_label in result][::-1] # [low to high]
    
    def bm25_div(self, bm25_examples, k = 10):
        """Diversity-decode BM25 results: greedily pick the best, then discount remaining by
        discount_factor * bm25_similarity to the picked example. Returns k turn_labels low to high.
        """
        # shape: (example_idx, embedding size)
        bm25_embeddings = [
            self.retriever.label_to_bm25_search_embedding(turn_label)
            for turn_label, score in bm25_examples
        ]
        if len(bm25_examples) == 0:
            return []
    
        # storing these as a list of indices so they can be interpreted: 0-(k-1) would mean our discount factor had no
        # impact on the score.
        result: List[int] = []

        # scores from retriever are euclidean distances, of unit vectors (range 0-2). cos(y, z) = 1-.5*euc(y, z)^2
        # We initialize example_scores with the cosine similarity of each example e to x, the input turn (implicit from
        #  order of argument examples to select_k).
        example_scores: NDArray = np.asarray([score for turn, score in bm25_examples])
        assert np.all(np.diff(example_scores) <= 0)  # verifies they are decreasing as expected
        while len(result) < k:
            # find the current best scoring example
            best_idx: int = np.argmax(example_scores).item()
            example_scores[best_idx] = -np.inf
            result.append(best_idx)
            # Update the scores. The worst-case decrease in score is defined by discount_factor.
            best_emb: NDArray = bm25_embeddings[best_idx]
            discount: NDArray = self.discount_factor * self.retriever.get_bm25_scores_for_query(best_emb, bm25_embeddings)
            example_scores = example_scores - discount
        return [bm25_examples[i][0] for i in result][::-1]  # [low to high]

    def sbert_div(self, examples, k = 10):
        """Diversity-decode SBERT results: greedily pick the best, then discount remaining by
        discount_factor * cosine_similarity to the picked example. Returns k turn_labels low to high.
        """
        # shape: (example_idx, embedding size)
        sbert_embeddings: NDArray = np.asarray([
            self.retriever.label_to_search_embedding(turn_label) 
            for turn_label, score in examples
        ])
        if len(examples) == 0:
            return []
        # storing these as a list of indices so they can be interpreted: 0-(k-1) would mean our discount factor had no
        # impact on the score.
        result: List[int] = []

        # scores from retriever are euclidean distances, of unit vectors (range 0-2). cos(y, z) = 1-.5*euc(y, z)^2
        # We initialize example_scores with the cosine similarity of each example e to x, the input turn (implicit from
        #  order of argument examples to select_k).
        example_scores: NDArray = np.asarray([score for turn, score in examples])
        assert np.all(np.diff(example_scores) <= 0)  # verifies they are decreasing as expected
        while len(result) < k:
            # find the current best scoring example
            best_idx: int = np.argmax(example_scores).item()
            example_scores[best_idx] = -np.inf
            result.append(best_idx)
            # Update the scores. The worst-case decrease in score is defined by discount_factor.
            best_emb: NDArray = sbert_embeddings[best_idx]
            discount: NDArray = self.discount_factor * cosine_similarity(best_emb[None, :], sbert_embeddings).squeeze(0)
            example_scores = example_scores - discount
        return [examples[i][0] for i in result][::-1]   # [low to high]
    
    def union_examples_update_score(self, sbert_examples, bm25_examples, cos_score=None, bm25_score=None):
        """Union the two pools (each truncated to its pool size), attach normalized scores, re-sort high to low.

        cos_score/bm25_score override the raw scores if given. Returns (sbert_examples, bm25_examples) as lists,
        which may differ in length since a turn_label can exist in only one retriever's results.
        """
        if cos_score is None:
            cos_score = np.array([score for turn, score in sbert_examples])
        sbert_dict = {
                turn_label: c_score 
            for (turn_label, _), c_score in zip(sbert_examples, cos_score)
        }

        if bm25_score is None:
            bm25_score = np.array([score for turn, score in bm25_examples])
        bm25_dict = {
                turn_label: b_score 
            for (turn_label, _), b_score in zip(bm25_examples.items(), bm25_score)
        }
        
        sbert_set = set(list(sbert_dict.keys())[:self.sbert_pool_size])
        bm25_set = set(list(bm25_dict.keys())[:self.bm25_pool_size])
        union_set = sbert_set.union(bm25_set)
        
        sbert_examples = [(turn_label, sbert_dict[turn_label]) for turn_label in union_set if turn_label in sbert_dict]
        bm25_examples = [(turn_label, bm25_dict[turn_label]) for turn_label in union_set if turn_label in bm25_dict]

        sbert_examples = list(sorted(sbert_examples, key=lambda x: x[1], reverse=True)) # [high to low]
        bm25_examples = list(sorted(bm25_examples, key=lambda x: x[1], reverse=True)) # [high to low]

        assert len(sbert_examples) == len(union_set) or len(bm25_examples) == len(union_set)
        return sbert_examples, bm25_examples

    def transform_distribution(self, sbert_score, bm25_score):
        """Map SBERT cosine [-1,1] to [0,1], then normalize both scores by z-score (if self.zscore) or min-max.

        Side effect: stores the subtrahend/divisor used later by standardize(). Returns (sbert_score, bm25_score).
        """
        # tf-idf => [35-4] => z-score
        # cos sim => [-1 1] => [0 2] => [0 1] => z-score
        sbert_score = (sbert_score+1)/2     # [0, 1]
        if self.zscore:
            self.sbert_subtrahend, self.sbert_divisor= np.mean(sbert_score), np.std(sbert_score)
            self.bm25_subtrahend, self.bm25_divisor = np.mean(bm25_score), np.std(bm25_score)
        else:
            self.sbert_subtrahend, self.sbert_divisor = np.min(sbert_score), (np.max(sbert_score) - np.min(sbert_score))
            self.bm25_subtrahend, self.bm25_divisor = np.min(bm25_score), (np.max(bm25_score) - np.min(bm25_score))
            
        sbert_score = (sbert_score - self.sbert_subtrahend)/self.sbert_divisor
        bm25_score = (bm25_score-self.bm25_subtrahend)/self.bm25_divisor
        return sbert_score, bm25_score

    def standardize(self, sbert_score=None, bm25_score=None):
        """Normalize new scores with the subtrahend/divisor set by transform_distribution() (must run first).

        SBERT path maps cosine [-1,1] to [0,1] first. Pass exactly one of sbert_score/bm25_score; raises ValueError otherwise.
        """
        if sbert_score is None and bm25_score is None:
            raise ValueError("Both scores provided to standard")
        if sbert_score is not None: 
            sbert_score = (sbert_score+1)/2     # [0, 1]
            return (sbert_score - self.sbert_subtrahend) / self.sbert_divisor
        elif bm25_score is not None:
            return (bm25_score - self.bm25_subtrahend) / self.bm25_divisor
        else:   
            raise ValueError("No score provided to standard")