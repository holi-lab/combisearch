"""InfoNCE training-sample loader with mixed hard + easy negatives.

For each anchor turn we pick positive examples from the top of the
CombiSearch-scored pool, the hardest half of the remaining pool as
state-similar negatives, and the rest as random negatives drawn from the
training set. Each row becomes a ``sentence_transformers.InputExample``
ready for ``MultipleNegativesRankingLoss``-style training.
"""

import random

import numpy as np
from sentence_transformers import InputExample
from tqdm import tqdm

from combisearch.evaluation.retrieval import compute_sv_sim
from combisearch.training.retriever.data import ScoredExampleDataset, MWDataset
from combisearch.retrieval.retrievers.index_based_retriever import IndexRetriever


class InfoNCEDataloader:
    def __init__(self, f1_set: ScoredExampleDataset, score_type: str = "score_delta", neg_size: int = 15):
        self.f1_set = f1_set
        self.score_type = score_type
        self.neg_size = neg_size

        self.hard_neg_size = int(np.ceil(neg_size / 2))
        self.easy_neg_size = neg_size - self.hard_neg_size
        self.turn_label_to_idx = {turn: idx for idx, turn in enumerate(self.f1_set.turn_labels)}

    def hard_negative_sampling(self, topk: int = 10, top_range: int = 100):
        all_train_samples = []

        for ind in tqdm(range(self.f1_set.n_turns)):
            anchor_state = self.f1_set.turn_states[ind]

            pool = self.f1_set.pool[ind][self.score_type]
            pool = sorted(pool.items(), key=lambda x: x[1], reverse=True)
            pool_idx = [self.turn_label_to_idx[x[0]] for x in pool]

            topk_score = pool[topk][1]
            pos_examples = [x for x in pool if x[1] > topk_score]
            chosen_positive_idx = [self.turn_label_to_idx[x[0]] for x in pos_examples]

            # if we don't have `topk` positives yet, break ties at topk_score by
            # state-similarity to the anchor
            n_select = topk - len(pos_examples)
            if n_select > 0:
                pos_candidate = [x for x in pool if x[1] == topk_score]
                pos_candidate_idx = [self.turn_label_to_idx[x[0]] for x in pos_candidate]
                pos_candidate_state = [self.f1_set.turn_states[i] for i in pos_candidate_idx]
                pos_sim = sorted(
                    enumerate(pos_candidate_state),
                    key=lambda x: compute_sv_sim(x[1], anchor_state),
                    reverse=True,
                )
                chosen_positive_idx += [pos_candidate_idx[i] for i, _ in pos_sim[:n_select]]

            neg_candidate_idx = [i for i in pool_idx if i not in chosen_positive_idx]
            neg_candidate_state = [self.f1_set.turn_states[i] for i in neg_candidate_idx]
            neg_sim_sorted = sorted(
                enumerate(neg_candidate_state),
                key=lambda x: compute_sv_sim(x[1], anchor_state),
                reverse=True,
            )
            hard_neg_idx = [neg_candidate_idx[i] for i, _ in neg_sim_sorted][-self.hard_neg_size:]
            hard_neg_utts = [self.f1_set.turn_utts[i] for i in hard_neg_idx]

            for pos_idx in chosen_positive_idx:
                row = [self.f1_set.turn_utts[ind], self.f1_set.turn_utts[pos_idx], *hard_neg_utts]
                visited = {ind, pos_idx, *hard_neg_idx}
                for _ in range(self.easy_neg_size):
                    idx = random.randint(0, self.f1_set.n_turns - 1)
                    while idx in visited:
                        idx = random.randint(0, self.f1_set.n_turns - 1)
                    visited.add(idx)
                    row.append(self.f1_set.turn_utts[idx])
                all_train_samples.append(row)

        return all_train_samples

    def generate_eval_examples(self, topk: int = 5, top_range: int = 100):
        return self.hard_negative_sampling(topk=topk, top_range=top_range)

    def generate_train_examples(self, topk: int = 5, top_range: int = 100):
        rows = self.generate_eval_examples(topk=topk, top_range=top_range)
        return [InputExample(texts=row) for row in rows]


class MWContrastiveDataloader:
    """Pairwise contrastive dataloader with hard-negative sampling, for RefPyDST retriever training.

    Copied from RefPyDST's ``retriever/code/retriever_finetuning.py`` into combisearch.
    For each anchor it asks the pretrained retriever for the ``top_range`` nearest turns,
    then takes the ``topk`` highest- and lowest-F1 of those as positives (label 1) and
    negatives (label 0).
    """

    def __init__(self, f1_set: MWDataset, pretrained_retriever: IndexRetriever):
        self.f1_set = f1_set
        self.pretrained_retriever = pretrained_retriever

    def hard_negative_sampling(self, topk=10, top_range=100):
        sentences1 = []
        sentences2 = []
        scores = []

        for ind in tqdm(range(self.f1_set.n_turns)):
            # nearest neighbours from the pretrained retriever (exclude the anchor itself)
            this_label = self.f1_set.turn_labels[ind]
            nearest_labels = self.pretrained_retriever.label_to_nearest_labels(
                this_label, k=top_range + 1)[:-1]
            nearest_args = [self.f1_set.turn_labels.index(l) for l in nearest_labels]

            similarities = self.f1_set.similarity_matrix[ind][nearest_args]
            sorted_args = similarities.argsort()
            chosen_positive_args = np.array(nearest_args)[list(sorted_args[-topk:])]
            chosen_negative_args = np.array(nearest_args)[list(sorted_args[:topk])]

            for chosen_arg in chosen_positive_args:
                sentences1.append(self.f1_set.turn_utts[ind])
                sentences2.append(self.f1_set.turn_utts[chosen_arg])
                scores.append(1)
            for chosen_arg in chosen_negative_args:
                sentences1.append(self.f1_set.turn_utts[ind])
                sentences2.append(self.f1_set.turn_utts[chosen_arg])
                scores.append(0)

        return sentences1, sentences2, scores

    def generate_eval_examples(self, topk=5, top_range=100):
        sentences1, sentences2, scores = self.hard_negative_sampling(topk=topk, top_range=top_range)
        scores = [float(s) for s in scores]
        return sentences1, sentences2, scores

    def generate_train_examples(self, topk=5, top_range=100):
        sentences1, sentences2, scores = self.generate_eval_examples(topk=topk, top_range=top_range)
        return [InputExample(texts=[sentences1[i], sentences2[i]], label=scores[i])
                for i in range(len(sentences1))]
