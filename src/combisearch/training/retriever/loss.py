"""InfoNCE contrastive loss with manually curated one positive and hard negatives per anchor.

A row from :class:`InfoNCEDataloader` looks like
``[anchor, positive, hard_neg_1, ..., hard_neg_n, easy_neg_1, ..., easy_neg_m]``.
The encoded embeddings produce one positive score and N+M negative scores;
the cross-entropy target is index 0 (the positive).
"""

import logging
from typing import Dict, Iterable

import torch

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss

logger = logging.getLogger(__name__)


class InfoNCELoss(MultipleNegativesRankingLoss):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct=util.pairwise_cos_sim,
    ) -> None:
        super().__init__(model=model, scale=scale, similarity_fct=similarity_fct)

    def forward(
        self,
        sentence_features: Iterable[Dict[str, torch.Tensor]],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        reps = [self.model(sf)["sentence_embedding"] for sf in sentence_features]
        anchor, pos, *negs = reps

        scores = self.similarity_fct(anchor, pos).unsqueeze(-1) * self.scale
        for neg in negs:
            scores = torch.cat(
                (scores, self.scale * self.similarity_fct(anchor, neg).unsqueeze(-1)),
                dim=-1,
            )

        target = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        return self.cross_entropy_loss(scores, target)
