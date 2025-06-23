"""ColBERT-style max-sim hinge loss (margin = 0.1)."""
from __future__ import annotations

import torch
from torch import nn


class MaxSimLoss(nn.Module):
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    @staticmethod
    def _maxsim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = torch.nn.functional.normalize(a, dim=-1)
        b = torch.nn.functional.normalize(b, dim=-1)
        sim = torch.einsum("btd,bsd->bts", a, b)  # (B,T₁,T₂)
        return sim.max(dim=-1).values.mean(dim=-1)  # (B,)

    def forward(self, anchor, pos, neg):
        pos_score = self._maxsim(anchor, pos)
        neg_score = self._maxsim(anchor, neg)
        return torch.clamp(self.margin + neg_score - pos_score, min=0).mean()
