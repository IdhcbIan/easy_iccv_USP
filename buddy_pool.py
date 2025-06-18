"""Create one Buddy (ROI) token per cue by 3×3 mean pooling
around each cue’s nearest patch token."""
from __future__ import annotations

import torch
from torch import nn


class BuddyPool(nn.Module):
    """Mean-pool 3×3 neighborhood of the nearest patch for each cue."""

    def forward(self, cue: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """
        cue:     (B, 5, D)
        patches: (B, H, W, D)
        returns: (B, 5, D) – one Buddy per cue
        """
        b, k, d  = cue.shape
        _, h, w, _ = patches.shape

        cue_n     = torch.nn.functional.normalize(cue,     dim=-1)
        patches_n = torch.nn.functional.normalize(patches, dim=-1)

        flat = patches_n.view(b, h*w, d)                   # (B, HW, D)
        sims = torch.einsum("bkd,bnd->bkn", cue_n, flat)   # (B, 5, HW)

        idx = sims.argmax(dim=-1)          # nearest patch index → (B, 5)
        y, x = idx // w, idx % w

        pad = torch.nn.functional.pad(patches, (0,0,1,1,1,1))  # (B,H+2,W+2,D)
        y, x = y+1, x+1                                        # offset for padding

        roi = torch.zeros_like(cue)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                roi += pad[:, y+dy, x+dx, :]
        roi /= 9.0
        return roi
