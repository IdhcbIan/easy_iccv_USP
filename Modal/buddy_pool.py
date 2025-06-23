"""Create one Buddy (ROI) token per cue by 3×3 mean pooling
around each cue's nearest patch token."""
from __future__ import annotations

import torch
from torch import nn
import math


class BuddyPool(nn.Module):
    """Mean-pool 3×3 neighborhood of the nearest patch for each cue."""

    def forward(self, cue: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """
        cue:     (B, 5, D)
        patches: (B, N, D) - can be either flat (B, N, D) or spatial (B, H, W, D)
        returns: (B, 5, D) – one Buddy per cue
        """
        b, k, d = cue.shape
        
        # Check if patches are already in spatial format or flat
        if len(patches.shape) == 4:
            # Already in spatial format (B, H, W, D)
            _, h, w, _ = patches.shape
        else:
            # Flat format (B, N, D) - reshape to spatial
            _, n, _ = patches.shape
            # Try to find factors that are close to square
            h = int(math.sqrt(n))
            w = n // h
            if h * w != n:
                # If we can't get exact factors, use a simple approach
                # Just create a dummy spatial dimension and use the flat approach
                h, w = 1, n
        
        cue_n = torch.nn.functional.normalize(cue, dim=-1)
        
        # Handle different input formats
        if len(patches.shape) == 4:
            # Original spatial format
            patches_n = torch.nn.functional.normalize(patches, dim=-1)
            flat = patches_n.view(b, h*w, d)  # (B, HW, D)
        else:
            # Already flat
            flat = torch.nn.functional.normalize(patches, dim=-1)  # (B, N, D)
        
        sims = torch.einsum("bkd,bnd->bkn", cue_n, flat)   # (B, 5, HW)
        
        # Find nearest neighbors
        idx = sims.argmax(dim=-1)          # nearest patch index → (B, 5)
        
        # For flat patches, we'll use the top-K nearest neighbors instead of spatial neighbors
        if len(patches.shape) != 4:
            # Get top 9 neighbors for each cue
            _, top_indices = torch.topk(sims, k=9, dim=-1)  # (B, 5, 9)
            
            # Gather the features for these neighbors
            # Reshape for gather: (B, 5, 9, D)
            roi = torch.zeros(b, k, 9, d, device=cue.device)
            
            for i in range(b):
                for j in range(k):
                    roi[i, j] = flat[i, top_indices[i, j]]
            
            # Average the 9 neighbors
            roi = roi.mean(dim=2)  # (B, 5, D)
            return roi
        
        # Original spatial approach
        y, x = idx // w, idx % w
        
        pad = torch.nn.functional.pad(patches, (0,0,1,1,1,1))  # (B,H+2,W+2,D)
        y, x = y+1, x+1                                        # offset for padding
        
        roi = torch.zeros_like(cue)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                roi += pad[:, y+dy, x+dx, :]
        roi /= 9.0
        return roi
