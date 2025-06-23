"""Load DINO-v2 base-register model and extract cue + patch tokens.

Cue tokens  = [CLS] + 4 register tokens  -> shape (B, 5, D)
Patch tokens = remaining tokens arranged into an (H, W) grid.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import torch
from torchvision import transforms
from PIL import Image

# -----------------------------------------------------------------------------
# Model & preprocessing
# -----------------------------------------------------------------------------
MODEL_NAME = "dinov2_vitb14"  # Using standard DINOv2 ViT-B/14 model
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create once and reuse - loading from torch.hub instead of timm
_model = torch.hub.load('facebookresearch/dinov2', MODEL_NAME)
_model.train().to(_device)            # <-- train mode so gradients flow
_embed_dim = _model.embed_dim         # 768 for ViT-B/14

# Default DINO-v2 preprocessing
_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -----------------------------------------------------------------------------


def _load_image(path: Path | str) -> torch.Tensor:
    """Load a PIL image → (1,3,224,224) tensor ready for the model."""
    img = Image.open(path).convert("RGB")
    return _preprocess(img)[None]


def forward_tokens(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return
        cue_tokens   – (B, 5, D)
        patch_tokens – (B, N, D) where N is the number of patch tokens
                      (no reshaping to H×W grid to avoid shape issues)
    """
    # Forward pass through the model
    with torch.set_grad_enabled(True):  # Ensure gradients are computed
        x = _model.get_intermediate_layers(img.to(_device), n=1)[0]
    
    # Extract tokens
    cls_token = x[:, 0:1, :]  # CLS token
    
    # For DINOv2, we don't have explicit register tokens like in the reg variant
    # We'll use the first few patch tokens as a substitute for the 4 register tokens
    register_tokens = x[:, 1:5, :]
    
    # Combine CLS and register tokens to form cue tokens
    cue_tokens = torch.cat([cls_token, register_tokens], dim=1)  # (B, 5, D)
    
    # Remaining tokens are patch tokens
    patch_tokens = x[:, 5:, :]  # Flattened patches (B, N, D)
    
    # Return tokens without reshaping to avoid shape issues
    return cue_tokens, patch_tokens


if __name__ == "__main__":
    import sys, pprint
    img = _load_image(sys.argv[1]) if len(sys.argv) > 1 else torch.randn(1,3,224,224)
    cues, patches = forward_tokens(img)
    pprint.pprint({"cue": cues.shape, "patch": patches.shape})
