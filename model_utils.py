"""Load DINO-v2 base-register model and extract cue + patch tokens.

Cue tokens  = [CLS] + 4 register tokens  -> shape (B, 5, D)
Patch tokens = remaining tokens arranged into an (H, W) grid.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import timm
import torch
from torchvision import transforms
from PIL import Image

# -----------------------------------------------------------------------------
# Model & preprocessing
# -----------------------------------------------------------------------------
MODEL_NAME = "dinov2_vitb14_reg"      # ViT-B/14 with register tokens
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create once and reuse
_model = timm.create_model(MODEL_NAME, pretrained=True)
_model.train().to(_device)            # <-- train mode so gradients flow
_embed_dim = _model.num_features      # 768 for ViT-B/14

# Default DINO-v2 preprocessing
_preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
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
        patch_tokens – (B, H, W, D)
    """
    x = _model.forward_features(img.to(_device))   # (B, 1+4+N, D)

    cue_tokens   = x[:, :5, :]     # CLS + 4 registers
    patch_tokens = x[:, 5:, :]     # Flattened patches

    # reshape patches back to H×W grid
    b, n, d = patch_tokens.shape
    side = int(math.sqrt(n))       # 16 for 224/14
    patch_tokens = patch_tokens.view(b, side, side, d)

    return cue_tokens, patch_tokens


if __name__ == "__main__":
    import sys, pprint
    img = _load_image(sys.argv[1]) if len(sys.argv) > 1 else torch.randn(1,3,224,224)
    cues, patches = forward_tokens(img)
    pprint.pprint({"cue": cues.shape, "patch": patches.shape})
