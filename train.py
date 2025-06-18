"""Quick demo training loop.
If --img_dir has ≤2 JPEGs, the script falls back to coloured noise."""
from __future__ import annotations

import argparse, random
from pathlib import Path

import torch
from tqdm import trange

from model_utils import forward_tokens, _load_image, _model
from buddy_pool import BuddyPool
from maxsim_loss import MaxSimLoss


def rand_img():
    return (torch.rand(3, 224, 224) * 255).clamp(0,255).byte()

def load_triplet(paths):
    if len(paths) >= 3:
        return random.sample(paths, 3)
    return [rand_img() for _ in range(3)]


def main(img_dir: Path, steps: int):
    paths = list(img_dir.glob("*.jpg")) if img_dir.exists() else []

    buddy_pool = BuddyPool().to(_model.device if hasattr(_model,'device') else "cpu")
    criterion  = MaxSimLoss()

    # ‼️ optimise BOTH BuddyPool **and** ViT backbone
    optimiser  = torch.optim.Adam(
        list(buddy_pool.parameters()) + list(_model.parameters()),
        lr=1e-4
    )

    for _ in trange(steps, desc="train", unit="step"):
        a,p,n = load_triplet(paths)

        a = _load_image(a) if isinstance(a,(str,Path)) else a.float()/255
        p = _load_image(p) if isinstance(p,(str,Path)) else p.float()/255
        n = _load_image(n) if isinstance(n,(str,Path)) else n.float()/255

        a_c, a_p = forward_tokens(a)
        p_c, p_p = forward_tokens(p)
        n_c, n_p = forward_tokens(n)

        a_b = buddy_pool(a_c, a_p)
        p_b = buddy_pool(p_c, p_p)
        n_b = buddy_pool(n_c, n_p)

        a_tok = torch.cat([a_c, a_b], dim=1)
        p_tok = torch.cat([p_c, p_b], dim=1)
        n_tok = torch.cat([n_c, n_b], dim=1)

        loss = criterion(a_tok, p_tok, n_tok)
        optimiser.zero_grad(); loss.backward(); optimiser.step()

    print(f"Final dummy loss: {loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=Path, default=Path("samples"))
    parser.add_argument("--steps",   type=int,  default=20)
    args = parser.parse_args()
    main(args.img_dir, args.steps)
