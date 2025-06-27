import modal
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import timm
from torchvision import transforms
from PIL import Image
from einops import rearrange
import os

# Build Modal Image including local Python source code
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "tqdm", "timm==0.9.12", "einops==0.7.0", "pillow")
    .add_local_file("buddy_pool.py", "/root/buddy_pool.py")
    .add_local_file("maxsim_loss.py", "/root/maxsim_loss.py")
    .add_local_file("modal_app_cub_trainable.py", "/root/modal_app_cub_trainable.py")
)

# Define Modal App with dataset volume
app = modal.App(
    "cub_triplet_multi_gpu_app",
    image=image,
    volumes={"/mnt/data": modal.Volume.from_name("cub-data")}
)

class TrainableMultiVectorEncoder(nn.Module):
    """TRAINABLE Multi-vector encoder with gradient checkpointing support."""
    
    def __init__(self, use_gradient_checkpointing=True):
        super().__init__()
        MODEL_NAME = "vit_base_patch14_reg4_dinov2.lvd142m"
        
        # Configuration matching original
        self.embed_dim = 768
        self.num_registers = 4
        self.img_size = 518
        self.roi_side = 3
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Create the model - but make it trainable by not using @torch.no_grad()
        self.backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
        
        # Note: We'll handle gradient checkpointing manually in forward() to avoid eval issues
        
        # Add a small projection layer to make it clearly trainable
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)
        
    def _buddy_pool(self, cue, patches2d):
        """Original buddy pooling implementation."""
        B, H, W, d = patches2d.shape
        flat = rearrange(patches2d, "b h w d -> b (h w) d")
        sim = torch.matmul(cue.unsqueeze(1), flat.transpose(1, 2)).squeeze(1)
        idx = sim.argmax(dim=-1)
        h = idx // W
        w = idx % W
        r = self.roi_side // 2
        roi = []
        for b in range(B):
            hs = slice(max(0, h[b]-r), min(H, h[b]+r+1))
            ws = slice(max(0, w[b]-r), min(W, w[b]+r+1))
            roi.append(patches2d[b, hs, ws, :].mean(dim=(0, 1)))
        return torch.stack(roi)
        
    def forward(self, x):
        """Forward pass WITH gradients - this is trainable!"""
        # Forward through backbone (with gradients)
        # Only use checkpointing during training AND when gradients are enabled AND input requires grad
        # Also check if any parameter requires grad to be extra safe
        any_param_requires_grad = any(p.requires_grad for p in self.backbone.parameters())
        
        use_checkpointing = (
            self.use_gradient_checkpointing and 
            self.training and 
            x.requires_grad and 
            torch.is_grad_enabled() and
            any_param_requires_grad
        )
        
        if use_checkpointing:
            # Use checkpoint for memory efficiency during training only
            from torch.utils.checkpoint import checkpoint
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                tokens = checkpoint(
                    self.backbone.forward_features, x, use_reentrant=False
                )
        else:
            # Normal forward pass (used during evaluation)
            tokens = self.backbone.forward_features(x)
        
        # Apply projection (trainable layer)
        tokens = self.projection(tokens)
        
        # Extract tokens following original structure
        cls_tok = tokens[:, 0:1, :]  # CLS token: (B, 1, D)
        regs_tok = tokens[:, 1:1 + self.num_registers, :]  # Register tokens: (B, 4, D)
        patch_tok = tokens[:, 1 + self.num_registers:, :]  # Patch tokens: (B, N, D)
        
        # Reshape patch tokens to spatial grid
        g = int(self.img_size // 14)  # ViT-B/14 grid size
        patches2d = rearrange(patch_tok, "b (h w) d -> b h w d", h=g, w=g)
        
        # Combine CLS and register tokens to form cues
        cues = torch.cat([cls_tok, regs_tok], dim=1)  # (B, 5, D)
        
        # Apply buddy pooling to get ROIs
        rois = torch.stack([self._buddy_pool(cues[:, i], patches2d)
                           for i in range(cues.size(1))], dim=1)
        
        # Combine cues and ROIs
        toks = torch.cat([cues, rois], dim=1)  # (B, 10, D)
        
        # Normalize
        return F.normalize(toks, dim=-1)


def colbert_score(X, Y):
    """ColBERT scoring function from the original implementation."""
    return torch.einsum("bnd,bmd->bnm", X, Y).max(dim=-1).values.sum(dim=-1)


class TripletColbertLoss(nn.Module):
    """Triplet loss using ColBERT scoring from the original implementation."""
    
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        pos_score = colbert_score(anchor, positive)
        neg_score = colbert_score(anchor, negative)
        loss = F.relu(neg_score - pos_score + self.margin)
        return loss.mean()


def _load_image(path):
    """Load a PIL image and preprocess it."""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(518),  # Training augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(path).convert("RGB")
    return preprocess(img)


@app.function(
    gpu="A100-80GB:2",  # 2 A100-80GB GPUs
    timeout=3600  # 1 hour timeout
)
def main(
    cub_root: str = "/mnt/data/CUB_200_2011",
    steps: int = 500,
    batch_size: int = 16,  # Increased batch size thanks to multi-GPU
    report_interval: int = 50,
    eval_batch_size: int = 16,  # Increased eval batch size
    lr: float = 2e-5,  # Slightly higher LR for larger effective batch
    use_mixed_precision: bool = True,
    use_gradient_checkpointing: bool = True
):
    """
    Train CUB triplet model on Modal with Multi-GPU + Mixed Precision.
    """
    # Setup multi-GPU environment
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    cub_root = Path(cub_root)

    def parse_cub(root: Path):
        cls_map = {}
        for line in (root / "classes.txt").read_text().splitlines():
            cid, cname = line.split()
            cls_map[int(cid)] = cname

        img_to_cid = {}
        for line in (root / "image_class_labels.txt").read_text().splitlines():
            iid, cid = line.split()
            img_to_cid[int(iid)] = int(cid)

        img_map = {}
        for line in (root / "images.txt").read_text().splitlines():
            iid, rel = line.split()
            img_map[int(iid)] = "/mnt/data/CUB_200_2011/images/" + rel

        train_ids = set()
        for line in (root / "train_test_split.txt").read_text().splitlines():
            iid, flag = line.split()
            if int(flag):
                train_ids.add(int(iid))

        train_paths = {c: [] for c in cls_map.values()}
        test_paths = {c: [] for c in cls_map.values()}
        for iid, path in img_map.items():
            cname = cls_map[img_to_cid[iid]]
            (train_paths[cname] if iid in train_ids else test_paths[cname]).append(path)

        train_paths = {c: ps for c, ps in train_paths.items() if len(ps) >= 2}
        test_paths = {c: ps for c, ps in test_paths.items() if c in train_paths and len(ps) >= 1}
        return train_paths, test_paths

    def load_cub_triplet(class_to_paths: dict[str, list[Path]]):
        """CORRECT triplet construction using independent sampling."""
        cls_pos = random.choice(list(class_to_paths.keys()))
        # Use random.choice twice to allow anchor==positive (matching original)
        a = random.choice(class_to_paths[cls_pos])
        p = random.choice(class_to_paths[cls_pos])
        neg_cls = random.choice([c for c in class_to_paths if c != cls_pos])
        n = random.choice(class_to_paths[neg_cls])
        return a, p, n

    def get_batch(class_to_paths: dict[str, list[Path]], batch_size: int):
        anchors, positives, negatives = [], [], []
        for _ in range(batch_size):
            a, p, n = load_cub_triplet(class_to_paths)
            for lst, img_path in [(anchors, a), (positives, p), (negatives, n)]:
                img = _load_image(img_path)
                lst.append(img)
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

    def evaluate_retrieval_recalls(train_paths, test_paths, model, device, ks, eval_batch_size, scaler=None):
        model.eval()
        
        # Ensure gradient checkpointing is disabled during evaluation
        if hasattr(model, 'module'):  # DataParallel case
            model.module.use_gradient_checkpointing = False
        else:
            model.use_gradient_checkpointing = False
        
        classes = sorted(train_paths.keys())
        cls2idx = {c: i for i, c in enumerate(classes)}
        gallery_embs, gallery_labels = [], []
        
        # Build gallery from training set
        with torch.no_grad():
            for c in classes:
                for p in train_paths[c][:5]:  # Limit to 5 per class for speed
                    img = _load_image(p).unsqueeze(0).to(device)
                    
                    if use_mixed_precision and scaler is not None:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=FutureWarning)
                            with torch.cuda.amp.autocast():
                                emb = model(img)
                    else:
                        emb = model(img)
                    
                    gallery_embs.append(emb[0, 0, :].cpu())  # Use CLS token
                    gallery_labels.append(cls2idx[c])
        
        gallery = torch.stack(gallery_embs, dim=0)
        gallery_norm = F.normalize(gallery, dim=1)

        test_items = [(cls2idx[c], p) for c, paths in test_paths.items() for p in paths[:10]]  # Limit for speed
        total = len(test_items)
        hits = {k: 0 for k in ks}

        for i in trange(0, total, eval_batch_size, desc="eval", unit="batch"):
            batch = test_items[i:i+eval_batch_size]
            labels = torch.tensor([lbl for lbl, _ in batch])
            imgs = []
            for _, p in batch:
                img = _load_image(p)
                imgs.append(img)
            imgs = torch.stack(imgs).to(device)

            with torch.no_grad():
                if use_mixed_precision and scaler is not None:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        with torch.cuda.amp.autocast():
                            embs = model(imgs)
                else:
                    embs = model(imgs)
                
                embs_cls = embs[:, 0, :].cpu()  # Use CLS token
                embs_norm = F.normalize(embs_cls, dim=1)

            sims = embs_norm @ gallery_norm.t()
            topk = sims.topk(max(ks), dim=1).indices.cpu().tolist()

            for k in ks:
                for qi, row in enumerate(topk):
                    if any(gallery_labels[idx] == labels[qi].item() for idx in row[:k]):
                        hits[k] += 1

        for k in ks:
            print(f"Recall@{k}: {hits[k] / total:.4f} ({hits[k]}/{total})")

        # Restore gradient checkpointing and training mode
        if hasattr(model, 'module'):  # DataParallel case
            model.module.use_gradient_checkpointing = use_gradient_checkpointing
        else:
            model.use_gradient_checkpointing = use_gradient_checkpointing
        
        model.train()

    # Initialize everything
    train_paths, test_paths = parse_cub(cub_root)
    
    # Setup multi-GPU
    device = torch.device("cuda:0")  # Primary device
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
    print(f"Primary device: {device}")
    
    # Create TRAINABLE model with gradient checkpointing
    model = TrainableMultiVectorEncoder(use_gradient_checkpointing=use_gradient_checkpointing)
    
    # Move to primary GPU first
    model = model.to(device)
    
    # Wrap with DataParallel for multi-GPU
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"✅ Model wrapped with DataParallel across {num_gpus} GPUs")
        effective_batch_size = batch_size * num_gpus
    else:
        effective_batch_size = batch_size
    
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    print(f"Effective batch size: {effective_batch_size} (per_gpu: {batch_size}, gpus: {num_gpus})")
    
    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = TripletColbertLoss(margin=0.2)
    
    # Setup mixed precision scaler
    if use_mixed_precision:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    hist = []
    best_recall = 0.0

    print("🚀 Starting Multi-GPU + Mixed Precision TRAINABLE triplet training...")
    print(f"📊 Mixed Precision: {'✅ Enabled' if use_mixed_precision else '❌ Disabled'}")
    print(f"📊 Gradient Checkpointing: {'✅ Enabled' if use_gradient_checkpointing else '❌ Disabled'}")
    
    for i in trange(steps, desc="train", unit="step"):
        # Get batch
        a, p, n = get_batch(train_paths, batch_size)
        a, p, n = a.to(device), p.to(device), n.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        if use_mixed_precision:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                with torch.cuda.amp.autocast():
                    emb_a = model(a)
                    emb_p = model(p)
                    emb_n = model(n)
                    loss = criterion(emb_a, emb_p, emb_n)
            
            # Backward pass with mixed precision
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)
            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()
        
        hist.append(loss.item())

        if (i + 1) % report_interval == 0:
            avg_loss = sum(hist[-report_interval:]) / report_interval
            print(f"[step {i+1:4d}] avg loss: {avg_loss:.4f} (effective_bs: {effective_batch_size})")
            
            # Quick evaluation every few steps  
            if (i + 1) % (report_interval * 2) == 0:
                #print("📊 Quick evaluation:")
                """
                evaluate_retrieval_recalls(
                    train_paths, test_paths, model, device,
                    ks=[1, 4], eval_batch_size=eval_batch_size, scaler=scaler
                )
                """
                pass

    # Final comprehensive evaluation
    print("--------------------------------")
    print("🎯 Final evaluation:")
    print(f"Final loss: {hist[-1]:.8f}")
    evaluate_retrieval_recalls(
        train_paths, test_paths, model, device,
        ks=[1, 2, 4], eval_batch_size=eval_batch_size, scaler=scaler
    )
    
    print("✅ Multi-GPU + Mixed Precision Training complete!")
    return {"final_loss": hist[-1] if hist else 0.0, "avg_final_loss": sum(hist[-10:]) / 10 if len(hist) >= 10 else 0.0}


if __name__ == "__main__":
    # For local testing
    with app.run():
        result = main.remote()
        print("Result:", result) 
