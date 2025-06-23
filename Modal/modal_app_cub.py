import modal
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import trange

from model_utils import forward_tokens, _load_image, _model
from buddy_pool import BuddyPool
from maxsim_loss import MaxSimLoss

# Build Modal Image including local Python source code
image = (
  modal.Image.debian_slim()
    .pip_install("torch","torchvision","tqdm")
    .add_local_file("model_utils.py",   "/root/model_utils.py")
    .add_local_file("buddy_pool.py",    "/root/buddy_pool.py")
    .add_local_file("maxsim_loss.py",   "/root/maxsim_loss.py")
    .add_local_file("modal_app.py",     "/root/modal_app.py")
)

# Define Modal App with dataset volume
app = modal.App(
    "cub_triplet_app",
    image=image,
    volumes={"/mnt/data": modal.Volume.from_name("cub-data")}
)

@app.function(
    #gpu="T4"
    #gpu="any"
    #gpu="A100"
    gpu="A100-80GB",
    #gpu="A100-80GB:4"
    timeout=300
)
def main(
    cub_root: str = "/mnt/data/CUB_200_2011",
    steps: int = 200,
    batch_size: int = 5,
    report_interval: int = 50,
    eval_batch_size: int = 5
):
    """
    Train CUB triplet model on Modal and evaluate recall@k.
    """
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
        test_paths  = {c: [] for c in cls_map.values()}
        for iid, path in img_map.items():
            cname = cls_map[img_to_cid[iid]]
            (train_paths[cname] if iid in train_ids else test_paths[cname]).append(path)

        train_paths = {c: ps for c, ps in train_paths.items() if len(ps) >= 2}
        test_paths  = {c: ps for c, ps in test_paths.items() if c in train_paths and len(ps) >= 1}
        return train_paths, test_paths

    def load_cub_triplet(class_to_paths: dict[str, list[Path]]):
        cls_pos = random.choice(list(class_to_paths.keys()))
        a, p = random.sample(class_to_paths[cls_pos], 2)
        neg_cls = random.choice([c for c in class_to_paths if c != cls_pos])
        n = random.choice(class_to_paths[neg_cls])
        return a, p, n

    def get_batch(class_to_paths: dict[str, list[Path]], batch_size: int):
        anchors, positives, negatives = [], [], []
        for _ in range(batch_size):
            a, p, n = load_cub_triplet(class_to_paths)
            for lst, img in ((anchors, a), (positives, p), (negatives, n)):
                t = _load_image(img) if isinstance(img, (str, Path)) else img.float() / 255
                if t.ndim == 3:
                    t = t.unsqueeze(0)
                if t.ndim == 4 and t.shape[0] == 1:
                    t = t.squeeze(0)
                lst.append(t)
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

    def evaluate_retrieval_recalls(
        train_paths, test_paths, buddy_pool, device, ks, eval_batch_size
    ):
        buddy_pool.eval()
        _model.eval()

        print("aqui")

        classes = sorted(train_paths.keys())
        cls2idx = {c: i for i, c in enumerate(classes)}
        gallery_embs, gallery_labels = [], []
        with torch.no_grad():
            for c in classes:
                for p in train_paths[c]:
                    img = _load_image(p)
                    if img.ndim == 3:
                        img = img.unsqueeze(0)
                    img = img.to(device)
                    c_tok, _ = forward_tokens(img)
                    gallery_embs.append(c_tok[0, 0, :].cpu())
                    gallery_labels.append(cls2idx[c])
        gallery = torch.stack(gallery_embs, dim=0)
        gallery_norm = F.normalize(gallery, dim=1)

        test_items = [(cls2idx[c], p) for c, paths in test_paths.items() for p in paths]
        total = len(test_items)
        hits = {k: 0 for k in ks}

        print("sai")

        from tqdm import trange

        for i in trange(0, total, eval_batch_size, desc="eval", unit="batch"):
            batch = test_items[i:i+eval_batch_size]
            labels = torch.tensor([lbl for lbl, _ in batch])
            imgs = []
            for _, p in batch:
                img = _load_image(p)
                if img.ndim == 3:
                    img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0).to(device)

            with torch.no_grad():
                c_tok, _ = forward_tokens(imgs)
                embs = c_tok[:, 0, :].cpu()
                embs_norm = F.normalize(embs, dim=1)

            sims = embs_norm @ gallery_norm.t()
            topk = sims.topk(max(ks), dim=1).indices.cpu().tolist()

            for k in ks:
                for qi, row in enumerate(topk):
                    if any(gallery_labels[idx] == labels[qi].item() for idx in row[:k]):
                        hits[k] += 1

        for k in ks:
            print(f"Recall@{k}: {hits[k] / total:.4f} ({hits[k]}/{total})")

        buddy_pool.train()
        _model.train()

    # Running training
    train_paths, test_paths = parse_cub(cub_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    _model.to(device)
    buddy_pool = BuddyPool().to(device)

    optimiser = torch.optim.AdamW(
        list(buddy_pool.parameters()) + list(_model.parameters()),
        lr=1e-4, weight_decay=1e-5
    )
    criterion = MaxSimLoss()
    hist = []

    for i in trange(steps, desc="train", unit="step"):
        a, p, n = get_batch(train_paths, batch_size)
        a, p, n = a.to(device), p.to(device), n.to(device)

        c_a, p_a = forward_tokens(a)
        c_p, p_p = forward_tokens(p)
        c_n, p_n = forward_tokens(n)

        b_a = buddy_pool(c_a, p_a)
        b_p = buddy_pool(c_p, p_p)
        b_n = buddy_pool(c_n, p_n)

        t_a = torch.cat([c_a, b_a], dim=1)
        t_p = torch.cat([c_p, b_p], dim=1)
        t_n = torch.cat([c_n, b_n], dim=1)

        loss = criterion(t_a, t_p, t_n)
        hist.append(loss.item())
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if (i + 1) % report_interval == 0:
            avg = sum(hist[-report_interval:]) / report_interval
            print(f"[step {i+1:4d}] avg loss: {avg:.4f}")

    # <— add this
    evaluate_retrieval_recalls(
        train_paths, test_paths, buddy_pool, device,
        ks=[1, 2, 4], eval_batch_size=eval_batch_size
    )


