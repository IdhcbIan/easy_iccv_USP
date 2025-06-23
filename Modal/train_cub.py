from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import trange, tqdm

from model_utils import forward_tokens, _load_image, _model
from buddy_pool import BuddyPool
from maxsim_loss import MaxSimLoss


def parse_cub(root: Path):
    """
    Ler metadados do CUB 200 2011 em root e retornar dicionarios de listas de imagens de treino e teste
    """
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
        img_map[int(iid)] = root / "images" / rel

    train_ids = set()
    for line in (root / "train_test_split.txt").read_text().splitlines():
        iid, flag = line.split()
        if int(flag):
            train_ids.add(int(iid))

    train_paths = {c: [] for c in cls_map.values()}
    test_paths  = {c: [] for c in cls_map.values()}
    for iid, path in img_map.items():
        cname = cls_map[img_to_cid[iid]]
        if iid in train_ids:
            train_paths[cname].append(path)
        else:
            test_paths[cname].append(path)

    # filtrar classes com pelo menos duas imagens em treino
    train_paths = {c: ps for c, ps in train_paths.items() if len(ps) >= 2}
    # filtrar em teste apenas classes que estao em treino e com ao menos uma imagem
    test_paths  = {c: ps for c, ps in test_paths.items()  if c in train_paths and len(ps) >= 1}
    return train_paths, test_paths


def load_cub_triplet(class_to_paths: dict[str, list[Path]]):
    """
    Escolher aleatoriamente um triplet anchor positivo negativo
    """
    cls_pos = random.choice(list(class_to_paths.keys()))
    # escolhe duas imagens diferentes para anchor e positivo
    a, p = random.sample(class_to_paths[cls_pos], 2)
    # para negativo, escolhe outra classe
    other = [c for c in class_to_paths if c != cls_pos]
    neg_cls = random.choice(other)
    n = random.choice(class_to_paths[neg_cls])
    return a, p, n


def get_batch(class_to_paths: dict[str, list[Path]], batch_size: int):
    """
    Construir um lote de triplets como tensores
    Retorna tensores anchors, positives, negatives com forma [batch_size, C, H, W]
    """
    anchors, positives, negatives = [], [], []
    for _ in range(batch_size):
        a, p, n = load_cub_triplet(class_to_paths)
        for lst, img in ((anchors, a), (positives, p), (negatives, n)):
            t = _load_image(img) if isinstance(img, (str, Path)) else img.float() / 255
            # se vier sem dim de batch, adicionar
            if t.ndim == 3:
                t = t.unsqueeze(0)
            # caso venha com dim extra de batch unica
            if t.ndim == 4 and t.shape[0] == 1:
                t = t.squeeze(0)
            lst.append(t)
    # garante empilhar no formato correto
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


def evaluate_retrieval_recalls(
    train_paths: dict[str, list[Path]],
    test_paths:  dict[str, list[Path]],
    buddy_pool:  BuddyPool,
    device:      torch.device,
    ks:          list[int],
    eval_batch_size: int
):
    """
    Calcular recall em varios k em lotes
    Embeds da galeria sao calculados uma vez no CPU
    Queries sao codificadas no dispositivo e retornam ao CPU para calculo de similaridade
    """
    buddy_pool.eval()
    _model.eval()

    # montar galeria de embeddings de token de classe no CPU
    classes = sorted(train_paths.keys())
    cls2idx = {c: i for i, c in enumerate(classes)}
    gallery_embs, gallery_labels = [], []
    with torch.no_grad():
        for c in classes:
            for p in train_paths[c]:
                img = _load_image(p)
                # garantir formato [1, C, H, W]
                if img.ndim == 3:
                    img = img.unsqueeze(0)
                img = img.to(device)
                c_tok, _ = forward_tokens(img)      # suposto [1, seq_len, D]
                # extrair token de classe no indice zero
                # c_tok tem forma [1, seq_len, D]
                emb = c_tok[0, 0, :].cpu()          # resulta em [D]
                gallery_embs.append(emb)
                gallery_labels.append(cls2idx[c])
    # empilhar no formato [N_gallery, D]
    gallery = torch.stack(gallery_embs, dim=0)
    gallery_norm = F.normalize(gallery, dim=1)   # [N_gallery, D]

    # preparar lista de queries de teste
    test_items = [(cls2idx[c], p) for c, paths in test_paths.items() for p in paths]
    total = len(test_items)
    hits = {k: 0 for k in ks}

    # avaliar em lotes
    for i in range(0, total, eval_batch_size):
        batch = test_items[i:i+eval_batch_size]
        labels = torch.tensor([lbl for lbl, _ in batch])  # CPU tensor de labels
        imgs = []
        for _, p in batch:
            img = _load_image(p)
            # garantir formato [1, C, H, W] antes de concatenar
            if img.ndim == 3:
                img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0).to(device)  # [B, C, H, W]

        with torch.no_grad():
            c_tok, _ = forward_tokens(imgs)         # [B, seq_len, D]
            # extrair token de classe para cada item
            embs = c_tok[:, 0, :].cpu()             # [B, D]
            embs_norm = F.normalize(embs, dim=1)    # [B, D] no CPU

        # calcular similaridade via produto interno
        sims = embs_norm @ gallery_norm.t()       # [B, N_gallery]
        topk = sims.topk(max(ks), dim=1).indices.cpu().tolist()

        # calcular hits
        for k in ks:
            for qi, row in enumerate(topk):
                # se em algum dos top k indices a label bater, contar como acerto
                if any(gallery_labels[idx] == labels[qi].item() for idx in row[:k]):
                    hits[k] += 1

    # exibir resultado
    for k in ks:
        print(f"Recall@{k}: {hits[k] / total:.4f} ({hits[k]}/{total})")

    buddy_pool.train()
    _model.train()


def main(
    cub_root: Path,
    steps: int,
    batch_size: int,
    report_interval: int,
    eval_batch_size: int
):
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

    print(f"Final loss: {hist[-1]:.4f}")
    print(f"Avg loss: {sum(hist) / len(hist):.4f}")

    evaluate_retrieval_recalls(
        train_paths, test_paths, buddy_pool, device,
        ks=[1, 2, 4], eval_batch_size=eval_batch_size
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub_root", type=Path, default=Path("CUB_200_2011"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--report_interval", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    args = parser.parse_args()
    main(
        args.cub_root, args.steps,
        args.batch_size, args.report_interval,
        args.eval_batch_size
    )

