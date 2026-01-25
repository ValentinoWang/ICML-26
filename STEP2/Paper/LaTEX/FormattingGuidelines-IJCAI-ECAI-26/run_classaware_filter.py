#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class-aware style filter (Swin-T + PCA + prototype/shrinkage Mahalanobis).

- Swin Transformer Tiny 提取风格特征（channel mean+std，约 1536 维）。
- PCA：仅用源域特征拟合，再将目标投影进去。
- Few-shot 分支：目标某类样本 < fewshot_thresh 时，用原型欧氏距离，保留 top keep_ratio_few。
- 非 few-shot：收缩协方差 (shrinkage) 的马氏距离，保留 top keep_ratio。
- 仅对类对齐的源样本筛选（按类分组处理），避免跨类负迁移。
- 输出：保留索引 keep_indices.bin，过滤后的 train images/labels（硬链接），以及简易 data.yaml（val 需后续替换）。
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from style_filter_core import SwinStyleExtractor


def load_yolo_labels(img_paths: List[Path], labels_dir: Path) -> List[int]:
    labels: List[int] = []
    for img in img_paths:
        lbl = labels_dir / f"{img.stem}.txt"
        if not lbl.exists():
            labels.append(-1)
            continue
        with open(lbl, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if not lines:
            labels.append(-1)
            continue
        try:
            cid = int(lines[0].split()[0])
        except Exception:
            cid = -1
        labels.append(cid)
    return labels


def extract_features(img_paths: List[Path], device: str = "cuda:0", batch_size: int = 32) -> torch.Tensor:
    if not img_paths:
        return torch.empty(0)
    extractor = SwinStyleExtractor().to(device)
    extractor.eval()
    tfm = extractor.transform
    feats: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(img_paths), batch_size):
            batch_imgs = []
            for p in img_paths[i : i + batch_size]:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_imgs.append(tfm(img))
                except Exception:
                    continue
            if not batch_imgs:
                continue
            batch_t = torch.stack(batch_imgs, dim=0).to(device)
            feat = extractor(batch_t)
            feats.append(feat.cpu())
    if not feats:
        return torch.empty(0)
    return torch.cat(feats, dim=0)


def fit_pca(X: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    if dim > Vt.shape[0]:
        dim = Vt.shape[0]
    comps = Vt[:dim]
    X_proj = np.dot(Xc, comps.T)
    return X_proj, comps, mu


def apply_pca(X: np.ndarray, comps: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return np.dot(X - mu, comps.T)


def shrink_cov(X: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    emp = np.cov(X, rowvar=False)
    if emp.ndim == 0:
        emp = np.array([[emp]])
    d = emp.shape[0]
    return (1 - alpha) * emp + alpha * np.eye(d)


def filter_class_euclidean(src_feats: np.ndarray, tgt_feats: np.ndarray, keep_ratio: float) -> np.ndarray:
    if tgt_feats.shape[0] == 0 or src_feats.shape[0] == 0:
        return np.ones(src_feats.shape[0], dtype=bool)
    proto = tgt_feats.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(src_feats - proto, axis=1)
    k = max(1, int(len(dists) * keep_ratio))
    keep_idx = np.argsort(dists)[:k]
    mask = np.zeros_like(dists, dtype=bool)
    mask[keep_idx] = True
    return mask


def filter_class_maha(src_feats: np.ndarray, tgt_feats: np.ndarray, keep_ratio: float, alpha: float = 0.1) -> np.ndarray:
    if tgt_feats.shape[0] == 0 or src_feats.shape[0] == 0:
        return np.ones(src_feats.shape[0], dtype=bool)
    mu = tgt_feats.mean(axis=0)
    cov = shrink_cov(tgt_feats, alpha=alpha)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
    diff = src_feats - mu
    dists = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
    k = max(1, int(len(dists) * keep_ratio))
    keep_idx = np.argsort(dists)[:k]
    mask = np.zeros_like(dists, dtype=bool)
    mask[keep_idx] = True
    return mask


def main():
    ap = argparse.ArgumentParser(description="Class-aware style filter (Swin + PCA + prototype/maha)")
    ap.add_argument("--src-img", required=True, help="source train images dir")
    ap.add_argument("--src-lbl", required=True, help="source train labels dir")
    ap.add_argument("--tgt-train", required=True, help="target train images dir")
    ap.add_argument("--tgt-train-lbl", required=True, help="target train labels dir")
    ap.add_argument("--tgt-val", required=True, help="target val images dir")
    ap.add_argument("--tgt-val-lbl", required=True, help="target val labels dir")
    ap.add_argument("--output", required=True, help="output root for filtered dataset")
    ap.add_argument("--pca-dim", type=int, default=32, help="PCA dim (few-shot branch)")
    ap.add_argument("--pca-dim-large", type=int, default=64, help="PCA dim (non few-shot, currently unused to comply with source-only fit)")
    ap.add_argument("--fewshot-thresh", type=int, default=20, help="per-class count threshold")
    ap.add_argument("--keep-ratio-few", type=float, default=0.7)
    ap.add_argument("--keep-ratio", type=float, default=0.6)
    ap.add_argument("--alpha", type=float, default=0.1, help="shrinkage alpha")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1088)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    src_imgs = sorted(Path(args.src_img).glob("*.jpg"))
    src_labels = load_yolo_labels(src_imgs, Path(args.src_lbl))

    tgt_train_imgs = sorted(Path(args.tgt_train).glob("*.jpg"))
    tgt_train_labels = load_yolo_labels(tgt_train_imgs, Path(args.tgt_train_lbl))
    tgt_val_imgs = sorted(Path(args.tgt_val).glob("*.jpg"))
    tgt_val_labels = load_yolo_labels(tgt_val_imgs, Path(args.tgt_val_lbl))

    tgt_imgs = tgt_train_imgs + tgt_val_imgs
    tgt_labels = tgt_train_labels + tgt_val_labels

    print(f"Source images: {len(src_imgs)}, Target images: {len(tgt_imgs)}")

    # Extract features
    src_feats = extract_features(src_imgs, device=args.device)
    tgt_feats = extract_features(tgt_imgs, device=args.device)
    if src_feats.numel() == 0 or tgt_feats.numel() == 0:
        raise RuntimeError("Feature extraction failed: empty features")

    src_np = src_feats.numpy()
    tgt_np = tgt_feats.numpy()

    # PCA fit on source only (as required)
    pca_dim = args.pca_dim
    src_pca, comps, mu = fit_pca(src_np, pca_dim)
    tgt_pca = apply_pca(tgt_np, comps, mu)

    src_labels_np = np.array(src_labels)
    tgt_labels_np = np.array(tgt_labels)

    keep_mask = np.zeros(len(src_imgs), dtype=bool)

    classes = sorted(set([c for c in src_labels if c >= 0] + [c for c in tgt_labels if c >= 0]))
    for c in classes:
        src_idx = np.where(src_labels_np == c)[0]
        tgt_idx = np.where(tgt_labels_np == c)[0]
        src_c = src_pca[src_idx]
        tgt_c = tgt_pca[tgt_idx]
        if tgt_c.shape[0] < args.fewshot_thresh:
            kmask = filter_class_euclidean(src_c, tgt_c, args.keep_ratio_few)
        else:
            kmask = filter_class_maha(src_c, tgt_c, args.keep_ratio, alpha=args.alpha)
        keep_mask[src_idx] = kmask
        print(f"Class {c}: tgt={len(tgt_idx)}, src={len(src_idx)}, keep={kmask.sum()} ({kmask.sum()/len(kmask)*100:.2f}%)")

    keep_indices = np.where(keep_mask)[0].tolist()

    out_root = Path(args.output)
    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            (out_root / split / sub).mkdir(parents=True, exist_ok=True)

    # Apply mask to source train split (input src_img assumed train). Val/test可按需扩展。
    for idx in keep_indices:
        img = src_imgs[idx]
        lbl = Path(args.src_lbl) / f"{img.stem}.txt"
        dst_img = out_root / "train" / "images" / img.name
        dst_lbl = out_root / "train" / "labels" / lbl.name
        if not dst_img.exists():
            os.link(img, dst_img)
        if lbl.exists() and not dst_lbl.exists():
            os.link(lbl, dst_lbl)

    with open(out_root / "keep_indices.bin", "wb") as f:
        pickle.dump(keep_indices, f)

    names = ['chamfer', 'multifaceted', 'bump', 'impurity', 'crack', 'grind']
    (out_root / "mixture_data.yaml").write_text(
        f"train: {out_root/'train/images'}\nval: {out_root/'train/images'}\n\n"
        f"nc: 6\nnames: {names}\n"
    )
    print(f"Saved keep_indices={len(keep_indices)} to {out_root}")


if __name__ == "__main__":
    main()
