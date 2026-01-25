import argparse
import json
import pathlib
import csv
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch import amp

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from filter.losses import classification_loss, ess_loss
from filter.set_aware.model import SetAwareBiasRobustFilter
from Tools.deterministic import set_deterministic, make_dataloader_seed

CSV_HEADERS = (
    ["generation", "train_size", "acc", "worst_class_acc", "ess_score"]
    + [f"acc_c{i}" for i in range(10)]
    + [f"hist_c{i}" for i in range(10)]
    + ["sel_pseudo_acc", "sel_mean_conf"]
    + [f"sel_pseudo_acc_c{i}" for i in range(10)]
    + ["sel_mean_weight", "sel_mean_score"]
)
MERGED_HEADERS = ["method"] + list(CSV_HEADERS)

def stratified_indices_from_targets(
    targets: List[int] | np.ndarray,
    total: int,
    seed: int,
    num_classes: int,
) -> List[int]:
    """
    Deterministically select a (roughly) class-balanced subset of indices.

    This is used to build a small clean validation set that covers all classes,
    avoiding run-to-run variance due to class-imbalanced random subsampling.
    """
    targets_arr = np.asarray(targets, dtype=np.int64)
    total = int(min(total, targets_arr.size))
    if total <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    per_class = total // num_classes
    remainder = total - per_class * num_classes
    extra_classes = set(rng.choice(num_classes, size=remainder, replace=False).tolist()) if remainder > 0 else set()
    chosen: List[int] = []
    for c in range(num_classes):
        class_idx = np.flatnonzero(targets_arr == c)
        take = per_class + (1 if c in extra_classes else 0)
        if take <= 0:
            continue
        if class_idx.size == 0:
            continue
        rng.shuffle(class_idx)
        chosen.extend(class_idx[: min(take, class_idx.size)].tolist())
    rng.shuffle(chosen)
    return chosen[:total]

def stratified_sample_from_indices(
    indices: List[int] | np.ndarray,
    targets: List[int] | np.ndarray,
    total: int,
    seed: int,
    num_classes: int,
) -> List[int]:
    """
    Stratified sampling of dataset indices given a candidate index set.
    """
    indices_arr = np.asarray(indices, dtype=np.int64)
    if indices_arr.size == 0 or total <= 0:
        return []
    targets_arr = np.asarray(targets, dtype=np.int64)
    y = targets_arr[indices_arr]
    chosen_pos = stratified_indices_from_targets(y, total=total, seed=seed, num_classes=num_classes)
    return indices_arr[np.asarray(chosen_pos, dtype=np.int64)].tolist()


def balanced_topk_indices(
    scores: np.ndarray,
    labels: np.ndarray,
    top_k: int,
    num_classes: int,
    alpha: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Class-aware top-k selection.

    We interpolate between the candidate label distribution p(y) and a uniform prior u(y):
        q(y) = (1-α) p(y) + α u(y),  α∈[0,1]
    and allocate per-class quotas proportional to q(y), then take the best-scoring items
    within each class. Remaining slots are filled by global score.

    This directly targets confirmation-bias collapse (class skew) while staying deterministic.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = int(scores.shape[0])
    top_k = int(min(top_k, n))
    if top_k <= 0:
        return np.array([], dtype=np.int64)
    if valid_mask is None:
        valid_mask = np.ones(n, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    if alpha <= 0.0:
        order = np.argsort(-scores, kind="mergesort")
        return order[:top_k]
    alpha = float(np.clip(alpha, 0.0, 1.0))

    valid_labels = labels[valid_mask]
    if valid_labels.size == 0:
        order = np.argsort(-scores, kind="mergesort")
        return order[:top_k]
    counts = np.bincount(valid_labels, minlength=num_classes).astype(np.float64)
    total_valid = counts.sum()
    if total_valid <= 0:
        order = np.argsort(-scores, kind="mergesort")
        return order[:top_k]
    p = counts / total_valid
    u = np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
    q = (1.0 - alpha) * p + alpha * u
    q = q / q.sum()
    raw = q * top_k
    quotas = np.floor(raw).astype(np.int64)
    remainder = int(top_k - quotas.sum())
    if remainder > 0:
        frac = raw - quotas
        for c in np.argsort(-frac, kind="mergesort")[:remainder]:
            quotas[int(c)] += 1

    selected: List[int] = []
    selected_mask = np.zeros(n, dtype=bool)
    for c in range(num_classes):
        if quotas[c] <= 0:
            continue
        class_mask = valid_mask & (labels == c)
        if not class_mask.any():
            continue
        class_idx = np.flatnonzero(class_mask)
        class_scores = scores[class_idx]
        class_order = class_idx[np.argsort(-class_scores, kind="mergesort")]
        take = int(min(quotas[c], class_order.size))
        if take <= 0:
            continue
        picked = class_order[:take].tolist()
        selected.extend(picked)
        selected_mask[picked] = True

    if len(selected) < top_k:
        order = np.argsort(-scores, kind="mergesort")
        for idx in order.tolist():
            if len(selected) >= top_k:
                break
            if not valid_mask[idx]:
                continue
            if selected_mask[idx]:
                continue
            selected.append(int(idx))
            selected_mask[idx] = True
    return np.asarray(selected[:top_k], dtype=np.int64)

def per_class_quantile_mask(
    scores: np.ndarray,
    labels: np.ndarray,
    quantile: float,
    num_classes: int,
) -> np.ndarray:
    """
    Keep top-q fraction per class by score, to avoid global thresholds that
    wipe out hard/rare classes.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    q = float(np.clip(quantile, 0.0, 1.0))
    if q >= 1.0:
        return np.ones(scores.shape[0], dtype=bool)
    if q <= 0.0:
        return np.zeros(scores.shape[0], dtype=bool)
    keep = np.zeros(scores.shape[0], dtype=bool)
    for c in range(num_classes):
        class_mask = labels == c
        if not class_mask.any():
            continue
        class_scores = scores[class_mask]
        if class_scores.size <= 1:
            keep[class_mask] = True
            continue
        threshold = np.quantile(class_scores, 1.0 - q)
        keep[class_mask] = class_scores >= threshold
    return keep

def k_center_greedy(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = points.shape[0]
    if k >= n:
        return np.arange(n, dtype=int)
    first = int(rng.integers(n))
    centers = [first]
    dist2 = np.sum((points - points[first]) ** 2, axis=1)
    for _ in range(1, k):
        idx = int(np.argmax(dist2))
        centers.append(idx)
        new_dist2 = np.sum((points - points[idx]) ** 2, axis=1)
        dist2 = np.minimum(dist2, new_dist2)
    return np.array(centers, dtype=int)


def rbf_kernel(points: np.ndarray, sigma: float | None, jitter: float) -> np.ndarray:
    pts = points.astype(np.float32)
    sq = np.sum(pts * pts, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2.0 * (pts @ pts.T)
    dist2 = np.maximum(dist2, 0.0)
    if sigma is None or sigma <= 0:
        nonzero = dist2[dist2 > 0]
        if nonzero.size == 0:
            sigma = 1.0
        else:
            sigma = float(np.sqrt(np.median(nonzero)))
            if sigma <= 0:
                sigma = 1.0
    kernel = np.exp(-dist2 / (2.0 * sigma**2))
    if jitter > 0:
        kernel = kernel + jitter * np.eye(kernel.shape[0])
    return kernel


def dpp_greedy(L: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = L.shape[0]
    if k >= n:
        return np.arange(n, dtype=int)
    cis = np.zeros((k, n), dtype=float)
    di2s = np.clip(np.diag(L).copy(), 0.0, None)
    di2s = di2s + 1e-12 * rng.random(n)
    selected: List[int] = []
    for it in range(k):
        idx = int(np.argmax(di2s))
        if di2s[idx] <= 1e-12:
            break
        selected.append(idx)
        if it == k - 1:
            break
        if it == 0:
            eis = L[idx, :] / np.sqrt(di2s[idx])
        else:
            proj = cis[:it, idx] @ cis[:it, :]
            eis = (L[idx, :] - proj) / np.sqrt(di2s[idx])
        cis[it, :] = eis
        di2s = di2s - eis**2
        di2s[idx] = -np.inf
    if len(selected) < k:
        remaining = np.argsort(di2s)[::-1]
        for idx in remaining:
            if idx not in selected:
                selected.append(int(idx))
            if len(selected) == k:
                break
    return np.array(selected, dtype=int)


def project_features(features: np.ndarray, proj_dim: int, rng: np.random.Generator) -> np.ndarray:
    if proj_dim <= 0 or proj_dim >= features.shape[1]:
        return features
    proj = rng.normal(size=(features.shape[1], proj_dim)).astype(np.float32) / np.sqrt(float(proj_dim))
    return features @ proj

def per_class_topk_mask(
    scores: np.ndarray,
    labels: np.ndarray,
    per_class_k: int,
    num_classes: int,
    base_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Keep up to top-k items per class by score, optionally constrained by a base mask.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if base_mask is None:
        base_mask = np.ones(scores.shape[0], dtype=bool)
    else:
        base_mask = np.asarray(base_mask, dtype=bool)
    k = int(per_class_k)
    if k <= 0 or scores.size == 0:
        return np.zeros(scores.shape[0], dtype=bool)
    keep = np.zeros(scores.shape[0], dtype=bool)
    for c in range(num_classes):
        class_mask = base_mask & (labels == c)
        if not class_mask.any():
            continue
        class_idx = np.flatnonzero(class_mask)
        class_scores = scores[class_idx]
        order = class_idx[np.argsort(-class_scores, kind="mergesort")]
        take = int(min(k, order.size))
        if take <= 0:
            continue
        keep[order[:take]] = True
    return keep

def compute_selection_stats(
    true_labels: np.ndarray,
    pseudo_labels: np.ndarray,
    confidences: np.ndarray,
    num_classes: int,
    weights: np.ndarray | None = None,
    scores: np.ndarray | None = None,
) -> Dict[str, float | str]:
    """
    Log-only diagnostics (uses ground-truth labels from CIFAR-10 train split).
    This is NOT used for training; it is written to CSV to show that improvements
    are not solely from the class-balance heuristic (α), but also from cleaner
    within-class selection (higher pseudo-label correctness).
    """
    true_labels = np.asarray(true_labels, dtype=np.int64)
    pseudo_labels = np.asarray(pseudo_labels, dtype=np.int64)
    confidences = np.asarray(confidences, dtype=np.float64)
    n = int(pseudo_labels.size)
    if n == 0:
        out: Dict[str, float | str] = {"sel_pseudo_acc": "", "sel_mean_conf": ""}
        for c in range(num_classes):
            out[f"sel_pseudo_acc_c{c}"] = ""
        out["sel_mean_weight"] = ""
        out["sel_mean_score"] = ""
        return out

    correct = (true_labels == pseudo_labels)
    out = {
        "sel_pseudo_acc": float(correct.mean()),
        "sel_mean_conf": float(confidences.mean()),
    }
    for c in range(num_classes):
        mask = pseudo_labels == c
        if not mask.any():
            out[f"sel_pseudo_acc_c{c}"] = ""
        else:
            out[f"sel_pseudo_acc_c{c}"] = float((true_labels[mask] == c).mean())

    if weights is None:
        out["sel_mean_weight"] = ""
    else:
        w = np.asarray(weights, dtype=np.float64)
        out["sel_mean_weight"] = float(w.mean()) if w.size > 0 else ""

    if scores is None:
        out["sel_mean_score"] = ""
    else:
        s = np.asarray(scores, dtype=np.float64)
        out["sel_mean_score"] = float(s.mean()) if s.size > 0 else ""

    return out


def prototype_agreement_scores(
    features: np.ndarray,
    pseudo_labels: np.ndarray,
    confidences: np.ndarray,
    num_classes: int,
    topk_per_class: int,
    temperature: float,
    conf_power: float,
) -> np.ndarray:
    """
    Prototype-based pseudo-label agreement score in [0,1].

    For each pseudo-class, build a prototype from the top-confidence examples,
    then score each sample by the softmax probability of its pseudo-label under
    cosine similarity to class prototypes.

    This provides an inexpensive, label-noise sensitive signal that goes beyond
    raw confidence, and helps set-aware weights learn within-class de-noising.
    """
    if temperature <= 0.0:
        raise ValueError("--proto-temp must be > 0")
    if topk_per_class <= 0:
        raise ValueError("--proto-topk must be > 0")
    if conf_power < 0.0:
        raise ValueError("--proto-conf-power must be >= 0")

    feats = np.asarray(features, dtype=np.float32)
    labels = np.asarray(pseudo_labels, dtype=np.int64)
    conf = np.asarray(confidences, dtype=np.float32)
    n, d = feats.shape

    # Normalize features for cosine similarity.
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    feats = feats / norms

    protos = np.zeros((num_classes, d), dtype=np.float32)
    for c in range(num_classes):
        idx = np.flatnonzero(labels == c)
        if idx.size == 0:
            continue
        conf_c = conf[idx]
        order = np.argsort(-conf_c, kind="mergesort")
        idx = idx[order[: min(int(topk_per_class), int(idx.size))]]
        w = (conf[idx].astype(np.float32) ** float(conf_power)).reshape(-1, 1)
        wsum = float(w.sum())
        if wsum <= 0.0:
            proto = feats[idx].mean(axis=0)
        else:
            proto = (feats[idx] * w).sum(axis=0) / wsum
        pnorm = float(np.linalg.norm(proto) + 1e-12)
        protos[c] = proto / pnorm

    sims = feats @ protos.T  # [N, C]
    logits = sims / float(temperature)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-12)
    return probs[np.arange(n), labels].astype(np.float32)


def apply_delta_phi_correction(
    weights: torch.Tensor,
    delta_phi: torch.Tensor,
    x: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Use Δϕ as a global correction direction to modulate per-sample weights.

    This keeps the "selection/filtering" experimental structure but makes the Δϕ head
    participate in training and inference via a differentiable logit adjustment:
        w' = sigmoid(logit(w) + scale * cos(x_i, Δϕ))

    Args:
        weights: [B, N] in (0,1)
        delta_phi: [B, D]
        x: [B, N, D]
        scale: scalar η; set 0.0 to disable.
    """
    if scale == 0.0:
        return weights
    weights_safe = weights.clamp(1e-6, 1.0 - 1e-6)
    logits = torch.logit(weights_safe)
    x_unit = F.normalize(x, dim=-1)
    phi_unit = F.normalize(delta_phi, dim=-1).unsqueeze(1)
    corr = (x_unit * phi_unit).sum(dim=-1)
    return torch.sigmoid(logits + scale * corr)


def set_seed(seed: int) -> None:
    set_deterministic(seed)


def build_transforms() -> Tuple[T.Compose, T.Compose]:
    normalize = T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
    train = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ]
    )
    eval_tf = T.Compose([T.ToTensor(), normalize])
    return train, eval_tf


class CIFARSubset(Dataset):
    def __init__(self, base: Dataset, indices: List[int], transform=None, return_index: bool = False):
        self.base = base
        self.indices = list(indices)
        self.transform = transform
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        img, target = self.base[real_idx]
        if self.transform is not None:
            img = self.transform(img)
        if self.return_index:
            return img, target, real_idx
        return img, target


class NoisyLabelSubset(Dataset):
    def __init__(
        self,
        base: Dataset,
        indices: List[int],
        transform=None,
        return_index: bool = False,
        noise_rate: float = 0.0,
        noise_seed: int = 0,
        num_classes: int = 10,
    ):
        self.base = base
        self.indices = list(indices)
        self.transform = transform
        self.return_index = return_index
        self.noise_rate = float(noise_rate)
        self.num_classes = int(num_classes)
        if not (0.0 <= self.noise_rate <= 1.0):
            raise ValueError("clean_val_noise_rate must be in [0, 1].")
        rng = np.random.default_rng(int(noise_seed))
        noisy_labels: List[int] = []
        flip_count = 0
        for real_idx in self.indices:
            _, target = self.base[real_idx]
            target = int(target)
            if self.noise_rate > 0.0 and rng.random() < self.noise_rate:
                # Symmetric label noise: pick any class except the original.
                noisy = int(rng.integers(self.num_classes - 1))
                if noisy >= target:
                    noisy += 1
                noisy_labels.append(noisy)
                flip_count += 1
            else:
                noisy_labels.append(target)
        self.noisy_labels = noisy_labels
        self.flip_count = flip_count

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        img, _ = self.base[real_idx]
        if self.transform is not None:
            img = self.transform(img)
        target = self.noisy_labels[idx]
        if self.return_index:
            return img, target, real_idx
        return img, target


class PseudoLabeledDataset(Dataset):
    """
    Dataset that uses label_map to override labels for selected indices.
    """

    def __init__(self, base: Dataset, indices: List[int], label_map: Dict[int, int], transform=None):
        self.base = base
        self.indices = list(indices)
        self.label_map = label_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        img, _ = self.base[real_idx]
        if self.transform is not None:
            img = self.transform(img)
        label = self.label_map[real_idx]
        return img, label


def forward_with_features(model: torchvision.models.ResNet, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Mirrors torchvision.resnet forward to expose pre-fc features.
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    feat = torch.flatten(x, 1)
    logits = model.fc(feat)
    return logits, feat


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    num_classes = 10
    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            logits, _ = forward_with_features(model, imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            for c in range(num_classes):
                mask = targets == c
                per_class_total[c] += mask.sum().item()
                per_class_correct[c] += (preds[mask] == targets[mask]).sum().item()
    per_class_acc = [
        (per_class_correct[c] / per_class_total[c]) if per_class_total[c] > 0 else 0.0 for c in range(num_classes)
    ]
    worst_class_acc = min(per_class_acc)
    return {
        "acc": correct / total if total > 0 else 0.0,
        "worst_class_acc": worst_class_acc,
        "per_class_acc": per_class_acc,
    }


def train_classifier(
    model: nn.Module,
    dataset: Dataset,
    epochs: int,
    device: torch.device,
    lr: float,
    weight_decay: float,
    batch_size: int,
    num_workers: int,
    desc: str,
    grad_accum_steps: int,
    use_amp: bool,
    seed: int,
) -> None:
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    dl_seed = make_dataloader_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # avoid BatchNorm failure on tiny final batches
        persistent_workers=num_workers > 0,
        worker_init_fn=dl_seed["worker_init_fn"],
        generator=dl_seed["generator"],
    )
    total_steps = epochs * len(loader)
    pbar = tqdm(total=total_steps, desc=desc, ncols=100, leave=False)
    scaler = amp.GradScaler("cuda", enabled=use_amp)
    accum = max(1, grad_accum_steps)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for step_idx, (imgs, targets) in enumerate(loader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            with amp.autocast("cuda", enabled=use_amp):
                logits, _ = forward_with_features(model, imgs)
                loss = criterion(logits, targets) / accum
            scaler.scale(loss).backward()
            if (step_idx + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            pbar.update(1)
        # Flush leftover grads if dataset size not divisible by accum
        if (step_idx + 1) % accum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    pbar.close()


def collect_predictions(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    dl_seed = make_dataloader_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=dl_seed["worker_init_fn"],
        generator=dl_seed["generator"],
    )
    model.eval()
    all_idx: List[int] = []
    all_true: List[int] = []
    all_pred: List[int] = []
    all_conf: List[float] = []
    all_margin: List[float] = []
    all_feat: List[np.ndarray] = []
    with torch.no_grad(), amp.autocast("cuda", enabled=device.type == "cuda"):
        for imgs, targets, idxs in loader:
            imgs = imgs.to(device)
            logits, feats = forward_with_features(model, imgs)
            probs = torch.softmax(logits, dim=1)
            top2 = torch.topk(probs, k=2, dim=1)
            conf = top2.values[:, 0]
            pred = top2.indices[:, 0]
            margin = top2.values[:, 0] - top2.values[:, 1]
            all_idx.extend(idxs.tolist())
            all_true.extend(targets.tolist())
            all_pred.extend(pred.cpu().tolist())
            all_conf.extend(conf.cpu().tolist())
            all_margin.extend(margin.cpu().tolist())
            all_feat.append(feats.cpu().numpy())
    return {
        "indices": np.array(all_idx, dtype=np.int64),
        "true_labels": np.array(all_true, dtype=np.int64),
        "pseudo_labels": np.array(all_pred, dtype=np.int64),
        "confidences": np.array(all_conf, dtype=np.float32),
        "margins": np.array(all_margin, dtype=np.float32),
        "features": np.concatenate(all_feat, axis=0),
    }


def select_baseline(
    preds: Dict[str, np.ndarray], top_k: int, threshold: float
) -> Tuple[List[int], List[int], Dict[str, float]]:
    conf = preds["confidences"]
    order = np.argsort(-conf)
    if threshold > 0.0:
        mask = conf[order] >= threshold
        order = order[mask]
        if len(order) < top_k:
            order = np.argsort(-conf)[:top_k]
    chosen_idx = order[:top_k]
    selected_indices = preds["indices"][chosen_idx].tolist()
    selected_labels = preds["pseudo_labels"][chosen_idx].tolist()
    hist = np.bincount(preds["pseudo_labels"][chosen_idx], minlength=10).astype(np.int64)
    stats = compute_selection_stats(
        true_labels=preds["true_labels"][chosen_idx],
        pseudo_labels=preds["pseudo_labels"][chosen_idx],
        confidences=preds["confidences"][chosen_idx],
        num_classes=10,
    )
    return selected_indices, selected_labels, {"pseudo_label_hist": hist.tolist(), **stats}


def _prep_diversity_candidates(
    preds: Dict[str, np.ndarray],
    candidate_pool: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    conf = preds["confidences"]
    order = np.argsort(-conf)
    cand_top = order[: min(len(order), int(candidate_pool))]
    return (
        cand_top,
        preds["indices"][cand_top],
        preds["true_labels"][cand_top],
        preds["pseudo_labels"][cand_top],
        preds["confidences"][cand_top],
    )


def select_kcenter(
    preds: Dict[str, np.ndarray],
    top_k: int,
    candidate_pool: int,
    proj_dim: int,
    seed: int,
) -> Tuple[List[int], List[int], Dict[str, float]]:
    rng = np.random.default_rng(int(seed))
    cand_top, cand_indices, true_labels, pseudo_labels, confidences = _prep_diversity_candidates(
        preds, candidate_pool=candidate_pool
    )
    feats = preds["features"][cand_top].astype(np.float32)
    feats = project_features(feats, proj_dim, rng)
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    feats = feats / norms
    chosen = k_center_greedy(feats, min(int(top_k), feats.shape[0]), rng)
    selected_indices = cand_indices[chosen].tolist()
    selected_labels = pseudo_labels[chosen].tolist()
    hist = np.bincount(pseudo_labels[chosen], minlength=10).astype(np.int64)
    stats = compute_selection_stats(
        true_labels=true_labels[chosen],
        pseudo_labels=pseudo_labels[chosen],
        confidences=confidences[chosen],
        num_classes=10,
    )
    return selected_indices, selected_labels, {"pseudo_label_hist": hist.tolist(), **stats}


def select_dpp(
    preds: Dict[str, np.ndarray],
    top_k: int,
    candidate_pool: int,
    proj_dim: int,
    sigma: float,
    jitter: float,
    seed: int,
) -> Tuple[List[int], List[int], Dict[str, float]]:
    rng = np.random.default_rng(int(seed))
    cand_top, cand_indices, true_labels, pseudo_labels, confidences = _prep_diversity_candidates(
        preds, candidate_pool=candidate_pool
    )
    feats = preds["features"][cand_top].astype(np.float32)
    feats = project_features(feats, proj_dim, rng)
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    feats = feats / norms
    L = rbf_kernel(feats, sigma=sigma, jitter=jitter)
    chosen = dpp_greedy(L, min(int(top_k), L.shape[0]), rng)
    selected_indices = cand_indices[chosen].tolist()
    selected_labels = pseudo_labels[chosen].tolist()
    hist = np.bincount(pseudo_labels[chosen], minlength=10).astype(np.int64)
    stats = compute_selection_stats(
        true_labels=true_labels[chosen],
        pseudo_labels=pseudo_labels[chosen],
        confidences=confidences[chosen],
        num_classes=10,
    )
    return selected_indices, selected_labels, {"pseudo_label_hist": hist.tolist(), **stats}
def select_baseline_balanced(
    preds: Dict[str, np.ndarray],
    top_k: int,
    threshold: float,
    candidate_pool: int,
    alpha: float,
) -> Tuple[List[int], List[int], Dict[str, float]]:
    """
    Stronger baseline: confidence ranking + class-balance quotas (same α as set-aware selection).

    This isolates the effect of the class-balance constraint from the set-aware weighting itself.
    """
    conf = preds["confidences"]
    order = np.argsort(-conf)
    cand_top = order[: min(len(order), int(candidate_pool))]
    cand_conf = conf[cand_top]
    cand_labels = preds["pseudo_labels"][cand_top]

    keep_mask = cand_conf >= threshold
    if keep_mask.sum() == 0:
        keep_mask = np.ones_like(keep_mask, dtype=bool)
    scores_filtered = cand_conf.astype(np.float64).copy()
    scores_filtered[~keep_mask] = -1e9

    chosen = balanced_topk_indices(
        scores_filtered,
        labels=cand_labels,
        top_k=top_k,
        num_classes=10,
        alpha=float(alpha),
        valid_mask=keep_mask,
    )
    selected_indices = preds["indices"][cand_top][chosen].tolist()
    selected_labels = cand_labels[chosen].tolist()
    hist = np.bincount(cand_labels[chosen], minlength=10).astype(np.int64)
    stats = compute_selection_stats(
        true_labels=preds["true_labels"][cand_top][chosen],
        pseudo_labels=cand_labels[chosen],
        confidences=cand_conf[chosen],
        num_classes=10,
    )
    return selected_indices, selected_labels, {"pseudo_label_hist": hist.tolist(), **stats}

def select_baseline_score_topk(
    preds: Dict[str, np.ndarray],
    top_k: int,
    candidate_pool: int,
    score_floor: float,
    per_class_k: int,
    alpha: float,
) -> Tuple[List[int], List[int], Dict[str, float]]:
    """
    Strict control for v3g selection:
      - Same score_topk keep-mask as v3g (per-class top-k after a global score floor)
      - Same selection-time class-balance constraint (α)
      - BUT score is confidence only (no set-aware learned weights).

    This isolates whether the v3g gains come from learned set-aware scoring or just from
    hard per-class selection rules.
    """
    conf = preds["confidences"]
    order = np.argsort(-conf)
    cand_top = order[: min(len(order), int(candidate_pool))]
    cand_conf = conf[cand_top]
    cand_labels = preds["pseudo_labels"][cand_top]

    floor_mask = cand_conf >= float(score_floor)
    if not floor_mask.any():
        floor_mask = np.ones_like(floor_mask, dtype=bool)

    k = int(per_class_k)
    if k <= 0:
        k = int(np.ceil(top_k / 10))
    keep_mask = per_class_topk_mask(
        scores=cand_conf.astype(np.float64),
        labels=cand_labels,
        per_class_k=k,
        num_classes=10,
        base_mask=floor_mask,
    )
    if not keep_mask.any():
        keep_mask = floor_mask

    scores_filtered = cand_conf.astype(np.float64).copy()
    scores_filtered[~keep_mask] = -1e9
    chosen = balanced_topk_indices(
        scores_filtered,
        labels=cand_labels,
        top_k=top_k,
        num_classes=10,
        alpha=float(alpha),
        valid_mask=keep_mask,
    )

    selected_indices = preds["indices"][cand_top][chosen].tolist()
    selected_labels = cand_labels[chosen].tolist()
    hist = np.bincount(cand_labels[chosen], minlength=10).astype(np.int64)
    stats = compute_selection_stats(
        true_labels=preds["true_labels"][cand_top][chosen],
        pseudo_labels=cand_labels[chosen],
        confidences=cand_conf[chosen],
        num_classes=10,
    )
    return selected_indices, selected_labels, {"pseudo_label_hist": hist.tolist(), **stats}


def build_filter_input(features: torch.Tensor, pseudo_labels: torch.Tensor, confidences: torch.Tensor, num_classes: int) -> torch.Tensor:
    one_hot = F.one_hot(pseudo_labels, num_classes=num_classes).float()
    conf_exp = confidences.unsqueeze(1)
    return torch.cat([features, one_hot, conf_exp], dim=1)


def train_set_aware_filter(
    device: torch.device,
    features: np.ndarray,
    pseudo_labels: np.ndarray,
    confidences: np.ndarray,
    candidate_dataset_indices: np.ndarray,
    base_train: torchvision.datasets.CIFAR10,
    eval_transform: T.Compose,
    classifier: nn.Module,
    clean_val_set: Dataset | None,
    args: argparse.Namespace,
    seed: int,
) -> SetAwareBiasRobustFilter:
    num_classes = 10
    feat_dim = features.shape[1]
    in_dim = feat_dim + num_classes + 1
    model = SetAwareBiasRobustFilter(
        dim=in_dim,
        hidden=args.filter_hidden,
        n_heads=args.filter_heads,
        n_layers=args.filter_layers,
        dropout=args.filter_dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.filter_lr, weight_decay=args.filter_wd)
    features_t = torch.from_numpy(features).float().to(device)
    pseudo_t = torch.from_numpy(pseudo_labels).long().to(device)
    conf_t = torch.from_numpy(confidences).float().to(device)
    proto_t: torch.Tensor | None = None
    if float(args.lambda_proto) > 0.0:
        proto_scores = prototype_agreement_scores(
            features=features,
            pseudo_labels=pseudo_labels,
            confidences=confidences,
            num_classes=num_classes,
            topk_per_class=int(args.proto_topk),
            temperature=float(args.proto_temp),
            conf_power=float(args.proto_conf_power),
        )
        proto_t = torch.from_numpy(proto_scores).float().to(device)

    rng = np.random.default_rng(seed)
    clean_feats: torch.Tensor | None = None
    clean_targets: torch.Tensor | None = None
    clean_imgs: torch.Tensor | None = None
    if args.meta_clean_val:
        if clean_val_set is None:
            raise ValueError("--meta-clean-val requires a non-empty clean_val_set.")
        dl_seed = make_dataloader_seed(seed + 999)
        clean_loader = DataLoader(
            clean_val_set,
            batch_size=min(args.clean_val_size, len(clean_val_set)),
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            worker_init_fn=dl_seed["worker_init_fn"],
            generator=dl_seed["generator"],
        )
        classifier.eval()
        feats_list: List[torch.Tensor] = []
        targets_list: List[torch.Tensor] = []
        imgs_list: List[torch.Tensor] = []
        with torch.no_grad():
            for imgs, targets in clean_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                _, feats = forward_with_features(classifier, imgs)
                feats_list.append(feats)
                targets_list.append(targets)
                imgs_list.append(imgs)
        clean_feats = torch.cat(feats_list, dim=0)
        clean_targets = torch.cat(targets_list, dim=0)
        clean_imgs = torch.cat(imgs_list, dim=0)

    filter_params = [p for p in model.parameters() if p.requires_grad]
    pbar = tqdm(range(args.filter_steps), desc="Filter train", ncols=100, leave=False)
    for step in pbar:
        is_meta_step = args.meta_clean_val and (step % max(1, args.meta_every) == 0)
        set_size = int(args.meta_set_size if is_meta_step else args.filter_set_size)
        effective_size = min(set_size, len(features_t))
        # Sample without replacement for a stable set signal (duplicates weaken both balance and meta objectives).
        idx_np = rng.choice(len(features_t), size=effective_size, replace=False).astype(np.int64)
        idx = torch.from_numpy(idx_np).to(device)
        feat_batch = features_t.index_select(0, idx)
        pseudo_batch = pseudo_t.index_select(0, idx)
        conf_batch = conf_t.index_select(0, idx)
        x = build_filter_input(feat_batch, pseudo_batch, conf_batch, num_classes).unsqueeze(0)
        weights, delta_phi = model(x)
        weights = apply_delta_phi_correction(weights, delta_phi, x, scale=args.delta_phi_scale)
        target_conf = conf_batch.unsqueeze(0)
        loss_conf = classification_loss(weights, target_conf)
        loss_proto = None
        if proto_t is not None:
            target_proto = proto_t.index_select(0, idx).unsqueeze(0)
            loss_proto = classification_loss(weights, target_proto)
        weights_safe = weights.clamp_min(1e-6)
        class_onehot = F.one_hot(pseudo_batch, num_classes=num_classes).float()
        weighted_freq = (weights_safe.squeeze(0).unsqueeze(-1) * class_onehot).sum(dim=0) / weights_safe.sum()
        loss_balance = torch.sum((weighted_freq - 1.0 / num_classes) ** 2)
        loss_ess = ess_loss(weights, tau=args.filter_tau)
        loss = (
            args.lambda_conf * loss_conf
            + float(args.lambda_proto) * (loss_proto if loss_proto is not None else 0.0)
            + args.lambda_balance * loss_balance
            + args.lambda_ess * loss_ess
        )

        meta_loss = None
        if is_meta_step:
            assert clean_feats is not None and clean_targets is not None and clean_imgs is not None
            dataset_indices = candidate_dataset_indices[idx_np]
            imgs_list: List[torch.Tensor] = []
            for ds_idx in dataset_indices.tolist():
                img, _ = base_train[int(ds_idx)]
                imgs_list.append(eval_transform(img))
            dirty_imgs = torch.stack(imgs_list, dim=0).to(device)
            dirty_weights = weights.squeeze(0)

            if args.meta_update_scope == "fc":
                with torch.no_grad():
                    _, dirty_feats = forward_with_features(classifier, dirty_imgs)
                dirty_logits = F.linear(dirty_feats, classifier.fc.weight, classifier.fc.bias)
                per_sample = F.cross_entropy(dirty_logits, pseudo_batch, reduction="none")
                inner_loss = (dirty_weights * per_sample).sum() / dirty_weights.sum().clamp_min(1e-6)
                grad_w, grad_b = torch.autograd.grad(
                    inner_loss, [classifier.fc.weight, classifier.fc.bias], create_graph=True
                )
                w_updated = classifier.fc.weight.detach() - args.meta_inner_lr * grad_w
                b_updated = classifier.fc.bias.detach() - args.meta_inner_lr * grad_b
                clean_logits = F.linear(clean_feats, w_updated, b_updated)
                meta_loss = F.cross_entropy(clean_logits, clean_targets)
            else:
                per_sample = F.cross_entropy(classifier(dirty_imgs), pseudo_batch, reduction="none")
                inner_loss = (dirty_weights * per_sample).sum() / dirty_weights.sum().clamp_min(1e-6)
                params_to_update = dict(classifier.named_parameters())
                names = list(params_to_update.keys())
                tensors = [params_to_update[n] for n in names]
                grads = torch.autograd.grad(inner_loss, tensors, create_graph=True)
                params_and_buffers: Dict[str, torch.Tensor] = {}
                for name, tensor in classifier.named_parameters():
                    params_and_buffers[name] = tensor.detach()
                for name, tensor in classifier.named_buffers():
                    params_and_buffers[name] = tensor.detach()
                for name, tensor, grad in zip(names, tensors, grads):
                    params_and_buffers[name] = tensor.detach() - args.meta_inner_lr * grad
                clean_logits = torch.func.functional_call(classifier, params_and_buffers, (clean_imgs,))
                meta_loss = F.cross_entropy(clean_logits, clean_targets)

            loss = loss + args.meta_lambda * meta_loss

        opt.zero_grad(set_to_none=True)
        grads = torch.autograd.grad(loss, filter_params, allow_unused=True)
        for param, grad in zip(filter_params, grads):
            param.grad = grad
        opt.step()
        postfix = {"loss": float(loss.item())}
        if meta_loss is not None:
            postfix["meta"] = float(meta_loss.item())
        if loss_proto is not None:
            postfix["proto"] = float(loss_proto.item())
        pbar.set_postfix(postfix)
    pbar.close()
    return model


def select_set_aware(
    preds: Dict[str, np.ndarray],
    top_k: int,
    threshold: float,
    device: torch.device,
    args: argparse.Namespace,
    base_train: torchvision.datasets.CIFAR10,
    eval_transform: T.Compose,
    classifier: nn.Module,
    clean_val_set: Dataset | None,
    seed: int,
) -> Tuple[List[int], List[int], Dict[str, float]]:
    conf = preds["confidences"]
    order = np.argsort(-conf)
    cand_top = order[: min(len(order), args.filter_candidate_pool)]
    cand_features = preds["features"][cand_top]
    cand_labels = preds["pseudo_labels"][cand_top]
    cand_conf = conf[cand_top]
    cand_margin = preds["margins"][cand_top]
    cand_dataset_indices = preds["indices"][cand_top]
    filter_model = train_set_aware_filter(
        device=device,
        features=cand_features,
        pseudo_labels=cand_labels,
        confidences=cand_conf,
        candidate_dataset_indices=cand_dataset_indices,
        base_train=base_train,
        eval_transform=eval_transform,
        classifier=classifier,
        clean_val_set=clean_val_set,
        args=args,
        seed=seed,
    )

    with torch.no_grad():
        features_t = torch.from_numpy(cand_features).float().to(device)
        labels_t = torch.from_numpy(cand_labels).long().to(device)
        conf_t = torch.from_numpy(cand_conf).float().to(device)
        x = build_filter_input(features_t, labels_t, conf_t, num_classes=10).unsqueeze(0)
        weights, delta_phi = filter_model(x)
        weights = apply_delta_phi_correction(weights, delta_phi, x, scale=args.delta_phi_scale)
        weights = weights.squeeze(0).cpu().numpy()

    if args.set_aware_score_mode == "weight":
        scores = weights.astype(np.float64)
    else:
        scores = (weights * cand_conf).astype(np.float64)
    if args.set_aware_threshold_mode == "margin_quantile":
        keep_mask = per_class_quantile_mask(
            scores=cand_margin,
            labels=cand_labels,
            quantile=float(args.set_aware_margin_quantile),
            num_classes=10,
        )
        if keep_mask.sum() < top_k:
            keep_mask = np.ones_like(keep_mask, dtype=bool)
    elif args.set_aware_threshold_mode == "score_topk":
        floor_mask = scores >= float(args.set_aware_score_floor)
        if not floor_mask.any():
            floor_mask = np.ones_like(floor_mask, dtype=bool)
        per_class_k = int(args.set_aware_per_class_k)
        if per_class_k <= 0:
            per_class_k = int(np.ceil(top_k / 10))
        keep_mask = per_class_topk_mask(
            scores=scores,
            labels=cand_labels,
            per_class_k=per_class_k,
            num_classes=10,
            base_mask=floor_mask,
        )
        if not keep_mask.any():
            keep_mask = floor_mask
    else:
        keep_mask = cand_conf >= threshold
        if keep_mask.sum() == 0:
            keep_mask = np.ones_like(keep_mask, dtype=bool)
    scores_filtered = scores.copy()
    scores_filtered[~keep_mask] = -1e9
    chosen = balanced_topk_indices(
        scores_filtered,
        labels=cand_labels,
        top_k=top_k,
        num_classes=10,
        alpha=float(args.set_aware_balance_alpha),
        valid_mask=keep_mask,
    )
    selected_indices = preds["indices"][cand_top][chosen].tolist()
    selected_labels = cand_labels[chosen].tolist()
    hist = np.bincount(cand_labels[chosen], minlength=10).astype(np.int64)
    ess = float((weights.sum() ** 2) / (np.sum(weights ** 2) + 1e-8))
    stats = compute_selection_stats(
        true_labels=preds["true_labels"][cand_top][chosen],
        pseudo_labels=cand_labels[chosen],
        confidences=cand_conf[chosen],
        num_classes=10,
        weights=weights[chosen],
        scores=scores[chosen],
    )
    return selected_indices, selected_labels, {"pseudo_label_hist": hist.tolist(), "ess_score": ess, **stats}


def split_labeled_unlabeled(dataset: torchvision.datasets.CIFAR10, per_class: int) -> Tuple[List[int], List[int]]:
    labeled: List[int] = []
    unlabeled: List[int] = []
    counter = [0] * 10
    for idx, (_, target) in enumerate(dataset):
        if counter[target] < per_class:
            labeled.append(idx)
            counter[target] += 1
        else:
            unlabeled.append(idx)
    return labeled, unlabeled


def build_clean_val_set(
    base: Dataset,
    indices: List[int],
    eval_transform: T.Compose,
    args: argparse.Namespace,
    num_classes: int = 10,
) -> Dataset:
    if float(args.clean_val_noise_rate) > 0.0:
        return NoisyLabelSubset(
            base=base,
            indices=indices,
            transform=eval_transform,
            return_index=False,
            noise_rate=float(args.clean_val_noise_rate),
            noise_seed=int(args.clean_val_noise_seed),
            num_classes=num_classes,
        )
    return CIFARSubset(base, indices, transform=eval_transform, return_index=False)


def run_single_seed(
    seed: int,
    mode: str,
    args: argparse.Namespace,
    merged_rows: List[Dict[str, float | int | str]] | None,
    merged_path: pathlib.Path | None,
) -> Dict:
    set_seed(seed)
    device = torch.device(args.device)
    train_tf, eval_tf = build_transforms()
    base_train = torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=None)
    labeled_idx, unlabeled_idx = split_labeled_unlabeled(base_train, per_class=250)

    clean_val_set: Dataset | None = None
    if args.meta_clean_val:
        if args.clean_val_source == "test":
            base_test = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=None)
            clean_size = min(int(args.clean_val_size), len(base_test))
            if args.clean_val_strategy == "stratified":
                clean_val_indices = stratified_indices_from_targets(
                    targets=getattr(base_test, "targets", []),
                    total=clean_size,
                    seed=int(args.clean_val_seed),
                    num_classes=10,
                )
            else:
                rng = np.random.default_rng(int(args.clean_val_seed))
                clean_val_indices = rng.choice(len(base_test), size=clean_size, replace=False).tolist()
            clean_val_set = build_clean_val_set(base_test, clean_val_indices, eval_tf, args, num_classes=10)
            clean_val_index_set = set(clean_val_indices)
            test_indices = [i for i in range(len(base_test)) if i not in clean_val_index_set]
            test_set = CIFARSubset(base_test, test_indices, transform=eval_tf, return_index=False)
        else:
            # Train-holdout clean set: pick from the unlabeled pool (train split) and remove them from recursion.
            clean_size = min(int(args.clean_val_size), len(unlabeled_idx))
            if args.clean_val_strategy == "stratified":
                clean_val_indices = stratified_sample_from_indices(
                    indices=unlabeled_idx,
                    targets=getattr(base_train, "targets", []),
                    total=clean_size,
                    seed=int(args.clean_val_seed),
                    num_classes=10,
                )
            else:
                rng = np.random.default_rng(int(args.clean_val_seed))
                clean_val_indices = rng.choice(np.asarray(unlabeled_idx, dtype=np.int64), size=clean_size, replace=False).tolist()
            clean_val_set = build_clean_val_set(base_train, clean_val_indices, eval_tf, args, num_classes=10)
            clean_val_index_set = set(clean_val_indices)
            unlabeled_idx = [i for i in unlabeled_idx if i not in clean_val_index_set]
            test_set = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=eval_tf)
    else:
        test_set = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=eval_tf)

    if isinstance(clean_val_set, NoisyLabelSubset):
        print(
            f"Clean-val noise rate={args.clean_val_noise_rate:.3f} "
            f"(flipped {clean_val_set.flip_count}/{len(clean_val_set)} labels)"
        )

    label_map: Dict[int, int] = {}
    for idx in labeled_idx:
        _, target = base_train[idx]
        label_map[idx] = target

    model = torchvision.models.resnet18(num_classes=10)
    model.to(device)

    # Gen0 training
    train_ds = PseudoLabeledDataset(base_train, labeled_idx, label_map, transform=train_tf)
    train_classifier(
        model,
        dataset=train_ds,
        epochs=args.gen0_epochs,
        device=device,
        lr=args.lr_gen0,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        desc=f"Gen0 train [{mode}][seed={seed}]",
        grad_accum_steps=args.grad_accum_steps,
        use_amp=not args.no_amp,
        seed=seed,
    )
    dl_seed = make_dataloader_seed(seed + 2025)
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=dl_seed["worker_init_fn"],
        generator=dl_seed["generator"],
    )
    metrics: List[Dict] = []
    eval_res = evaluate(model, test_loader, device)
    metrics.append(
        {
            "generation": 0,
            "train_size": len(train_ds),
            **eval_res,
            "pseudo_label_hist": [0] * 10,
        }
    )
    if merged_rows is not None and merged_path is not None:
        merged_rows.extend(metrics_to_rows([metrics[-1]], method=mode))
        write_merged_csv(merged_rows, merged_path)

    # Recursive loop
    for gen in range(1, args.generations + 1):
        unlabeled_ds = CIFARSubset(base_train, unlabeled_idx, transform=eval_tf, return_index=True)
        preds = collect_predictions(
            model,
            unlabeled_ds,
            device,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            seed=seed + gen,  # vary seed per generation for reproducibility without clashes
        )
        if mode == "baseline":
            selected_indices, selected_labels, extra = select_baseline(
                preds, top_k=args.per_gen_add, threshold=args.baseline_conf_threshold
            )
        elif mode == "baseline_balanced":
            selected_indices, selected_labels, extra = select_baseline_balanced(
                preds,
                top_k=args.per_gen_add,
                threshold=args.baseline_conf_threshold,
                candidate_pool=args.filter_candidate_pool,
                alpha=float(args.set_aware_balance_alpha),
            )
        elif mode == "baseline_score_topk":
            selected_indices, selected_labels, extra = select_baseline_score_topk(
                preds,
                top_k=args.per_gen_add,
                candidate_pool=args.filter_candidate_pool,
                score_floor=float(args.set_aware_score_floor),
                per_class_k=int(args.set_aware_per_class_k),
                alpha=float(args.set_aware_balance_alpha),
            )
        elif mode == "k_center":
            selected_indices, selected_labels, extra = select_kcenter(
                preds,
                top_k=args.per_gen_add,
                candidate_pool=args.diversity_candidate_pool,
                proj_dim=int(args.diversity_proj_dim),
                seed=seed + 30_000 + gen,
            )
        elif mode == "dpp":
            selected_indices, selected_labels, extra = select_dpp(
                preds,
                top_k=args.per_gen_add,
                candidate_pool=args.dpp_candidate_pool,
                proj_dim=int(args.diversity_proj_dim),
                sigma=float(args.dpp_sigma),
                jitter=float(args.dpp_jitter),
                seed=seed + 40_000 + gen,
            )
        else:
            selected_indices, selected_labels, extra = select_set_aware(
                preds,
                top_k=args.per_gen_add,
                threshold=args.set_aware_conf_threshold,
                device=device,
                args=args,
                base_train=base_train,
                eval_transform=eval_tf,
                classifier=model,
                clean_val_set=clean_val_set,
                seed=seed + 10_000 + gen,
            )

        # Update pools
        for idx, label in zip(selected_indices, selected_labels):
            label_map[idx] = int(label)
        train_indices = labeled_idx + selected_indices
        labeled_idx = train_indices  # All selected become labeled going forward
        selected_set = set(selected_indices)
        unlabeled_idx = [i for i in unlabeled_idx if i not in selected_set]

        train_ds = PseudoLabeledDataset(base_train, train_indices, label_map, transform=train_tf)
        train_classifier(
            model,
            dataset=train_ds,
            epochs=args.finetune_epochs,
            device=device,
            lr=args.lr_finetune,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            desc=f"Gen{gen} train [{mode}][seed={seed}]",
            grad_accum_steps=args.grad_accum_steps,
            use_amp=not args.no_amp,
            seed=seed + gen,
        )
        eval_res = evaluate(model, test_loader, device)
        metrics.append(
            {
                "generation": gen,
                "train_size": len(train_ds),
                **eval_res,
                **extra,
            }
        )
        if merged_rows is not None and merged_path is not None:
            merged_rows.extend(metrics_to_rows([metrics[-1]], method=mode))
            write_merged_csv(merged_rows, merged_path)
    return {"seed": seed, "mode": mode, "metrics": metrics}


def save_results(results: Dict, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def save_seed_config(seed: int, args: argparse.Namespace, merged_path: pathlib.Path, json_path: pathlib.Path) -> None:
    cfg = {"seed": seed, "merged_path": str(merged_path), "args": vars(args)}
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def write_metrics_csv(metrics: List[Dict], csv_path: pathlib.Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, float | int | str]] = []
    for m in metrics:
        row: Dict[str, float | int | str] = {
            "generation": m.get("generation", 0),
            "train_size": m.get("train_size", 0),
            "acc": m.get("acc", 0.0),
            "worst_class_acc": m.get("worst_class_acc", 0.0),
            "ess_score": m.get("ess_score", ""),
        }
        per_class = m.get("per_class_acc", ["" for _ in range(10)])
        for i in range(10):
            row[f"acc_c{i}"] = per_class[i] if i < len(per_class) else ""
        hist = m.get("pseudo_label_hist", ["" for _ in range(10)])
        for i in range(10):
            row[f"hist_c{i}"] = hist[i] if i < len(hist) else ""
        row["sel_pseudo_acc"] = m.get("sel_pseudo_acc", "")
        row["sel_mean_conf"] = m.get("sel_mean_conf", "")
        for i in range(10):
            row[f"sel_pseudo_acc_c{i}"] = m.get(f"sel_pseudo_acc_c{i}", "")
        row["sel_mean_weight"] = m.get("sel_mean_weight", "")
        row["sel_mean_score"] = m.get("sel_mean_score", "")
        rows.append(row)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def metrics_to_rows(metrics: List[Dict], method: str) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    for m in metrics:
        row: Dict[str, float | int | str] = {"method": method}
        row.update(
            {
                "generation": m.get("generation", 0),
                "train_size": m.get("train_size", 0),
                "acc": m.get("acc", 0.0),
                "worst_class_acc": m.get("worst_class_acc", 0.0),
                "ess_score": m.get("ess_score", ""),
            }
        )
        per_class = m.get("per_class_acc", ["" for _ in range(10)])
        for i in range(10):
            row[f"acc_c{i}"] = per_class[i] if i < len(per_class) else ""
        hist = m.get("pseudo_label_hist", ["" for _ in range(10)])
        for i in range(10):
            row[f"hist_c{i}"] = hist[i] if i < len(hist) else ""
        row["sel_pseudo_acc"] = m.get("sel_pseudo_acc", "")
        row["sel_mean_conf"] = m.get("sel_mean_conf", "")
        for i in range(10):
            row[f"sel_pseudo_acc_c{i}"] = m.get(f"sel_pseudo_acc_c{i}", "")
        row["sel_mean_weight"] = m.get("sel_mean_weight", "")
        row["sel_mean_score"] = m.get("sel_mean_score", "")
        rows.append(row)
    return rows


def write_merged_csv(all_rows: List[Dict[str, float | int | str]], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MERGED_HEADERS)
        writer.writeheader()
        writer.writerows(all_rows)

def seed_results_complete(merged_path: pathlib.Path, modes: List[str], generations: int) -> bool:
    """
    Check whether a per-seed merged CSV already contains all required (mode, generation) rows.
    Used to avoid re-running expensive experiments and to prevent overwriting existing outputs.
    """
    if not merged_path.exists():
        return False
    required = {(m, g) for m in modes for g in range(int(generations) + 1)}
    seen: set[tuple[str, int]] = set()
    with merged_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = str(row.get("method", "")).strip()
            if not method:
                continue
            try:
                gen = int(float(row.get("generation", -1)))
            except (TypeError, ValueError):
                continue
            seen.add((method, gen))
    return required.issubset(seen)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 recursive self-training with Set-Aware filter.")
    parser.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(ROOT / "Total_results" / "Tables" / "exp9_cifar10_setaware" / "results"),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["baseline", "set_aware"],
        choices=[
            "baseline",
            "baseline_balanced",
            "baseline_score_topk",
            "k_center",
            "dpp",
            "set_aware",
        ],
        help="Which methods to run. baseline_balanced uses Top-Conf + the same class-balance quotas as set_aware; "
        "baseline_score_topk uses the v3g score_topk keep-mask but scores by confidence only (no learned weights). "
        "k_center and dpp are diversity-only selection baselines over feature space.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1088, 2195, 4960])
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--per-gen-add", type=int, default=4000)
    parser.add_argument(
        "--baseline-conf-threshold",
        type=float,
        default=0.0,
        help="Threshold for baseline (Top-K); use 0 to force pure ranking.",
    )
    parser.add_argument("--set-aware-conf-threshold", type=float, default=0.0, help="Threshold for Set-Aware filtering.")
    parser.add_argument(
        "--set-aware-threshold-mode",
        choices=["confidence", "margin_quantile", "score_topk"],
        default="confidence",
        help="How to build the keep-mask for set-aware selection.",
    )
    parser.add_argument(
        "--set-aware-margin-quantile",
        type=float,
        default=0.5,
        help="Top-q fraction per class to keep when using margin_quantile mode.",
    )
    parser.add_argument(
        "--set-aware-score-floor",
        type=float,
        default=0.4,
        help="Minimum score to keep before per-class top-k in score_topk mode.",
    )
    parser.add_argument(
        "--set-aware-per-class-k",
        type=int,
        default=0,
        help="Max items per class for score_topk mode (0 uses per-gen-add / num_classes).",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gen0-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--lr-gen0", type=float, default=0.1)
    parser.add_argument("--lr-finetune", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Set-aware hyperparameters
    parser.add_argument("--filter-hidden", type=int, default=256)
    parser.add_argument("--filter-heads", type=int, default=4)
    parser.add_argument("--filter-layers", type=int, default=2)
    parser.add_argument("--filter-dropout", type=float, default=0.1)
    parser.add_argument("--filter-steps", type=int, default=200)
    parser.add_argument("--filter-set-size", type=int, default=2048)
    parser.add_argument("--filter-candidate-pool", type=int, default=12000)
    parser.add_argument("--filter-lr", type=float, default=1e-3)
    parser.add_argument("--filter-wd", type=float, default=1e-4)
    parser.add_argument("--filter-tau", type=float, default=0.2)
    # Diversity-only selection baselines
    parser.add_argument(
        "--diversity-candidate-pool",
        type=int,
        default=12000,
        help="Candidate pool size for k-center/DPP selection (from top-confidence candidates).",
    )
    parser.add_argument(
        "--diversity-proj-dim",
        type=int,
        default=64,
        help="Random projection dimension for diversity selection (0 disables).",
    )
    parser.add_argument(
        "--dpp-candidate-pool",
        type=int,
        default=4000,
        help="Candidate pool size for DPP baseline (must be >= per-gen-add).",
    )
    parser.add_argument(
        "--dpp-sigma",
        type=float,
        default=0.0,
        help="RBF kernel bandwidth for DPP; 0 uses median pairwise distance.",
    )
    parser.add_argument(
        "--dpp-jitter",
        type=float,
        default=1e-6,
        help="Diagonal jitter for DPP kernel stability.",
    )
    parser.add_argument(
        "--delta-phi-scale",
        type=float,
        default=1.0,
        help="Δϕ logit-correction scale η; set 0.0 to disable using Δϕ in set-aware selection.",
    )
    parser.add_argument("--lambda-conf", type=float, default=1.0)
    parser.add_argument(
        "--lambda-proto",
        type=float,
        default=0.0,
        help="Aux loss weight: make weights track a prototype-agreement score (helps within-class de-noising).",
    )
    parser.add_argument("--lambda-balance", type=float, default=1.0)
    parser.add_argument("--lambda-ess", type=float, default=0.1)
    parser.add_argument(
        "--proto-topk",
        type=int,
        default=200,
        help="Top-K per pseudo-class (by confidence) to build prototypes for agreement scoring.",
    )
    parser.add_argument(
        "--proto-temp",
        type=float,
        default=0.2,
        help="Softmax temperature for prototype agreement scoring (smaller => sharper).",
    )
    parser.add_argument(
        "--proto-conf-power",
        type=float,
        default=1.0,
        help="Exponent applied to confidence when averaging prototypes (>=0).",
    )
    parser.add_argument(
        "--set-aware-score-mode",
        choices=["weight_conf", "weight"],
        default="weight_conf",
        help="How to score candidates inside set_aware: weight_conf uses w*conf, weight uses w only (stronger deviation from confidence).",
    )
    parser.add_argument(
        "--set-aware-balance-alpha",
        type=float,
        default=0.0,
        help="Selection-time class-balance strength α∈[0,1]; 0 disables (global top-k), 1 targets uniform class quotas.",
    )
    parser.add_argument(
        "--meta-clean-val",
        action="store_true",
        help="Enable clean-validation meta-training (Scheme A) for the set-aware filter.",
    )
    parser.add_argument("--clean-val-size", type=int, default=100, help="Clean validation size (from CIFAR-10 test).")
    parser.add_argument("--clean-val-seed", type=int, default=0, help="Seed for selecting clean validation indices.")
    parser.add_argument(
        "--clean-val-strategy",
        choices=["stratified", "random"],
        default="stratified",
        help="How to choose the clean validation set indices from CIFAR-10 test split.",
    )
    parser.add_argument(
        "--clean-val-source",
        choices=["test", "train_holdout"],
        default="test",
        help="Where to draw the clean validation set from: 'test' (and exclude from test eval) or 'train_holdout' (from train unlabeled pool).",
    )
    parser.add_argument(
        "--clean-val-noise-rate",
        type=float,
        default=0.0,
        help="Label noise rate injected into the clean validation set (symmetric flips).",
    )
    parser.add_argument(
        "--clean-val-noise-seed",
        type=int,
        default=0,
        help="Seed for clean-val label noise injection.",
    )
    parser.add_argument("--meta-lambda", type=float, default=1.0, help="Weight of clean-val meta loss.")
    parser.add_argument("--meta-inner-lr", type=float, default=0.1, help="Inner (lookahead) LR used in meta step.")
    parser.add_argument("--meta-every", type=int, default=10, help="Apply meta loss every k filter steps.")
    parser.add_argument(
        "--meta-set-size",
        type=int,
        default=256,
        help="Set size used for meta steps (dirty batch size).",
    )
    parser.add_argument(
        "--meta-update-scope",
        choices=["fc", "all"],
        default="fc",
        help="Which classifier parameters to update in the meta lookahead step.",
    )
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision (AMP).")
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Allow overwriting existing per-seed merged/config outputs under --results-dir (default: refuse/skip).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = pathlib.Path(args.results_dir)
    for seed in args.seeds:
        merged_path = results_dir / f"exp9_seed{seed}_merged.csv"
        config_path = results_dir / f"exp9_seed{seed}_config.json"

        if seed_results_complete(merged_path, modes=list(args.modes), generations=args.generations):
            print(f"Skip seed={seed}: found complete results at {merged_path}")
            continue
        if merged_path.exists() and not args.overwrite_results:
            raise FileExistsError(
                f"Refusing to overwrite existing outputs for seed={seed} under {results_dir}. "
                f"Delete {merged_path}, change --results-dir, or pass --overwrite-results."
            )

        merged_rows: List[Dict[str, float | int | str]] = []
        # Always (re-)write the seed config for this attempt so it matches the run.
        # This is safe because we refuse to overwrite merged results unless --overwrite-results is set.
        save_seed_config(seed, args, merged_path, config_path)
        for mode in args.modes:
            run_result = run_single_seed(seed, mode, args, merged_rows=merged_rows, merged_path=merged_path)
            print(f"Completed mode={mode}, seed={seed}")
        if merged_rows:
            write_merged_csv(merged_rows, merged_path)
            print(f"Saved merged results for seed={seed} at {merged_path}")


if __name__ == "__main__":
    main()
