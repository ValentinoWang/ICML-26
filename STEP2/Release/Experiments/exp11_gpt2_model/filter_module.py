from typing import Dict, List, Tuple, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW

from filter.losses import ess_loss
from filter.set_aware.model import SetAwareBiasRobustFilter

def apply_delta_phi_correction(
    weights: torch.Tensor,
    delta_phi: torch.Tensor,
    x: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Use Δϕ as a global correction direction to modulate per-sample weights:
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

def apply_ppl_leash(
    weights: torch.Tensor,
    ppl_scores: List[float],
    ppl_ref: float | None,
    tau: float,
    mode: str,
    strength: float,
) -> torch.Tensor:
    """
    Semantic leash based on clean-validation perplexity.

    Penalizes candidates whose perplexity is (much) worse than a reference ppl_ref,
    using log-perplexity distances for scale robustness.

    Args:
        weights: [N] weights in [0,1].
        ppl_scores: length-N list of candidate perplexities under current model.
        ppl_ref: reference perplexity (e.g., previous generation's clean val_ppl). If None, no-op.
        tau: temperature on log-perplexity distance (must be > 0).
        mode: 'upper' penalizes only ppl>ref; 'abs' penalizes both sides; 'lower' penalizes only ppl<ref.
        strength: multiplier on the penalty (0 disables).
    """
    if ppl_ref is None or strength == 0.0:
        return weights
    if tau <= 0.0:
        raise ValueError("ppl_leash_tau must be > 0")
    if mode not in {"upper", "abs", "lower"}:
        raise ValueError(f"Unsupported ppl_leash_mode={mode!r}")

    eps = 1e-8
    ppl_ref = float(max(ppl_ref, eps))
    log_ref = float(np.log(ppl_ref))
    ppl = torch.tensor(ppl_scores, dtype=torch.float32, device=weights.device).clamp_min(eps)
    log_ppl = torch.log(ppl)
    if mode == "upper":
        dist = torch.relu(log_ppl - log_ref)
    elif mode == "lower":
        dist = torch.relu(log_ref - log_ppl)
    else:
        dist = torch.abs(log_ppl - log_ref)
    leash = torch.exp(-(dist / tau) * strength)
    return weights * leash


def dispersion_targets(x: torch.Tensor, k: int, temperature: float) -> Tuple[torch.Tensor, float]:
    """
    Args:
        x: [N, D] embeddings
        k: number of reference points for dispersion
    Returns:
        target weights in [0,1], collapse score (mean pairwise distance)
    """
    n = x.size(0)
    k = min(k, n)
    idx = torch.randperm(n)[:k]
    ref = x[idx]
    dist = torch.cdist(x, ref)
    mean_dist = dist.mean(dim=1)
    collapse_score = float(dist.mean().item())
    z = (mean_dist - mean_dist.mean()) / (mean_dist.std() + 1e-6)
    target = torch.sigmoid(z / temperature)
    return target, collapse_score


def train_set_aware_filter(
    embeddings: torch.Tensor,
    device: torch.device,
    set_size: int,
    steps: int,
    lr: float,
    hidden: int,
    heads: int,
    layers: int,
    dropout: float,
    knn: int,
    temperature: float,
    ess_tau: float,
    ess_weight: float,
    delta_phi_scale: float,
) -> SetAwareBiasRobustFilter:
    set_size = min(set_size, embeddings.size(0))
    model = SetAwareBiasRobustFilter(
        dim=embeddings.size(1),
        hidden=hidden,
        n_heads=heads,
        n_layers=layers,
        dropout=dropout,
    ).to(device)
    opt = AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randperm(embeddings.size(0))[:set_size]
        subset = embeddings[idx].to(device)
        target, _ = dispersion_targets(subset, k=knn, temperature=temperature)
        weights, delta_phi = model(subset.unsqueeze(0))
        weights = apply_delta_phi_correction(weights, delta_phi, subset.unsqueeze(0), scale=delta_phi_scale).squeeze(0)
        loss = nn.functional.mse_loss(weights, target)
        loss = loss + ess_weight * ess_loss(weights.unsqueeze(0), tau=ess_tau)
        entropy = -(weights * torch.log(weights + 1e-8)).mean()
        loss = loss - 0.1 * entropy
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
    return model


def infer_set_aware_weights(
    model: SetAwareBiasRobustFilter,
    embeddings: torch.Tensor,
    device: torch.device,
    set_size: int,
    delta_phi_scale: float,
) -> torch.Tensor:
    model.eval()
    weights = torch.zeros(embeddings.size(0))
    counts = torch.zeros(embeddings.size(0))
    with torch.no_grad():
        for start in range(0, embeddings.size(0), set_size):
            end = min(start + set_size, embeddings.size(0))
            subset = embeddings[start:end].to(device)
            w, delta_phi = model(subset.unsqueeze(0))
            w = apply_delta_phi_correction(w, delta_phi, subset.unsqueeze(0), scale=delta_phi_scale)
            weights[start:end] += w.squeeze(0).cpu()
            counts[start:end] += 1
    return weights / counts.clamp_min(1e-6)


def select_training_indices(
    method: str,
    train_size: int,
    ppl_scores: List[float],
    embeddings: torch.Tensor,
    texts: Sequence[str] | None,
    filter_args: Dict,
    device: torch.device,
) -> List[int]:
    n = len(ppl_scores)
    if method == "no_filter":
        return np.random.choice(n, size=train_size, replace=False).tolist()
    if method == "pointwise":
        return list(np.argsort(np.array(ppl_scores))[:train_size])
    if method == "dispersion":
        embeddings_device = embeddings.to(device)
        weights, _ = dispersion_targets(
            embeddings_device,
            k=int(filter_args.get("knn", 32)),
            temperature=float(filter_args.get("temperature", 1.0)),
        )
        weights = apply_ppl_leash(
            weights,
            ppl_scores=ppl_scores,
            ppl_ref=filter_args.get("ppl_ref"),
            tau=float(filter_args.get("ppl_leash_tau", 1.0)),
            mode=str(filter_args.get("ppl_leash_mode", "upper")),
            strength=float(filter_args.get("ppl_leash_strength", 0.0)),
        )
        return torch.topk(weights, k=train_size).indices.tolist()
    if method == "ppl_safety":
        ppl_ref = filter_args.get("ppl_ref")
        tau = float(filter_args.get("ppl_leash_tau", 1.0))
        mode = str(filter_args.get("ppl_leash_mode", "upper"))
        strength = float(filter_args.get("ppl_leash_strength", 0.0))
        min_weight = float(filter_args.get("ppl_safety_min_weight", 0.5))
        if ppl_ref is None:
            raise ValueError("method='ppl_safety' requires ppl_ref; set --ppl-leash-strength > 0 to initialize it.")
        weights = torch.ones(n, dtype=torch.float32, device=device)
        weights = apply_ppl_leash(
            weights,
            ppl_scores=ppl_scores,
            ppl_ref=float(ppl_ref),
            tau=tau,
            mode=mode,
            strength=strength,
        )
        safe = (weights >= min_weight).nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
        if safe.size >= train_size:
            chosen = np.random.choice(safe, size=train_size, replace=False)
            return chosen.tolist()
        # Fallback: take all safe then fill by descending safety weights.
        topk = torch.topk(weights, k=train_size).indices.cpu().numpy()
        if safe.size == 0:
            return topk.tolist()
        remaining = [i for i in topk.tolist() if i not in set(safe.tolist())]
        fill = remaining[: max(0, train_size - int(safe.size))]
        return safe.tolist() + fill
    if method == "rep_filter":
        if texts is None:
            raise ValueError("method='rep_filter' requires candidate texts.")
        ngram_n = int(filter_args.get("rep_filter_n", 4))
        thr = float(filter_args.get("rep_filter_threshold", 0.6))

        def rep_ratio(text: str) -> float:
            toks = [t for t in text.strip().split() if t]
            if len(toks) < ngram_n:
                return 0.0
            total = len(toks) - ngram_n + 1
            grams = set(tuple(toks[i : i + ngram_n]) for i in range(total))
            unique = len(grams)
            return 1.0 - (unique / float(total)) if total > 0 else 0.0

        reps = np.array([rep_ratio(t) for t in texts], dtype=np.float32)
        keep = np.nonzero(reps <= thr)[0]
        if keep.size >= train_size:
            chosen = np.random.choice(keep, size=train_size, replace=False)
            return chosen.tolist()
        # Fill with lowest repetition among the remaining candidates.
        order = np.argsort(reps)
        picked = keep.tolist()
        for idx in order.tolist():
            if idx in picked:
                continue
            picked.append(idx)
            if len(picked) >= train_size:
                break
        return picked[:train_size]
    if method == "ppl_leash":
        ppl_ref = filter_args.get("ppl_ref")
        tau = float(filter_args.get("ppl_leash_tau", 1.0))
        mode = str(filter_args.get("ppl_leash_mode", "upper"))
        strength = float(filter_args.get("ppl_leash_strength", 0.0))
        if ppl_ref is None:
            raise ValueError("method='ppl_leash' requires a ppl_ref; pass --ppl-leash-strength > 0 to initialize it.")
        weights = torch.ones(n, dtype=torch.float32, device=device)
        weights = apply_ppl_leash(
            weights,
            ppl_scores=ppl_scores,
            ppl_ref=float(ppl_ref),
            tau=tau,
            mode=mode,
            strength=strength,
        )
        return torch.topk(weights, k=train_size).indices.tolist()
    filter_model = train_set_aware_filter(
        embeddings=embeddings,
        device=device,
        set_size=filter_args["set_size"],
        steps=filter_args["steps"],
        lr=filter_args["lr"],
        hidden=filter_args["hidden"],
        heads=filter_args["heads"],
        layers=filter_args["layers"],
        dropout=filter_args["dropout"],
        knn=filter_args["knn"],
        temperature=filter_args["temperature"],
        ess_tau=filter_args["ess_tau"],
        ess_weight=filter_args["ess_weight"],
        delta_phi_scale=filter_args.get("delta_phi_scale", 0.0),
    )
    w = infer_set_aware_weights(
        filter_model,
        embeddings,
        device=device,
        set_size=filter_args["set_size"],
        delta_phi_scale=filter_args.get("delta_phi_scale", 0.0),
    )
    w = apply_ppl_leash(
        w,
        ppl_scores=ppl_scores,
        ppl_ref=filter_args.get("ppl_ref"),
        tau=float(filter_args.get("ppl_leash_tau", 1.0)),
        mode=str(filter_args.get("ppl_leash_mode", "upper")),
        strength=float(filter_args.get("ppl_leash_strength", 1.0)),
    )
    del filter_model
    torch.cuda.empty_cache()
    topk = torch.topk(w, k=train_size).indices.tolist()
    return topk
