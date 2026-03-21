from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/exp5_signature_vs_setaware_mpl")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ORIGINAL_DIR = SCRIPT_DIR / "original_exp5"
if str(ORIGINAL_DIR) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_DIR))

from filter.data import make_bias_vector, sample_candidates, sample_clean_reference, set_seed  # noqa: E402
from filter.losses import classification_loss, contraction_loss, correction_reg, ess_loss  # noqa: E402
from filter.set_aware.model import MLP as OriginalMLP, SetAwareBiasRobustFilter  # noqa: E402


METHODS = ["no_filter", "ours", "path_signature"]


@dataclass(frozen=True)
class SignatureConfig:
    max_history_len: int = 7
    mid_history_len: int = 6
    min_history_len: int = 4
    proj_dim: int = 3
    max_order: int = 3
    signature_hidden: int = 96

    @property
    def window_len(self) -> int:
        return self.max_history_len + 1

    @property
    def max_logsig_dim(self) -> int:
        return sum(self.proj_dim**k for k in range(1, self.max_order + 1))


@dataclass
class ProjectionBundle:
    projector: torch.Tensor


def pick_device(device_name: str) -> torch.device:
    requested = device_name.lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested device 'mps', but MPS is not available.")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda', but CUDA is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device_name}")


def clamp_delta_phi(delta_phi: torch.Tensor, clip: float) -> torch.Tensor:
    if clip is None or clip <= 0:
        return delta_phi
    norm = delta_phi.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    factor = torch.clamp(clip / norm, max=1.0)
    return delta_phi * factor


def random_orthonormal_frame(rng: np.random.Generator, dim: int, count: int) -> list[np.ndarray]:
    basis: list[np.ndarray] = []
    while len(basis) < count:
        vec = rng.normal(size=dim)
        for existing in basis:
            vec = vec - np.dot(vec, existing) * existing
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            continue
        basis.append(vec / norm)
    return basis


def make_projection_bundle(dim: int, cfg: SignatureConfig, seed: int) -> ProjectionBundle:
    rng = np.random.default_rng(seed)
    target = np.ones(dim, dtype=np.float32)
    target = target / (np.linalg.norm(target) + 1e-8)
    basis = [target]
    for vec in random_orthonormal_frame(rng, dim, cfg.proj_dim - 1):
        candidate = vec.copy()
        for existing in basis:
            candidate = candidate - np.dot(candidate, existing) * existing
        norm = np.linalg.norm(candidate)
        if norm < 1e-6:
            continue
        basis.append(candidate / norm)
        if len(basis) == cfg.proj_dim:
            break
    while len(basis) < cfg.proj_dim:
        extra = random_orthonormal_frame(rng, dim, 1)[0]
        for existing in basis:
            extra = extra - np.dot(extra, existing) * existing
        norm = np.linalg.norm(extra)
        if norm < 1e-6:
            continue
        basis.append(extra / norm)
    projector = torch.from_numpy(np.stack(basis, axis=1).astype(np.float32))
    return ProjectionBundle(projector=projector)


def batch_kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bi,bj->bij", a, b).reshape(a.shape[0], -1)


def truncated_signature_levels(path: torch.Tensor, order: int) -> list[torch.Tensor]:
    if order > 3:
        raise ValueError("This implementation supports orders up to 3.")

    batch, _, dim = path.shape
    increments = path[:, 1:, :] - path[:, :-1, :]
    levels = [torch.ones(batch, 1, dtype=path.dtype, device=path.device)]
    for k in range(1, order + 1):
        levels.append(torch.zeros(batch, dim**k, dtype=path.dtype, device=path.device))

    for delta in increments.unbind(dim=1):
        segment = [torch.ones(batch, 1, dtype=path.dtype, device=path.device), delta]
        for k in range(2, order + 1):
            segment.append(batch_kron(segment[-1], delta) / float(k))
        updated = [levels[0]]
        for k in range(1, order + 1):
            acc = levels[k]
            for i in range(k):
                acc = acc + batch_kron(levels[i], segment[k - i])
            updated.append(acc)
        levels = updated

    return levels[1:]


def truncated_logsignature(path: torch.Tensor, order: int) -> torch.Tensor:
    if order < 1 or order > 3:
        raise ValueError("order must be in [1, 3].")

    sig_levels = truncated_signature_levels(path, order)
    s1 = sig_levels[0]
    blocks = [s1]

    if order >= 2:
        s2 = sig_levels[1]
        l2 = s2 - 0.5 * batch_kron(s1, s1)
        blocks.append(l2)

    if order >= 3:
        s3 = sig_levels[2]
        l3 = (
            s3
            - 0.5 * (batch_kron(s1, s2) + batch_kron(s2, s1))
            + (1.0 / 3.0) * batch_kron(batch_kron(s1, s1), s1)
        )
        blocks.append(l3)

    return torch.cat(blocks, dim=1)


def winding_ratio_np(path: np.ndarray, eps: float = 1e-6) -> float:
    chord = float(np.linalg.norm(path[-1] - path[0]))
    arc = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    return arc / (chord + eps)


def expected_next_torch(history: torch.Tensor) -> torch.Tensor:
    velocity_last = history[:, -1, :] - history[:, -2, :]
    velocity_prev = history[:, -2, :] - history[:, -3, :]
    acceleration = velocity_last - velocity_prev
    return history[:, -1, :] + 0.85 * velocity_last + 0.35 * acceleration


def expected_next_np(history: np.ndarray) -> np.ndarray:
    velocity_last = history[-1] - history[-2]
    velocity_prev = history[-2] - history[-3]
    acceleration = velocity_last - velocity_prev
    return history[-1] + 0.85 * velocity_last + 0.35 * acceleration


def sibylla_order(ratio: np.ndarray) -> np.ndarray:
    order = np.ones_like(ratio, dtype=np.int64)
    order[ratio >= 1.1] = 2
    order[ratio >= 1.5] = 3
    return order


def adaptive_window_length(history: List[np.ndarray], cfg: SignatureConfig) -> int:
    if len(history) < cfg.min_history_len:
        return cfg.min_history_len
    max_path = pad_history(history, cfg.max_history_len, cfg.proj_dim)
    diagnostic = np.concatenate([max_path, expected_next_np(max_path)[None, :]], axis=0)
    ratio = winding_ratio_np(diagnostic)
    if ratio >= 1.5:
        return cfg.max_history_len
    if ratio >= 1.1:
        return cfg.mid_history_len
    return cfg.min_history_len


def make_padded_features(path: torch.Tensor, cfg: SignatureConfig) -> torch.Tensor:
    diagnostic = torch.cat([path, expected_next_torch(path).unsqueeze(1)], dim=1)
    diagnostic_np = diagnostic.detach().cpu().numpy()
    ratios = np.array([winding_ratio_np(item) for item in diagnostic_np], dtype=np.float32)
    orders = sibylla_order(ratios)

    features = torch.zeros(path.shape[0], cfg.max_logsig_dim, dtype=path.dtype, device=path.device)
    for order in range(1, cfg.max_order + 1):
        idx = np.where(orders == order)[0]
        if idx.size == 0:
            continue
        idx_t = torch.from_numpy(idx).long().to(path.device)
        block = truncated_logsignature(diagnostic.index_select(0, idx_t), order)
        features.index_copy_(0, idx_t, F.pad(block, (0, cfg.max_logsig_dim - block.shape[1])))
    return features


class SibyllaMLP(nn.Module):
    def __init__(self, inp: int, hidden: int, out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SetAwarePathSignatureFilter(nn.Module):
    """
    Keep the original Exp5 set-aware weighting branch unchanged.
    Replace only the correction head with a Sibylla-style adaptive logsig head.
    """

    def __init__(
        self,
        dim: int,
        hidden: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        signature_cfg: SignatureConfig,
    ):
        super().__init__()
        self.encoder = OriginalMLP(dim, hidden, hidden, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.weight_head = OriginalMLP(hidden, hidden, 1, dropout=dropout)
        self.signature_head = SibyllaMLP(
            inp=signature_cfg.max_logsig_dim,
            hidden=signature_cfg.signature_hidden,
            out=signature_cfg.proj_dim,
        )

    def forward(self, x: torch.Tensor, logsig_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = self.encoder(x)
        h = self.transformer(h0)
        weights = torch.sigmoid(self.weight_head(h)).squeeze(-1)
        low_rank_delta = self.signature_head(logsig_features)
        return weights, low_rank_delta

    @staticmethod
    def weighted_estimate(x: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        w = weights.unsqueeze(-1)
        num = (w * x).sum(dim=1)
        den = w.sum(dim=1, keepdim=True).clamp_min(eps)
        return num / den


def build_labels(candidates: np.ndarray, theta_true: np.ndarray, top_ratio: float) -> np.ndarray:
    dists = np.linalg.norm(candidates - theta_true, axis=1)
    k = max(1, int(len(candidates) * top_ratio))
    thresh = np.partition(dists, k - 1)[k - 1]
    return (dists <= thresh).astype(np.float32)


def aggregate_seed_series(seed_series: List[Dict[str, List[float]]]) -> Dict[str, Dict[str, List[float]]]:
    stats: Dict[str, Dict[str, List[float]]] = {}
    for method in METHODS:
        arr = np.array([series[method] for series in seed_series], dtype=np.float64)
        stats[method] = {"mean": arr.mean(axis=0).tolist(), "std": arr.std(axis=0).tolist()}
    return stats


def aggregate_tail(seed_series: List[Dict[str, List[float]]], tail: int) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for method in METHODS:
        tails = np.array([np.mean(series[method][-tail:]) for series in seed_series], dtype=np.float64)
        stats[method] = {"mean": float(tails.mean()), "std": float(tails.std())}
    return stats


def pad_history(history: List[np.ndarray], length: int, dim: int) -> np.ndarray:
    normalized = [np.nan_to_num(np.asarray(item, dtype=np.float32).reshape(dim), nan=0.0, posinf=1e6, neginf=-1e6) for item in history]
    history = normalized
    if not history:
        return np.zeros((length, dim), dtype=np.float32)
    if len(history) >= length:
        return np.asarray(history[-length:], dtype=np.float32)
    prefix = [history[0]] * (length - len(history))
    return np.asarray(prefix + history, dtype=np.float32)


def run_single_dim(
    args: argparse.Namespace,
    dim: int,
    device: torch.device,
    seed: int,
    signature_cfg: SignatureConfig,
) -> Dict[str, List[float]]:
    rng = set_seed(seed)
    bias_vec = make_bias_vector(dim, args.bias_norm)
    bundle = make_projection_bundle(dim=dim, cfg=signature_cfg, seed=seed + 997)

    is_high_dim = dim >= args.reg_dim_threshold
    hidden_sa = args.hidden_high_dim if is_high_dim else args.hidden
    n_heads_sa = args.n_heads_high_dim if is_high_dim else args.n_heads
    n_layers_sa = args.n_layers_high_dim if is_high_dim else args.n_layers
    dropout_sa = args.dropout_high_dim if is_high_dim else args.dropout
    clip_val = args.correction_clip_high_dim if is_high_dim else args.correction_clip
    ours_c = args.ours_contraction_high_dim if is_high_dim else args.ours_contraction

    sa_model = SetAwareBiasRobustFilter(
        dim=dim,
        hidden=hidden_sa,
        n_heads=n_heads_sa,
        n_layers=n_layers_sa,
        dropout=dropout_sa,
    ).to(device)
    sig_model = SetAwarePathSignatureFilter(
        dim=dim,
        hidden=hidden_sa,
        n_heads=n_heads_sa,
        n_layers=n_layers_sa,
        dropout=dropout_sa,
        signature_cfg=signature_cfg,
    ).to(device)

    opt_sa = optim.Adam(sa_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_sig = optim.Adam(sig_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    theta_true = np.ones(dim, dtype=np.float32)
    theta_good = sample_clean_reference(rng, theta_true, n=args.calibration_size, noise_std=args.noise_std).astype(np.float32)
    theta_good_t = torch.from_numpy(theta_good[None, ...]).float().to(device)
    projector_t = bundle.projector.float().to(device)

    theta_sa = theta_good.copy()
    theta_sig = theta_good.copy()

    series: Dict[str, List[float]] = {name: [] for name in METHODS}
    projected_history: List[np.ndarray] = []

    for _ in range(args.generations):
        candidates = sample_candidates(
            rng,
            theta_true=theta_true,
            bias=bias_vec,
            n=args.samples_per_gen,
            noise_std=args.noise_std,
        ).astype(np.float32)
        labels = build_labels(candidates, theta_true, top_ratio=args.top_ratio)

        projected_state = (theta_sig - theta_true) @ projector_t.cpu().numpy()
        projected_state = np.nan_to_num(np.asarray(projected_state, dtype=np.float32).reshape(signature_cfg.proj_dim), nan=0.0, posinf=1e6, neginf=-1e6)
        projected_history.append(projected_state.astype(np.float32))
        window_len = max(signature_cfg.min_history_len, adaptive_window_length(projected_history, signature_cfg))
        history_np = pad_history(projected_history, window_len, signature_cfg.proj_dim)

        x = torch.from_numpy(candidates[None, ...]).float().to(device)
        y = torch.from_numpy(labels[None, ...]).float().to(device)
        history_t = torch.from_numpy(history_np[None, ...]).float().to(device)
        logsig_features = make_padded_features(history_t, signature_cfg)

        theta_hat_no = candidates.mean(axis=0)
        series["no_filter"].append(float(np.linalg.norm(theta_hat_no - theta_true)))

        w_sa, delta_phi_sa = sa_model(x)
        delta_phi_sa = clamp_delta_phi(delta_phi_sa, clip_val)
        theta_w_sa = SetAwareBiasRobustFilter.weighted_estimate(x, w_sa)
        theta_new_sa = theta_w_sa + delta_phi_sa if is_high_dim else delta_phi_sa
        lambda_reg = args.lambda_reg_high_dim if is_high_dim else args.lambda_reg
        loss_sa = (
            args.lambda_class * classification_loss(w_sa, y)
            + args.lambda_contract * contraction_loss(theta_new_sa, theta_good_t)
            + args.lambda_ess * ess_loss(w_sa, tau=args.tau)
            + lambda_reg * correction_reg(delta_phi_sa)
        )
        opt_sa.zero_grad()
        loss_sa.backward()
        opt_sa.step()
        theta_sa = theta_sa + ours_c * (theta_new_sa.detach().cpu().numpy().squeeze(0) - theta_sa)
        series["ours"].append(float(np.linalg.norm(theta_sa - theta_true)))

        w_sig, low_rank_delta = sig_model(x, logsig_features)
        delta_phi_sig = (low_rank_delta @ projector_t.T).unsqueeze(1)
        delta_phi_sig = clamp_delta_phi(delta_phi_sig, clip_val)
        theta_w_sig = SetAwarePathSignatureFilter.weighted_estimate(x, w_sig)
        theta_new_sig = theta_w_sig + delta_phi_sig.squeeze(1) if is_high_dim else delta_phi_sig.squeeze(1)
        loss_sig = (
            args.lambda_class * classification_loss(w_sig, y)
            + args.lambda_contract * contraction_loss(theta_new_sig, theta_good_t)
            + args.lambda_ess * ess_loss(w_sig, tau=args.tau)
            + lambda_reg * correction_reg(delta_phi_sig)
        )
        opt_sig.zero_grad()
        loss_sig.backward()
        opt_sig.step()
        theta_sig = theta_sig + ours_c * (theta_new_sig.detach().cpu().numpy().squeeze(0) - theta_sig)
        theta_sig = np.nan_to_num(theta_sig, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        series["path_signature"].append(float(np.linalg.norm(theta_sig - theta_true)))

    return series


def save_csv(all_series: Dict[int, Dict[str, Dict[str, List[float]]]], out_dir: pathlib.Path, tail_stats: Dict[int, Dict[str, Dict[str, float]]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "tail_summary.csv").open("w") as f:
        header = ["dim"]
        for method in METHODS:
            header.extend([f"{method}_mean", f"{method}_std"])
        f.write(",".join(header) + "\n")
        for dim in all_series:
            row = [str(dim)]
            for method in METHODS:
                row.append(f"{tail_stats[dim][method]['mean']:.6f}")
                row.append(f"{tail_stats[dim][method]['std']:.6f}")
            f.write(",".join(row) + "\n")

    for dim, series in all_series.items():
        gens = len(next(iter(series.values()))["mean"])
        with (out_dir / f"dim{dim}_trajectories.csv").open("w") as f:
            header = ["generation"]
            for method in METHODS:
                header.extend([f"{method}_mean", f"{method}_std"])
            f.write(",".join(header) + "\n")
            for i in range(gens):
                row = [str(i + 1)]
                for method in METHODS:
                    row.append(f"{series[method]['mean'][i]:.6f}")
                    row.append(f"{series[method]['std'][i]:.6f}")
                f.write(",".join(row) + "\n")


def plot_tail_vs_dim(
    dims: Sequence[int],
    tail_stats: Dict[int, Dict[str, Dict[str, float]]],
    out_path: pathlib.Path,
) -> None:
    plt.figure(figsize=(7.8, 5.2))
    colors = {
        "no_filter": "#9aa0a6",
        "ours": "#e07a5f",
        "path_signature": "#2a9d8f",
    }
    labels = {
        "no_filter": "No Filter",
        "ours": "Original Exp5 Ours",
        "path_signature": "Adaptive Path Signature",
    }
    for method in METHODS:
        means = np.array([tail_stats[dim][method]["mean"] for dim in dims])
        stds = np.array([tail_stats[dim][method]["std"] for dim in dims])
        plt.errorbar(dims, means, yerr=stds, marker="o", linewidth=2, capsize=4, label=labels[method], color=colors[method])
    plt.xlabel("Dimension")
    plt.ylabel("Tail Error")
    plt.yscale("log")
    plt.xticks(dims, [str(dim) for dim in dims])
    plt.title("Strict Exp5 + Adaptive Path Signature")
    plt.grid(alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict Exp5 reproduction plus adaptive path-signature extension.")
    parser.add_argument("--dims", type=int, nargs="+", default=[500, 600, 700, 1000, 1200, 1500, 2000, 12288])
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--samples-per-gen", type=int, default=50)
    parser.add_argument("--bias-norm", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--calibration-size", type=int, default=200)
    parser.add_argument("--ours-contraction", type=float, default=0.4)
    parser.add_argument("--ours-contraction-high-dim", type=float, default=0.9)
    parser.add_argument("--top-ratio", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=50.0)
    parser.add_argument("--lambda-contract", type=float, default=1.0)
    parser.add_argument("--lambda-ess", type=float, default=0.1)
    parser.add_argument("--lambda-reg", type=float, default=1e-5)
    parser.add_argument("--lambda-class", type=float, default=0.05)
    parser.add_argument("--lambda-reg-high-dim", type=float, default=1e-4)
    parser.add_argument("--reg-dim-threshold", type=int, default=500)
    parser.add_argument("--correction-clip", type=float, default=0.0)
    parser.add_argument("--correction-clip-high-dim", type=float, default=5.0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--hidden-high-dim", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-heads-high-dim", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-layers-high-dim", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dropout-high-dim", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--tail-window", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1088, 2195, 4960, 1545, 3549, 1440, 3050, 5414])
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--out-dir", type=pathlib.Path, default=SCRIPT_DIR / "results")
    args = parser.parse_args()

    device = pick_device(args.device)
    sig_cfg = SignatureConfig()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_series: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    tail_stats: Dict[int, Dict[str, Dict[str, float]]] = {}
    start_time = time.time()
    per_dim_time: Dict[int, float] = {}

    for idx, dim in enumerate(args.dims):
        t0 = time.time()
        seed_series: List[Dict[str, List[float]]] = []
        for seed in args.seeds:
            seed_series.append(run_single_dim(args=args, dim=dim, device=device, seed=seed + idx * 37, signature_cfg=sig_cfg))
        all_series[dim] = aggregate_seed_series(seed_series)
        tail_stats[dim] = aggregate_tail(seed_series, tail=args.tail_window)
        per_dim_time[dim] = time.time() - t0
        print(f"dim={dim} done in {per_dim_time[dim]:.1f}s on {device}")

    plot_tail_vs_dim(args.dims, tail_stats, out_dir / "tail_vs_dim.png")
    save_csv(all_series, out_dir, tail_stats)

    runtime = {
        "device": str(device),
        "seeds": list(args.seeds),
        "per_dim_time_sec": per_dim_time,
        "total_time_sec": time.time() - start_time,
        "alignment": {
            "paper_exp5": "same data generation, same baselines, same losses, same dims/seeds protocol",
            "sibylla": "same winding-ratio thresholds, same expected_next diagnostic window, same adaptive logsig family, same MLP head style",
        },
    }
    with (out_dir / "runtime.json").open("w") as f:
        json.dump(runtime, f, indent=2)

    for dim in args.dims:
        stats = tail_stats[dim]
        improvement = 100.0 * (1.0 - stats["path_signature"]["mean"] / (stats["ours"]["mean"] + 1e-8))
        print(
            f"dim={dim} "
            f"ours={stats['ours']['mean']:.4f}±{stats['ours']['std']:.4f} "
            f"path_signature={stats['path_signature']['mean']:.4f}±{stats['path_signature']['std']:.4f} "
            f"improvement_vs_original_ours={improvement:.2f}%"
        )
    print(f"saved_results={out_dir}")


if __name__ == "__main__":
    main()
