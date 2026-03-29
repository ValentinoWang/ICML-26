import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn, optim

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BASE_TABLES_DIR = ROOT / "Total_results" / "Tables" / SCRIPT_DIR.name
BASE_FIGURES_DIR = ROOT / "Total_results" / "Figures" / SCRIPT_DIR.name
TABLES_DIR = BASE_TABLES_DIR / "results"
FIGURES_DIR = BASE_FIGURES_DIR / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from filter.data import make_bias_vector, sample_candidates, sample_clean_reference, set_seed  # noqa: E402
from filter.losses import classification_loss, contraction_loss, correction_reg, ess_loss  # noqa: E402
from filter.set_aware.model import SetAwareBiasRobustFilter  # noqa: E402
from filter.standard.model import StandardFilter  # noqa: E402
from make_fig_bias_reduction import plot_tail_vs_dim, plot_trajectories  # noqa: E402

METHODS = ["no_filter", "mlp_filter", "mlp_correction", "ours"]
LEGEND_KWARGS = {"loc": "upper right", "bbox_to_anchor": (0.98, 0.98), "framealpha": 0.9, "borderaxespad": 0.4}


def clamp_delta_phi(delta_phi: torch.Tensor, clip: float) -> torch.Tensor:
    if clip is None or clip <= 0:
        return delta_phi
    norm = delta_phi.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    factor = torch.clamp(clip / norm, max=1.0)
    return delta_phi * factor


class CorrectionMLP(nn.Module):
    """Mean-pooling MLP that outputs a global correction (no set interaction)."""

    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)  # [B, D]
        return self.net(pooled).unsqueeze(1)  # [B, 1, D] for broadcast consistency


def build_labels(candidates: np.ndarray, theta_true: np.ndarray, top_ratio: float) -> np.ndarray:
    dists = np.linalg.norm(candidates - theta_true, axis=1)
    k = max(1, int(len(candidates) * top_ratio))
    thresh = np.partition(dists, k - 1)[k - 1]
    return (dists <= thresh).astype(np.float32)


def aggregate_seed_series(seed_series: List[Dict[str, List[float]]]) -> Dict[str, Dict[str, List[float]]]:
    stats: Dict[str, Dict[str, List[float]]] = {}
    for m in METHODS:
        arr = np.array([s[m] for s in seed_series])
        stats[m] = {"mean": arr.mean(axis=0).tolist(), "std": arr.std(axis=0).tolist()}
    return stats


def aggregate_tail(seed_series: List[Dict[str, List[float]]], tail: int) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for m in METHODS:
        tails = np.array([np.mean(s[m][-tail:]) for s in seed_series])
        stats[m] = {"mean": float(tails.mean()), "std": float(tails.std())}
    return stats


def run_single_dim(args: argparse.Namespace, dim: int, device: torch.device, seed: int) -> Dict[str, List[float]]:
    rng = set_seed(seed)
    bias_vec = make_bias_vector(dim, args.bias_norm)

    is_high_dim = dim >= args.reg_dim_threshold
    hidden_sa = args.hidden_high_dim if is_high_dim else args.hidden
    n_heads_sa = args.n_heads_high_dim if is_high_dim else args.n_heads
    n_layers_sa = args.n_layers_high_dim if is_high_dim else args.n_layers
    dropout_sa = args.dropout_high_dim if is_high_dim else args.dropout
    clip_val = args.correction_clip_high_dim if is_high_dim else args.correction_clip
    ours_c = args.ours_contraction_high_dim if is_high_dim else args.ours_contraction

    std_model = StandardFilter(dim=dim, hidden=args.hidden, dropout=args.dropout).to(device)
    corr_hidden = hidden_sa if is_high_dim else args.hidden
    corr_dropout = dropout_sa if is_high_dim else args.dropout
    corr_mlp = CorrectionMLP(dim=dim, hidden=corr_hidden, dropout=corr_dropout).to(device)
    sa_model = SetAwareBiasRobustFilter(
        dim=dim,
        hidden=hidden_sa,
        n_heads=n_heads_sa,
        n_layers=n_layers_sa,
        dropout=dropout_sa,
    ).to(device)
    opt_std = optim.Adam(std_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_corr = optim.Adam(corr_mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_sa = optim.Adam(sa_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    theta_true = np.ones(dim)
    theta_good = sample_clean_reference(rng, theta_true, n=args.calibration_size, noise_std=args.noise_std)
    theta_good_t = torch.from_numpy(theta_good[None, ...]).float().to(device)
    theta_std = theta_good.copy()
    theta_mlp_corr = theta_good.copy()
    theta_sa = theta_good.copy()

    series: Dict[str, List[float]] = {k: [] for k in METHODS}

    for _ in range(args.generations):
        candidates = sample_candidates(
            rng,
            theta_true=theta_true,
            bias=bias_vec,
            n=args.samples_per_gen,
            noise_std=args.noise_std,
        )
        labels = build_labels(candidates, theta_true, top_ratio=args.top_ratio)
        x = torch.from_numpy(candidates[None, ...]).float().to(device)
        y = torch.from_numpy(labels[None, ...]).float().to(device)
        # No filter baseline: raw mean
        theta_hat_no = candidates.mean(axis=0)
        series["no_filter"].append(float(np.linalg.norm(theta_hat_no - theta_true)))
        x_sa = x
        theta_good_t_sa = theta_good_t

        # MLP filter (standard)
        w_std = std_model(x)
        theta_w_std = StandardFilter.weighted_estimate(x, w_std)
        loss_std = (
            classification_loss(w_std, y)
            + args.lambda_contract * contraction_loss(theta_w_std, theta_good_t)
            + args.lambda_ess * ess_loss(w_std, tau=args.tau)
        )
        opt_std.zero_grad()
        loss_std.backward()
        opt_std.step()
        theta_std = theta_std + args.standard_contraction * (
            theta_w_std.detach().cpu().numpy().squeeze(0) - theta_std
        )
        series["mlp_filter"].append(float(np.linalg.norm(theta_std - theta_true)))

        # MLP + correction (no set interaction, weights=1)
        delta_phi_mlp = corr_mlp(x)
        delta_phi_mlp = clamp_delta_phi(delta_phi_mlp, clip_val)
        theta_new_mlp = delta_phi_mlp
        lambda_reg_eff = args.lambda_reg_high_dim if is_high_dim else args.lambda_reg
        loss_mlp = args.lambda_contract * contraction_loss(theta_new_mlp, theta_good_t) + lambda_reg_eff * correction_reg(delta_phi_mlp)
        opt_corr.zero_grad()
        loss_mlp.backward()
        opt_corr.step()
        theta_mlp_corr = theta_mlp_corr + ours_c * (
            theta_new_mlp.detach().cpu().numpy().squeeze(0) - theta_mlp_corr
        )
        series["mlp_correction"].append(float(np.linalg.norm(theta_mlp_corr - theta_true)))

        # Set-aware filter with auxiliary weighting head; estimation uses weighted mean + correction in high-d.
        w_sa, delta_phi = sa_model(x_sa)
        delta_phi = clamp_delta_phi(delta_phi, clip_val)
        theta_w_sa = SetAwareBiasRobustFilter.weighted_estimate(x, w_sa)
        theta_new_sa = theta_w_sa + delta_phi if is_high_dim else delta_phi
        lambda_reg = args.lambda_reg_high_dim if is_high_dim else args.lambda_reg
        loss_sa = (
            args.lambda_class * classification_loss(w_sa, y)
            + args.lambda_contract * contraction_loss(theta_new_sa, theta_good_t)
            + args.lambda_ess * ess_loss(w_sa, tau=args.tau)
            + lambda_reg * correction_reg(delta_phi)
        )
        opt_sa.zero_grad()
        loss_sa.backward()
        opt_sa.step()
        delta_full = theta_new_sa.detach().cpu().numpy().squeeze(0)
        theta_sa = theta_sa + ours_c * (delta_full - theta_sa)
        series["ours"].append(float(np.linalg.norm(theta_sa - theta_true)))

    return series


def save_csv(all_series: Dict[int, Dict[str, Dict[str, List[float]]]], out_dir: pathlib.Path, tail_stats: Dict[int, Dict[str, Dict[str, float]]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Tail summary
    with (out_dir / "exp5_tail_summary.csv").open("w") as f:
        header = ["dim"]
        for m in METHODS:
            header.extend([f"{m}_mean", f"{m}_std"])
        f.write(",".join(header) + "\n")
        for d in all_series:
            row = [str(d)]
            for m in METHODS:
                row.append(f"{tail_stats[d][m]['mean']:.6f}")
                row.append(f"{tail_stats[d][m]['std']:.6f}")
            f.write(",".join(row) + "\n")
    # Per-dim trajectories
    for d, series in all_series.items():
        gens = len(next(iter(series.values()))["mean"])
        with (out_dir / f"exp5_dim{d}_trajectories.csv").open("w") as f:
            header = ["generation"]
            for m in METHODS:
                header.extend([f"{m}_mean", f"{m}_std"])
            f.write(",".join(header) + "\n")
            for i in range(gens):
                row = [str(i + 1)]
                for m in METHODS:
                    row.append(f"{series[m]['mean'][i]:.6f}")
                    row.append(f"{series[m]['std'][i]:.6f}")
                f.write(",".join(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Exp 5: High-dimensional robustness of set-aware filter.")
    parser.add_argument("--dims", type=int, nargs="+", default=[50, 100, 500])
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--samples-per-gen", type=int, default=50, help="High-dim low-sample regime.")
    parser.add_argument("--bias-norm", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--calibration-size", type=int, default=200)
    parser.add_argument("--standard-contraction", type=float, default=0.2)
    parser.add_argument("--ours-contraction", type=float, default=0.4)
    parser.add_argument("--ours-contraction-high-dim", type=float, default=0.9)
    parser.add_argument("--top-ratio", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=50.0)
    parser.add_argument("--lambda-contract", type=float, default=1.0)
    parser.add_argument("--lambda-ess", type=float, default=0.1)
    parser.add_argument("--lambda-reg", type=float, default=1e-5)
    parser.add_argument("--lambda-class", type=float, default=0.05)
    parser.add_argument(
        "--lambda-reg-high-dim",
        type=float,
        default=1e-4,
        help="Stronger correction L2 for high-d dims (>= reg_dim_threshold) to prevent Δϕ from drifting.",
    )
    parser.add_argument(
        "--reg-dim-threshold",
        type=int,
        default=500,
        help="Apply lambda_reg_high_dim when dim >= this threshold.",
    )
    parser.add_argument("--correction-clip", type=float, default=0.0, help="Clip ||delta_phi||; 0 disables.")
    parser.add_argument(
        "--correction-clip-high-dim",
        type=float,
        default=5.0,
        help="Clip ||delta_phi|| for high dimensions (>= reg_dim_threshold).",
    )
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--hidden-high-dim", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-heads-high-dim", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-layers-high-dim", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dropout-high-dim", type=float, default=0.1)
    parser.add_argument(
        "--use-pca-high-dim",
        action="store_true",
        default=True,
        help="Enable PCA preprocessing for high-d regimes.",
    )
    parser.add_argument("--pca-dim", type=int, default=20, help="PCA dimension when use_pca_high_dim is enabled.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--tail-window", type=int, default=50)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1088, 2195, 4960, 1545, 3549, 1440, 3050, 5414],
        help="Seeds for mean/std/CI.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out-dir", type=pathlib.Path, default=TABLES_DIR, help="Directory to store CSV/JSON outputs.")
    parser.add_argument("--fig-dir", type=pathlib.Path, default=FIGURES_DIR, help="Directory to store figures.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    seeds: Sequence[int] = args.seeds if args.seeds is not None else [args.seed]

    out_dir: pathlib.Path = args.out_dir
    fig_dir: pathlib.Path = args.fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_series: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    tail_stats: Dict[int, Dict[str, Dict[str, float]]] = {}
    start_time = time.time()
    per_dim_time: Dict[int, float] = {}

    for idx, d in enumerate(args.dims):
        t0 = time.time()
        seed_series: List[Dict[str, List[float]]] = []
        for s in seeds:
            seed_series.append(
                run_single_dim(
                    args=args,
                    dim=d,
                    device=device,
                    seed=s + idx * 37,
                )
            )
        stats = aggregate_seed_series(seed_series)
        all_series[d] = stats
        tail_stats[d] = aggregate_tail(seed_series, tail=args.tail_window)
        per_dim_time[d] = time.time() - t0
        print(f"Dim {d} done in {per_dim_time[d]:.1f}s on {device}")

    plot_trajectories(all_series, fig_dir / "exp5_trajs_by_dim.png", n_seeds=len(seeds), methods=METHODS, legend_kwargs=LEGEND_KWARGS)
    plot_tail_vs_dim(args.dims, tail_stats, fig_dir / "exp5_tail_vs_dim.png", n_seeds=len(seeds), methods=METHODS)
    save_csv(all_series, out_dir, tail_stats)

    runtime = {
        "device": str(device),
        "seeds": list(seeds),
        "per_dim_time_sec": per_dim_time,
        "total_time_sec": time.time() - start_time,
        "note": "High-dim low-sample regime (n=50), bias_norm=0.5, dims in {50,100,500}",
    }
    with (out_dir / "runtime_exp5.json").open("w") as f:
        json.dump(runtime, f, indent=2)
    print(f"Saved Exp 5 results to {out_dir} (tables) and {fig_dir} (figures)")


if __name__ == "__main__":
    main()
