import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from filter.data import make_bias_vector, sample_candidates, sample_clean_reference, set_seed
from filter.losses import classification_loss, contraction_loss, correction_reg, ess_loss
from filter.set_aware.model import SetAwareBiasRobustFilter


@dataclass
class Metrics:
    no_filter: List[float]
    standard_filter: List[float]
    ours: List[float]
    delta_phi_norm: List[float]


def build_labels(candidates: np.ndarray, theta_true: np.ndarray, top_ratio: float) -> np.ndarray:
    """
    Label top k% closest points to theta_true as positives.
    """
    dists = np.linalg.norm(candidates - theta_true, axis=1)
    k = max(1, int(len(candidates) * top_ratio))
    thresh = np.partition(dists, k - 1)[k - 1]
    labels = (dists <= thresh).astype(np.float32)
    return labels


def plot_curves(metrics: Metrics, out_path: Path) -> None:
    plt.figure(figsize=(8, 4.2))
    plt.plot(metrics.no_filter, label="No Filter", linewidth=2)
    plt.plot(metrics.standard_filter, label="Standard Filter", linewidth=2)
    plt.plot(metrics.ours, label="Ours", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel(r"$\|\theta_t - \theta^*\|_2$")
    plt.title("Set-Aware Bias-Robust Filter")
    plt.legend()
    plt.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_experiment(args: argparse.Namespace) -> Metrics:
    rng = set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    theta_true = rng.normal(size=args.dim)
    bias = make_bias_vector(args.dim, args.bias_norm)
    theta_good = sample_clean_reference(rng, theta_true, n=args.calibration_size, noise_std=args.noise_std)

    model = SetAwareBiasRobustFilter(
        dim=args.dim,
        hidden=args.hidden,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Baseline states
    theta_standard = theta_good.copy()
    theta_ours = theta_good.copy()

    metrics = Metrics(no_filter=[], standard_filter=[], ours=[], delta_phi_norm=[])

    for _ in range(args.generations):
        candidates = sample_candidates(
            rng,
            theta_true=theta_true,
            bias=bias,
            n=args.samples_per_gen,
            noise_std=args.noise_std,
        )
        labels = build_labels(candidates, theta_true, top_ratio=args.top_ratio)

        # Baseline estimators
        theta_hat = candidates.mean(axis=0)  # biased estimator
        theta_standard = theta_standard + args.standard_contraction * (theta_hat - theta_standard)

        # Forward pass
        x = torch.from_numpy(candidates[None, ...]).float().to(device)
        y = torch.from_numpy(labels[None, ...]).float().to(device)

        weights, delta_phi = model(x)
        theta_weighted = model.weighted_estimate(x, weights)
        theta_new = theta_weighted + delta_phi

        # Losses
        l_class = classification_loss(weights, y)
        l_contract = contraction_loss(theta_new, torch.from_numpy(theta_good[None, ...]).float().to(device))
        l_ess = ess_loss(weights, tau=args.tau)
        l_reg = correction_reg(delta_phi)

        loss = (
            l_class
            + args.lambda_contract * l_contract
            + args.lambda_ess * l_ess
            + args.lambda_reg * l_reg
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update ours estimator (optionally contracted to smooth dynamics).
        theta_new_np = theta_new.detach().cpu().numpy().squeeze(0)
        theta_ours = theta_ours + args.ours_contraction * (theta_new_np - theta_ours)

        # Track metrics
        metrics.no_filter.append(float(np.linalg.norm(theta_hat - theta_true)))
        metrics.standard_filter.append(float(np.linalg.norm(theta_standard - theta_true)))
        metrics.ours.append(float(np.linalg.norm(theta_ours - theta_true)))
        metrics.delta_phi_norm.append(float(np.linalg.norm(delta_phi.detach().cpu().numpy().squeeze(0))))

    return metrics


def save_metrics(metrics: Metrics, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "set_filter_metrics.csv"
    with csv_path.open("w") as f:
        header = ["generation", "no_filter", "standard_filter", "ours", "delta_phi_norm"]
        f.write(",".join(header) + "\n")
        for i in range(len(metrics.no_filter)):
            row = [
                str(i + 1),
                f"{metrics.no_filter[i]:.6f}",
                f"{metrics.standard_filter[i]:.6f}",
                f"{metrics.ours[i]:.6f}",
                f"{metrics.delta_phi_norm[i]:.6f}",
            ]
            f.write(",".join(row) + "\n")


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Set-Aware Bias-Robust Neural Filter experiment.")
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--samples-per-gen", type=int, default=200)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--bias-norm", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--calibration-size", type=int, default=120)
    parser.add_argument("--standard-contraction", type=float, default=0.5)
    parser.add_argument("--ours-contraction", type=float, default=0.35)
    parser.add_argument("--top-ratio", type=float, default=0.3)
    parser.add_argument("--tau", type=float, default=50.0)
    parser.add_argument("--lambda-contract", type=float, default=1.0)
    parser.add_argument("--lambda-ess", type=float, default=0.1)
    parser.add_argument("--lambda-reg", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=script_dir / "results",
        help="Directory to store plots and CSV.",
    )
    args = parser.parse_args()

    metrics = run_experiment(args)
    save_metrics(metrics, args.out_dir)
    plot_curves(metrics, args.out_dir / "set_filter_plot.png")

    print(f"Saved metrics to {args.out_dir}")
    print(f"Final errors -> No Filter: {metrics.no_filter[-1]:.4f}, "
          f"Standard: {metrics.standard_filter[-1]:.4f}, Ours: {metrics.ours[-1]:.4f}")


if __name__ == "__main__":
    main()
