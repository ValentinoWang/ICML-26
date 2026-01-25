#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Synthetic (v2) toy experiment:
- Baseline: update with fixed biased data stream, no constraints.
- Ours: add L_bias anchor + simple distance-based filter to suppress the biased source.
Generates the "money plot" showing drift vs. controllable convergence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib
import numpy as np

# Force non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class ExperimentConfig:
    steps: int = 300
    batch_size: int = 64
    bad_ratio: float = 0.7
    lr: float = 0.1
    data_std: float = 0.1
    lambda_bias: float = 2.0
    filter_strength: float = 2.0
    seed: int = 2025
    theta_init_x: float = 0.2
    theta_init_y: float = -0.2
    theta_good_x: float = 0.0
    theta_good_y: float = 0.0
    beta_drift_x: float = 2.0
    beta_drift_y: float = 2.0
    save_dir: Path = Path(__file__).resolve().parent / "results"

    @property
    def theta_good(self) -> np.ndarray:
        return np.array([self.theta_good_x, self.theta_good_y], dtype=float)

    @property
    def beta_drift(self) -> np.ndarray:
        return np.array([self.beta_drift_x, self.beta_drift_y], dtype=float)

    @property
    def theta_init(self) -> np.ndarray:
        return np.array([self.theta_init_x, self.theta_init_y], dtype=float)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Synthetic drift toy experiment (v2 baseline vs ours)")
    parser.add_argument("--steps", type=int, default=300, help="Number of SGD steps to simulate")
    parser.add_argument("--batch-size", type=int, default=64, help="Samples per iteration")
    parser.add_argument("--bad-ratio", type=float, default=0.7, help="Proportion of biased samples in each batch")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for the toy optimizer")
    parser.add_argument("--data-std", type=float, default=0.1, help="Gaussian noise std around each source")
    parser.add_argument("--lambda-bias", type=float, default=2.0, help="Weight for L_bias anchor term")
    parser.add_argument(
        "--filter-strength",
        type=float,
        default=2.0,
        help="Exponential distance filter strength (higher = aggressively down-weight biased samples)",
    )
    parser.add_argument("--seed", type=int, default=2025, help="RNG seed")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory to store figures and arrays")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        bad_ratio=args.bad_ratio,
        lr=args.lr,
        data_std=args.data_std,
        lambda_bias=args.lambda_bias,
        filter_strength=args.filter_strength,
        seed=args.seed,
    )
    if args.save_dir is not None:
        cfg.save_dir = Path(args.save_dir)
    return cfg


def simulate(cfg: ExperimentConfig, use_bias_control: bool) -> Dict[str, np.ndarray]:
    """
    Run the toy SGD loop.

    Baseline: use_bias_control=False (no anchor, no filter).
    Ours: use_bias_control=True (L_bias anchor + distance-based reweight).
    """

    rng = np.random.default_rng(cfg.seed)
    theta = cfg.theta_init.copy()
    theta_good = cfg.theta_good
    bias_source = theta_good + cfg.beta_drift

    traj = [theta.copy()]
    dist_to_good = [np.linalg.norm(theta - theta_good)]
    dist_to_bad = [np.linalg.norm(theta - bias_source)]

    for _ in range(cfg.steps):
        n_bad = int(cfg.batch_size * cfg.bad_ratio)
        n_good = cfg.batch_size - n_bad

        good_batch = rng.normal(loc=theta_good, scale=cfg.data_std, size=(n_good, theta_good.shape[0]))
        bad_batch = rng.normal(loc=bias_source, scale=cfg.data_std, size=(n_bad, theta_good.shape[0]))
        batch = np.vstack([good_batch, bad_batch])

        if use_bias_control:
            distances = np.linalg.norm(batch - theta_good, axis=1)
            weights = np.exp(-cfg.filter_strength * distances)
            weights = weights / weights.mean()
        else:
            weights = np.ones(len(batch), dtype=float)

        grad_data = ((theta - batch) * weights[:, None]).mean(axis=0)
        grad = grad_data
        if use_bias_control:
            grad = grad_data + cfg.lambda_bias * (theta - theta_good)

        theta = theta - cfg.lr * grad

        traj.append(theta.copy())
        dist_to_good.append(np.linalg.norm(theta - theta_good))
        dist_to_bad.append(np.linalg.norm(theta - bias_source))

    return {
        "theta_traj": np.stack(traj),
        "dist_to_good": np.array(dist_to_good),
        "dist_to_bad": np.array(dist_to_bad),
    }


def plot_money_figure(
    baseline: Dict[str, np.ndarray],
    ours: Dict[str, np.ndarray],
    cfg: ExperimentConfig,
    save_path: Path,
) -> None:
    theta_good = cfg.theta_good
    bias_source = theta_good + cfg.beta_drift

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Trajectory in parameter space
    ax0 = axes[0]
    ax0.plot(
        baseline["theta_traj"][:, 0],
        baseline["theta_traj"][:, 1],
        color="tomato",
        lw=2.0,
        label="Baseline (no control)",
    )
    ax0.plot(
        ours["theta_traj"][:, 0],
        ours["theta_traj"][:, 1],
        color="royalblue",
        lw=2.0,
        label="Ours (L_bias + filter)",
    )
    ax0.scatter(theta_good[0], theta_good[1], color="gold", marker="*", s=140, label="theta_good")
    ax0.scatter(
        bias_source[0],
        bias_source[1],
        color="black",
        marker="X",
        s=70,
        label="theta_good + beta_drift",
    )
    ax0.set_title("Trajectory in parameter space")
    ax0.set_xlabel("theta[0]")
    ax0.set_ylabel("theta[1]")
    ax0.grid(alpha=0.3)
    ax0.legend()

    # Distance to theta_good over time
    ax1 = axes[1]
    steps = np.arange(len(baseline["dist_to_good"]))
    ax1.plot(steps, baseline["dist_to_good"], color="tomato", lw=2.0, label="Baseline -> theta_good")
    ax1.plot(steps, ours["dist_to_good"], color="royalblue", lw=2.0, label="Ours -> theta_good")
    ax1.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Distance to theta_good")
    ax1.set_title("Figure 1: drift vs controllable convergence")
    ax1.grid(alpha=0.3)
    ax1.legend()

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def main() -> None:
    cfg = parse_args()
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    print("Synthetic toy experiment (v2)")
    print(f"   theta_good: [{cfg.theta_good_x}, {cfg.theta_good_y}]")
    print(f"   beta_drift: [{cfg.beta_drift_x}, {cfg.beta_drift_y}] (fixed biased source)")
    print(f"   steps: {cfg.steps}, batch_size: {cfg.batch_size}, bad_ratio: {cfg.bad_ratio}")
    print(f"   lr: {cfg.lr}, data_std: {cfg.data_std}")
    print(f"   lambda_bias: {cfg.lambda_bias}, filter_strength: {cfg.filter_strength}")
    print(f"   seed: {cfg.seed}")

    baseline = simulate(cfg, use_bias_control=False)
    ours = simulate(cfg, use_bias_control=True)

    fig_path = cfg.save_dir / "figure1_money_plot.png"
    plot_money_figure(baseline, ours, cfg, fig_path)

    np.savez(
        cfg.save_dir / "synthetic_v2_results.npz",
        baseline=baseline["theta_traj"],
        baseline_dist=baseline["dist_to_good"],
        ours=ours["theta_traj"],
        ours_dist=ours["dist_to_good"],
        theta_good=cfg.theta_good,
        bias_source=cfg.theta_good + cfg.beta_drift,
        cfg=dict(
            steps=cfg.steps,
            batch_size=cfg.batch_size,
            bad_ratio=cfg.bad_ratio,
            lr=cfg.lr,
            data_std=cfg.data_std,
            lambda_bias=cfg.lambda_bias,
            filter_strength=cfg.filter_strength,
            seed=cfg.seed,
        ),
    )

    print("Done. Figure saved to:", fig_path)
    print(
        f"   Final dist (baseline -> theta_good): {baseline['dist_to_good'][-1]:.3f}; "
        f"Ours: {ours['dist_to_good'][-1]:.3f}"
    )
    print(
        f"   Final dist (baseline -> bias source): {baseline['dist_to_bad'][-1]:.3f}; "
        f"Ours: {ours['dist_to_bad'][-1]:.3f}"
    )


if __name__ == "__main__":
    main()
