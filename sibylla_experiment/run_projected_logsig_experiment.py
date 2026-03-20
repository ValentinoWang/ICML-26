from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sibylla_experiment_mpl")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim


@dataclass(frozen=True)
class ExperimentConfig:
    history_len: int = 7
    latent_dim: int = 12
    proj_dim: int = 3
    max_order: int = 3
    output_dim: int = 2

    @property
    def window_len(self) -> int:
        return self.history_len + 1

    @property
    def max_logsig_dim(self) -> int:
        return logsig_dim(self.max_order, self.proj_dim)


@dataclass
class ProjectionBundle:
    projector: torch.Tensor
    null_basis: torch.Tensor


@dataclass
class Dataset:
    histories: torch.Tensor
    diagnostic_windows: torch.Tensor
    targets: torch.Tensor
    family_ids: np.ndarray
    winding_ratio: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-8)


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


def batch_kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bi,bj->bij", a, b).reshape(a.shape[0], -1)


def logsig_dim(order: int, proj_dim: int) -> int:
    return sum(proj_dim**k for k in range(1, order + 1))


def truncated_signature_levels(path: torch.Tensor, order: int) -> list[torch.Tensor]:
    if order > 3:
        raise ValueError("This prototype only supports signature truncation up to order 3.")

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
        raise ValueError("This prototype only supports log-signature orders 1..3.")

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


def winding_ratio_torch(path: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    chord = torch.linalg.vector_norm(path[:, -1, :] - path[:, 0, :], dim=1)
    arc = torch.linalg.vector_norm(path[:, 1:, :] - path[:, :-1, :], dim=2).sum(dim=1)
    return arc / (chord + eps)


def sibylla_order(ratio: np.ndarray) -> np.ndarray:
    order = np.ones_like(ratio, dtype=np.int64)
    order[ratio >= 1.1] = 2
    order[ratio >= 1.5] = 3
    return order


def make_projection_bundle(cfg: ExperimentConfig, seed: int) -> ProjectionBundle:
    rng = np.random.default_rng(seed)
    full_basis, _ = np.linalg.qr(rng.normal(size=(cfg.latent_dim, cfg.latent_dim)))
    projector = torch.from_numpy(full_basis[:, : cfg.proj_dim].astype(np.float32))
    null_basis = torch.from_numpy(full_basis[:, cfg.proj_dim :].astype(np.float32))
    return ProjectionBundle(projector=projector, null_basis=null_basis)


def project_histories(histories: torch.Tensor, bundle: ProjectionBundle) -> torch.Tensor:
    return histories @ bundle.projector.to(histories.dtype)


def expected_next_np(history: np.ndarray) -> np.ndarray:
    velocity_last = history[-1] - history[-2]
    velocity_prev = history[-2] - history[-3]
    acceleration = velocity_last - velocity_prev
    return history[-1] + 0.85 * velocity_last + 0.35 * acceleration


def expected_next_torch(history: torch.Tensor) -> torch.Tensor:
    velocity_last = history[:, -1, :] - history[:, -2, :]
    velocity_prev = history[:, -2, :] - history[:, -3, :]
    acceleration = velocity_last - velocity_prev
    return history[:, -1, :] + 0.85 * velocity_last + 0.35 * acceleration


def make_diagnostic_windows(histories: torch.Tensor, bundle: ProjectionBundle) -> torch.Tensor:
    projected_history = project_histories(histories, bundle)
    expected = expected_next_torch(projected_history)
    return torch.cat([projected_history, expected.unsqueeze(1)], dim=1)


def make_straight_history(rng: np.random.Generator, cfg: ExperimentConfig) -> np.ndarray:
    start = rng.uniform(-0.9, 0.9, size=cfg.proj_dim)
    direction, perp, axial = random_orthonormal_frame(rng, cfg.proj_dim, 3)
    distance = rng.uniform(1.0, 1.8)
    t = np.linspace(0.0, 1.0, cfg.history_len)
    path = start + distance * t[:, None] * direction[None, :]
    path += rng.normal(scale=0.02, size=(cfg.history_len, 1)) * perp[None, :]
    path += (0.03 * np.sin(np.pi * t))[:, None] * axial[None, :]
    return path.astype(np.float32)


def make_curve_history(rng: np.random.Generator, cfg: ExperimentConfig) -> np.ndarray:
    start = rng.uniform(-0.8, 0.8, size=cfg.proj_dim)
    direction, perp, axial = random_orthonormal_frame(rng, cfg.proj_dim, 3)
    end = start + rng.uniform(1.2, 1.8) * direction + rng.uniform(-0.18, 0.18) * axial
    control = 0.5 * (start + end) + rng.uniform(0.55, 0.95) * rng.choice([-1.0, 1.0]) * perp
    control += rng.uniform(-0.25, 0.25) * axial
    t = np.linspace(0.0, 1.0, cfg.history_len)
    path = (
        (1.0 - t)[:, None] ** 2 * start
        + 2.0 * (1.0 - t)[:, None] * t[:, None] * control
        + t[:, None] ** 2 * end
    )
    return path.astype(np.float32)


def make_spiral_history(rng: np.random.Generator, cfg: ExperimentConfig) -> np.ndarray:
    center = rng.uniform(-0.25, 0.25, size=cfg.proj_dim)
    axis_u, axis_v, axis_w = random_orthonormal_frame(rng, cfg.proj_dim, 3)
    radius0 = rng.uniform(0.95, 1.25)
    radius1 = rng.uniform(0.18, 0.35)
    turns = rng.uniform(1.45 * np.pi, 1.95 * np.pi)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    axial_drift = rng.uniform(-0.45, 0.45)
    t = np.linspace(0.0, 1.0, cfg.history_len)
    radius = (1.0 - t) * radius0 + t * radius1
    angle = phase + turns * t
    path = (
        center[None, :]
        + radius[:, None] * np.cos(angle)[:, None] * axis_u[None, :]
        + radius[:, None] * np.sin(angle)[:, None] * axis_v[None, :]
        + axial_drift * t[:, None] * axis_w[None, :]
    )
    return path.astype(np.float32)


def lift_to_latent_history(path: np.ndarray, bundle: ProjectionBundle, rng: np.random.Generator) -> np.ndarray:
    projector = bundle.projector.numpy()
    null_basis = bundle.null_basis.numpy()
    latent = path @ projector.T
    if null_basis.shape[1] == 0:
        return latent.astype(np.float32)

    nuisance = rng.normal(scale=0.045, size=(path.shape[0], null_basis.shape[1]))
    nuisance = np.cumsum(nuisance, axis=0) * 0.15
    latent = latent + nuisance @ null_basis.T
    return latent.astype(np.float32)


def sample_family_history(rng: np.random.Generator, family: int, cfg: ExperimentConfig) -> tuple[np.ndarray, np.ndarray, float]:
    for _ in range(300):
        if family == 0:
            projected_history = make_straight_history(rng, cfg)
        elif family == 1:
            projected_history = make_curve_history(rng, cfg)
        else:
            projected_history = make_spiral_history(rng, cfg)

        diagnostic_window = np.concatenate([projected_history, expected_next_np(projected_history)[None, :]], axis=0)
        ratio = winding_ratio_np(diagnostic_window)

        if family == 0 and ratio < 1.1:
            return projected_history, diagnostic_window, ratio
        if family == 1 and 1.1 <= ratio < 1.5:
            return projected_history, diagnostic_window, ratio
        if family == 2 and ratio >= 1.5:
            return projected_history, diagnostic_window, ratio

    raise RuntimeError(f"Failed to sample valid family {family} history.")


def make_target_weights(cfg: ExperimentConfig, seed: int = 123) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(scale=0.75, size=(cfg.proj_dim, cfg.output_dim)).astype(np.float32)
    w2 = rng.normal(scale=0.52, size=(cfg.proj_dim**2, cfg.output_dim)).astype(np.float32)
    w3 = rng.normal(scale=0.30, size=(cfg.proj_dim**3, cfg.output_dim)).astype(np.float32)
    return w1, w2, w3


def build_dataset(rng: np.random.Generator, n_per_family: int, cfg: ExperimentConfig, bundle: ProjectionBundle) -> Dataset:
    histories = []
    windows = []
    family_ids = []
    ratios = []
    w1, w2, w3 = make_target_weights(cfg)

    for family in range(3):
        for _ in range(n_per_family):
            projected_history, diagnostic_window, ratio = sample_family_history(rng, family, cfg)
            histories.append(lift_to_latent_history(projected_history, bundle, rng))
            windows.append(diagnostic_window)
            family_ids.append(family)
            ratios.append(ratio)

    history_tensor = torch.from_numpy(np.stack(histories))
    window_tensor = torch.from_numpy(np.stack(windows))

    logsig1 = truncated_logsignature(window_tensor, 1).numpy()
    logsig2 = truncated_logsignature(window_tensor, 2).numpy()
    logsig3 = truncated_logsignature(window_tensor, 3).numpy()
    logsig2_block = logsig2[:, logsig_dim(1, cfg.proj_dim) :]
    logsig3_block = logsig3[:, logsig_dim(2, cfg.proj_dim) :]

    targets = []
    for i, family in enumerate(family_ids):
        target = logsig1[i] @ w1
        if family >= 1:
            target = target + logsig2_block[i] @ w2
        if family >= 2:
            target = target + logsig3_block[i] @ w3
        target = target + rng.normal(scale=0.02, size=cfg.output_dim)
        targets.append(target.astype(np.float32))

    return Dataset(
        histories=history_tensor,
        diagnostic_windows=window_tensor,
        targets=torch.from_numpy(np.stack(targets)),
        family_ids=np.array(family_ids, dtype=np.int64),
        winding_ratio=np.array(ratios, dtype=np.float32),
    )


def make_padded_features(histories: torch.Tensor, orders: np.ndarray, cfg: ExperimentConfig, bundle: ProjectionBundle) -> torch.Tensor:
    windows = make_diagnostic_windows(histories, bundle)
    features = torch.zeros(histories.shape[0], cfg.max_logsig_dim, dtype=histories.dtype)
    for order in range(1, cfg.max_order + 1):
        idx = np.where(orders == order)[0]
        if idx.size == 0:
            continue
        idx_t = torch.from_numpy(idx).long()
        block = truncated_logsignature(windows.index_select(0, idx_t), order)
        features[idx_t, : block.shape[1]] = block
    return features


class MLP(nn.Module):
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


def batches(n: int, batch_size: int):
    for idx in torch.randperm(n).split(batch_size):
        yield idx


def train_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device, epochs: int, batch_size: int, lr: float) -> None:
    opt = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for idx in batches(x.shape[0], batch_size):
            pred = model(x[idx].to(device))
            target = y[idx].to(device)
            loss = nn.functional.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def evaluate_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor, family_ids: np.ndarray, device: torch.device) -> tuple[float, dict[int, float]]:
    model.eval()
    pred = model(x.to(device)).cpu().numpy()
    target = y.numpy()
    rmse = float(np.sqrt(np.mean(np.sum((pred - target) ** 2, axis=1))))
    per_family = {}
    for family in range(3):
        mask = family_ids == family
        per_family[family] = float(np.sqrt(np.mean(np.sum((pred[mask] - target[mask]) ** 2, axis=1))))
    return rmse, per_family


def family_name(fid: int) -> str:
    return ["straight", "curve", "spiral"][fid]


def model_display_name(name: str) -> str:
    return {
        "fixed_m1": "fixed-1",
        "fixed_m2": "fixed-2",
        "fixed_m3": "fixed-3",
        "adaptive_sibylla": "adaptive",
    }[name]


def plot_results(out_dir: Path, rows: list[dict[str, str]], adaptive_orders: np.ndarray, adaptive_ratios: np.ndarray) -> Path:
    labels = [model_display_name(r["model"]) for r in rows]
    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.8))

    axes[0].bar(
        labels,
        [float(r["overall_rmse"]) for r in rows],
        color=["#e76f51", "#4c78a8", "#72b7b2", "#2a9d8f"],
    )
    axes[0].set_title("Overall RMSE")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(axis="y", alpha=0.25)

    order_bins = [np.mean(adaptive_orders == 1), np.mean(adaptive_orders == 2), np.mean(adaptive_orders == 3)]
    axes[1].bar(["m=1", "m=2", "m=3"], order_bins, color=["#e9c46a", "#4c78a8", "#2a9d8f"])
    axes[1].set_title("Sibylla Order Usage")
    axes[1].set_ylabel("fraction")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].scatter(np.arange(adaptive_ratios.shape[0]), adaptive_ratios, s=11, c=adaptive_orders, cmap="viridis")
    axes[2].axhline(1.1, color="#f58518", linestyle="--", linewidth=1)
    axes[2].axhline(1.5, color="#e45756", linestyle="--", linewidth=1)
    axes[2].set_title("Winding Ratio Thresholds")
    axes[2].set_xlabel("sample")
    axes[2].set_ylabel("R")
    axes[2].grid(alpha=0.2)

    dims = [float(r["avg_feature_dim"]) for r in rows]
    axes[3].bar(labels, dims, color=["#e76f51", "#4c78a8", "#72b7b2", "#2a9d8f"])
    ax2 = axes[3].twinx()
    ax2.plot(
        labels,
        [float(r["feature_build_sec"]) for r in rows],
        marker="o",
        color="#8c564b",
        linewidth=2,
    )
    axes[3].set_title("Compute Cost")
    axes[3].set_ylabel("avg active dim")
    ax2.set_ylabel("build sec")
    axes[3].grid(axis="y", alpha=0.25)

    fig.suptitle("New Experiment: Sibylla Projected Log-Signature Forecast", fontsize=14, y=1.04)
    for ax in axes[[0, 3]]:
        ax.tick_params(axis="x", rotation=12)
    fig.tight_layout(pad=1.2, w_pad=2.4)
    path = out_dir / "projected_logsig_summary.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def validate_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.history_len < 4:
        raise ValueError("history_len must be at least 4 so the one-step forecast has curvature information.")
    if args.proj_dim < 3:
        raise ValueError("proj_dim must be at least 3 for the straight/curve/spiral projected families.")
    if args.latent_dim < args.proj_dim:
        raise ValueError("latent_dim must be greater than or equal to proj_dim.")
    if args.max_order != 3:
        raise ValueError("This single-file Sibylla experiment is intentionally fixed at max_order=3.")
    return ExperimentConfig(
        history_len=args.history_len,
        latent_dim=args.latent_dim,
        proj_dim=args.proj_dim,
        max_order=args.max_order,
        output_dim=args.output_dim,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sibylla adaptive geometric diagnostics experiment.")
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--train-per-family", type=int, default=600)
    parser.add_argument("--test-per-family", type=int, default=200)
    parser.add_argument("--history-len", type=int, default=7)
    parser.add_argument("--latent-dim", type=int, default=12)
    parser.add_argument("--proj-dim", type=int, default=3)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--output-dim", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "results_projected_logsig")
    args = parser.parse_args()

    cfg = validate_args(args)
    set_seed(args.seed)
    device = pick_device(args.cpu)
    bundle = make_projection_bundle(cfg, seed=args.seed + 99)

    train_data = build_dataset(np.random.default_rng(args.seed), n_per_family=args.train_per_family, cfg=cfg, bundle=bundle)
    test_data = build_dataset(np.random.default_rng(args.seed + 1), n_per_family=args.test_per_family, cfg=cfg, bundle=bundle)

    adaptive_train_orders = sibylla_order(train_data.winding_ratio)
    adaptive_test_orders = sibylla_order(test_data.winding_ratio)

    feature_specs = {
        "fixed_m1": np.ones(train_data.histories.shape[0], dtype=np.int64),
        "fixed_m2": np.full(train_data.histories.shape[0], 2, dtype=np.int64),
        "fixed_m3": np.full(train_data.histories.shape[0], 3, dtype=np.int64),
        "adaptive_sibylla": adaptive_train_orders,
    }
    test_feature_specs = {
        "fixed_m1": np.ones(test_data.histories.shape[0], dtype=np.int64),
        "fixed_m2": np.full(test_data.histories.shape[0], 2, dtype=np.int64),
        "fixed_m3": np.full(test_data.histories.shape[0], 3, dtype=np.int64),
        "adaptive_sibylla": adaptive_test_orders,
    }

    rows = []
    for model_idx, name in enumerate(["fixed_m1", "fixed_m2", "fixed_m3", "adaptive_sibylla"]):
        t0 = time.perf_counter()
        x_train = make_padded_features(train_data.histories, feature_specs[name], cfg, bundle)
        x_test = make_padded_features(test_data.histories, test_feature_specs[name], cfg, bundle)
        feature_build_sec = time.perf_counter() - t0

        torch.manual_seed(args.seed + 500 + model_idx)
        model = MLP(inp=cfg.max_logsig_dim, hidden=args.hidden, out=cfg.output_dim).to(device)
        train_model(model, x_train, train_data.targets, device, args.epochs, args.batch_size, args.lr)
        overall_rmse, family_rmse = evaluate_model(model, x_test, test_data.targets, test_data.family_ids, device)

        row = {
            "model": name,
            "overall_rmse": f"{overall_rmse:.6f}",
            "straight_rmse": f"{family_rmse[0]:.6f}",
            "curve_rmse": f"{family_rmse[1]:.6f}",
            "spiral_rmse": f"{family_rmse[2]:.6f}",
            "avg_order": f"{np.mean(test_feature_specs[name]):.6f}",
            "avg_feature_dim": f"{np.mean([logsig_dim(int(m), cfg.proj_dim) for m in test_feature_specs[name]]):.6f}",
            "feature_build_sec": f"{feature_build_sec:.6f}",
        }
        rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "overall_rmse",
                "straight_rmse",
                "curve_rmse",
                "spiral_rmse",
                "avg_order",
                "avg_feature_dim",
                "feature_build_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    order_path = args.out_dir / "adaptive_orders.csv"
    with order_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["family", "winding_ratio", "suggested_order"])
        for family_id, ratio, order in zip(test_data.family_ids, test_data.winding_ratio, adaptive_test_orders):
            writer.writerow([family_name(int(family_id)), f"{ratio:.6f}", int(order)])

    plot_path = plot_results(args.out_dir, rows, adaptive_test_orders, test_data.winding_ratio)

    print(
        f"device={device} latent_dim={cfg.latent_dim} proj_dim={cfg.proj_dim} "
        f"history_len={cfg.history_len} window_len={cfg.window_len} max_order={cfg.max_order}"
    )
    for row in rows:
        print(
            f"{row['model']:>16} | overall={row['overall_rmse']} | "
            f"straight={row['straight_rmse']} curve={row['curve_rmse']} spiral={row['spiral_rmse']} | "
            f"avg_order={row['avg_order']} avg_dim={row['avg_feature_dim']} build_sec={row['feature_build_sec']}"
        )
    print(f"adaptive_order_usage=m1:{np.mean(adaptive_test_orders == 1):.3f}, m2:{np.mean(adaptive_test_orders == 2):.3f}, m3:{np.mean(adaptive_test_orders == 3):.3f}")
    print(f"mean_winding_ratio=straight:{np.mean(test_data.winding_ratio[test_data.family_ids == 0]):.3f}, curve:{np.mean(test_data.winding_ratio[test_data.family_ids == 1]):.3f}, spiral:{np.mean(test_data.winding_ratio[test_data.family_ids == 2]):.3f}")
    print(f"saved={csv_path}")
    print(f"orders={order_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
