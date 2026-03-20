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


PATH_LEN = 8
PATH_DIM = 2
MAX_ORDER = 3
MAX_SIG_DIM = sum(PATH_DIM**k for k in range(1, MAX_ORDER + 1))


@dataclass
class Dataset:
    paths: torch.Tensor
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


def batch_kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bi,bj->bij", a, b).reshape(a.shape[0], -1)


def signature_dim(order: int) -> int:
    return sum(PATH_DIM**k for k in range(1, order + 1))


def truncated_signature(path: torch.Tensor, order: int) -> torch.Tensor:
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

    return torch.cat(levels[1:], dim=1)


def winding_ratio(path: np.ndarray, eps: float = 1e-6) -> float:
    chord = float(np.linalg.norm(path[-1] - path[0]))
    arc = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    return arc / (chord + eps)


def sibylla_order(ratio: np.ndarray) -> np.ndarray:
    order = np.ones_like(ratio, dtype=np.int64)
    order[ratio >= 1.1] = 2
    order[ratio >= 1.5] = 3
    return order


def make_straight_path(rng: np.random.Generator) -> np.ndarray:
    start = rng.uniform(-1.0, 1.0, size=2)
    direction = rng.normal(size=2)
    direction /= np.linalg.norm(direction) + 1e-8
    length = rng.uniform(0.8, 1.6)
    end = start + length * direction
    t = np.linspace(0.0, 1.0, PATH_LEN)
    path = (1.0 - t)[:, None] * start + t[:, None] * end
    perp = np.array([-direction[1], direction[0]])
    path += rng.normal(scale=0.015, size=(PATH_LEN, 1)) * perp[None, :]
    return path.astype(np.float32)


def make_curve_path(rng: np.random.Generator) -> np.ndarray:
    start = rng.uniform(-0.8, 0.8, size=2)
    direction = rng.normal(size=2)
    direction /= np.linalg.norm(direction) + 1e-8
    perp = np.array([-direction[1], direction[0]])
    end = start + rng.uniform(1.2, 1.8) * direction
    height = rng.uniform(0.55, 0.95) * rng.choice([-1.0, 1.0])
    control = 0.5 * (start + end) + height * perp
    t = np.linspace(0.0, 1.0, PATH_LEN)
    path = (1 - t)[:, None] ** 2 * start + 2 * (1 - t)[:, None] * t[:, None] * control + t[:, None] ** 2 * end
    return path.astype(np.float32)


def make_spiral_path(rng: np.random.Generator) -> np.ndarray:
    center = rng.uniform(-0.3, 0.3, size=2)
    radius0 = rng.uniform(0.9, 1.2)
    radius1 = rng.uniform(0.12, 0.28)
    turns = rng.uniform(1.35 * np.pi, 1.85 * np.pi)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    t = np.linspace(0.0, 1.0, PATH_LEN)
    radius = (1.0 - t) * radius0 + t * radius1
    angle = phase + turns * t
    path = np.stack([center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)], axis=1)
    return path.astype(np.float32)


def sample_family_path(rng: np.random.Generator, family: int) -> np.ndarray:
    for _ in range(200):
        if family == 0:
            path = make_straight_path(rng)
        elif family == 1:
            path = make_curve_path(rng)
        else:
            path = make_spiral_path(rng)
        ratio = winding_ratio(path)
        if family == 0 and ratio < 1.1:
            return path
        if family == 1 and 1.1 <= ratio < 1.5:
            return path
        if family == 2 and ratio >= 1.5:
            return path
    raise RuntimeError(f"Failed to sample valid family {family} path.")


def make_target_weights(seed: int = 123) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(scale=0.8, size=(signature_dim(1), 2)).astype(np.float32)
    w2 = rng.normal(scale=0.55, size=(PATH_DIM**2, 2)).astype(np.float32)
    w3 = rng.normal(scale=0.35, size=(PATH_DIM**3, 2)).astype(np.float32)
    return w1, w2, w3


def build_dataset(rng: np.random.Generator, n_per_family: int) -> Dataset:
    paths = []
    family_ids = []
    ratios = []
    w1, w2, w3 = make_target_weights()

    for family in range(3):
        for _ in range(n_per_family):
            path = sample_family_path(rng, family)
            paths.append(path)
            family_ids.append(family)
            ratios.append(winding_ratio(path))

    path_tensor = torch.from_numpy(np.stack(paths))
    sig1 = truncated_signature(path_tensor, 1).numpy()
    sig2 = truncated_signature(path_tensor, 2).numpy()
    sig3 = truncated_signature(path_tensor, 3).numpy()
    sig2_block = sig2[:, signature_dim(1) :]
    sig3_block = sig3[:, signature_dim(2) :]

    targets = []
    for i, family in enumerate(family_ids):
        out = sig1[i] @ w1
        if family >= 1:
            out = out + sig2_block[i] @ w2
        if family >= 2:
            out = out + sig3_block[i] @ w3
        targets.append(out.astype(np.float32))

    return Dataset(
        paths=path_tensor,
        targets=torch.from_numpy(np.stack(targets)),
        family_ids=np.array(family_ids, dtype=np.int64),
        winding_ratio=np.array(ratios, dtype=np.float32),
    )


def make_padded_features(paths: torch.Tensor, orders: np.ndarray) -> torch.Tensor:
    features = torch.zeros(paths.shape[0], MAX_SIG_DIM, dtype=paths.dtype)
    for order in (1, 2, 3):
        idx = np.where(orders == order)[0]
        if idx.size == 0:
            continue
        block = truncated_signature(paths[idx], order)
        features[idx, : block.shape[1]] = block
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


def plot_results(out_dir: Path, rows: list[dict[str, str]], adaptive_orders: np.ndarray) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))

    axes[0].bar(
        [r["model"] for r in rows],
        [float(r["overall_rmse"]) for r in rows],
        color=["#f58518", "#4c78a8", "#72b7b2", "#54a24b"],
    )
    axes[0].set_title("Overall RMSE")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(axis="y", alpha=0.25)

    order_bins = [np.mean(adaptive_orders == 1), np.mean(adaptive_orders == 2), np.mean(adaptive_orders == 3)]
    axes[1].bar(["m=1", "m=2", "m=3"], order_bins, color=["#f58518", "#4c78a8", "#54a24b"])
    axes[1].set_title("Sibylla Order Usage")
    axes[1].set_ylabel("fraction")
    axes[1].grid(axis="y", alpha=0.25)

    dims = [float(r["avg_feature_dim"]) for r in rows]
    axes[2].bar([r["model"] for r in rows], dims, color=["#f58518", "#4c78a8", "#72b7b2", "#54a24b"], label="avg active dim")
    ax2 = axes[2].twinx()
    ax2.plot([r["model"] for r in rows], [float(r["feature_build_sec"]) for r in rows], marker="o", color="#e45756", linewidth=2, label="build sec")
    axes[2].set_title("Compute Cost")
    axes[2].set_ylabel("avg active dim")
    ax2.set_ylabel("feature build sec")
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("Legacy Experiment: 2D Signature Prototype for Sibylla", fontsize=14, y=1.02)
    fig.tight_layout()
    path = out_dir / "legacy_signature_summary.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy Sibylla adaptive signature-order prototype.")
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--train-per-family", type=int, default=600)
    parser.add_argument("--test-per-family", type=int, default=200)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "results_legacy")
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.cpu)

    train_data = build_dataset(np.random.default_rng(args.seed), n_per_family=args.train_per_family)
    test_data = build_dataset(np.random.default_rng(args.seed + 1), n_per_family=args.test_per_family)

    train_ratios = train_data.winding_ratio
    test_ratios = test_data.winding_ratio
    adaptive_train_orders = sibylla_order(train_ratios)
    adaptive_test_orders = sibylla_order(test_ratios)

    feature_specs = {
        "fixed_m1": np.ones(train_data.paths.shape[0], dtype=np.int64),
        "fixed_m2": np.full(train_data.paths.shape[0], 2, dtype=np.int64),
        "fixed_m3": np.full(train_data.paths.shape[0], 3, dtype=np.int64),
        "adaptive_sibylla": adaptive_train_orders,
    }
    test_feature_specs = {
        "fixed_m1": np.ones(test_data.paths.shape[0], dtype=np.int64),
        "fixed_m2": np.full(test_data.paths.shape[0], 2, dtype=np.int64),
        "fixed_m3": np.full(test_data.paths.shape[0], 3, dtype=np.int64),
        "adaptive_sibylla": adaptive_test_orders,
    }

    rows = []
    for name in ["fixed_m1", "fixed_m2", "fixed_m3", "adaptive_sibylla"]:
        t0 = time.perf_counter()
        x_train = make_padded_features(train_data.paths, feature_specs[name])
        x_test = make_padded_features(test_data.paths, test_feature_specs[name])
        feature_build_sec = time.perf_counter() - t0

        model = MLP(inp=MAX_SIG_DIM, hidden=args.hidden, out=2).to(device)
        train_model(model, x_train, train_data.targets, device, args.epochs, args.batch_size, args.lr)
        overall_rmse, family_rmse = evaluate_model(model, x_test, test_data.targets, test_data.family_ids, device)

        row = {
            "model": name,
            "overall_rmse": f"{overall_rmse:.6f}",
            "straight_rmse": f"{family_rmse[0]:.6f}",
            "curve_rmse": f"{family_rmse[1]:.6f}",
            "spiral_rmse": f"{family_rmse[2]:.6f}",
            "avg_order": f"{np.mean(test_feature_specs[name]):.6f}",
            "avg_feature_dim": f"{np.mean([signature_dim(int(m)) for m in test_feature_specs[name]]):.6f}",
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
        for family_id, ratio, order in zip(test_data.family_ids, test_ratios, adaptive_test_orders):
            writer.writerow([family_name(int(family_id)), f"{ratio:.6f}", int(order)])

    plot_path = plot_results(args.out_dir, rows, adaptive_test_orders)

    print(f"device={device}")
    for row in rows:
        print(
            f"{row['model']:>16} | overall={row['overall_rmse']} | "
            f"straight={row['straight_rmse']} curve={row['curve_rmse']} spiral={row['spiral_rmse']} | "
            f"avg_order={row['avg_order']} avg_dim={row['avg_feature_dim']} build_sec={row['feature_build_sec']}"
        )
    print(f"adaptive_order_usage=m1:{np.mean(adaptive_test_orders==1):.3f}, m2:{np.mean(adaptive_test_orders==2):.3f}, m3:{np.mean(adaptive_test_orders==3):.3f}")
    print(f"saved={csv_path}")
    print(f"orders={order_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
