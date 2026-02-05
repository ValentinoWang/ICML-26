import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import optim

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BASE_TABLES_DIR = ROOT / "Total_results" / "Tables" / SCRIPT_DIR.name
BASE_FIGURES_DIR = ROOT / "Total_results" / "Figures" / SCRIPT_DIR.name
TABLES_DIR = BASE_TABLES_DIR / "results"
FIGURES_DIR = BASE_FIGURES_DIR / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from filter.losses import classification_loss, contraction_loss, correction_reg, ess_loss  # noqa: E402
from filter.set_aware.model import SetAwareBiasRobustFilter  # noqa: E402
from filter.standard.model import StandardFilter  # noqa: E402
from plot_exp7_recursive import plot_series  # noqa: E402


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def build_labels(candidates: np.ndarray, theta_true: np.ndarray, top_ratio: float) -> np.ndarray:
    dists = np.linalg.norm(candidates - theta_true, axis=1)
    k = max(1, int(len(candidates) * top_ratio))
    thresh = np.partition(dists, k - 1)[k - 1]
    return (dists <= thresh).astype(np.float32)


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    d = X.shape[1]
    xtx = X.T @ X
    return np.linalg.solve(xtx + (alpha + 1e-6) * np.eye(d), X.T @ y)


def generate_synthetic_batch(rng: np.random.Generator, X_pool: np.ndarray, theta: np.ndarray, noise_std: float, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.choice(len(X_pool), size=batch_size, replace=False)
    X_b = X_pool[idx]
    y_b = X_b @ theta + noise_std * rng.normal(size=batch_size)
    return X_b, y_b


def run_recursive(args: argparse.Namespace, device: torch.device, seed: int, data: Dict[str, np.ndarray]) -> Dict[str, List[float]]:
    rng = set_seed(seed)
    theta_star = data["theta_star"]
    theta_t_no = theta_star.copy()
    theta_t_batch = theta_star.copy()
    theta_t_ours = theta_star.copy()
    theta_t_dst = theta_star.copy()
    theta_t_l2ac = theta_star.copy()

    model = SetAwareBiasRobustFilter(
        dim=theta_star.shape[0],
        hidden=args.hidden,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dst_bias = StandardFilter(dim=theta_star.shape[0], hidden=args.hidden, dropout=args.dropout).to(device)
    dst_main = StandardFilter(dim=theta_star.shape[0], hidden=args.hidden, dropout=args.dropout).to(device)
    opt_dst = optim.Adam(
        list(dst_bias.parameters()) + list(dst_main.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    l2ac_model = StandardFilter(dim=theta_star.shape[0], hidden=args.hidden, dropout=args.dropout).to(device)
    opt_l2ac = optim.Adam(l2ac_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    batch_model = StandardFilter(dim=theta_star.shape[0] * 3, hidden=args.hidden, dropout=args.dropout).to(device)
    opt_batch = optim.Adam(batch_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    mse_series = {"no_filter": [], "batch_stats": [], "dst": [], "l2ac": [], "ours": []}
    norm_series = {"no_filter": [], "batch_stats": [], "dst": [], "l2ac": [], "ours": []}

    X_test = data["X_test"]
    y_test_true = data["y_test_true"]

    theta_star_t = torch.from_numpy(theta_star[None, None, :]).float().to(device)

    for _ in range(args.generations):
        # Synthesize biased dataset from current theta_t_no
        X_b, y_b = generate_synthetic_batch(rng, data["X_test"], theta_t_no, args.noise_std, args.batch_size)
        theta_hat = ridge_fit(X_b, y_b, alpha=args.alpha_ridge)

        # Build candidates around theta_hat
        candidates = theta_hat + args.candidate_noise * rng.normal(size=(args.candidates_per_gen, theta_star.shape[0]))
        labels = build_labels(candidates, theta_star, top_ratio=args.top_ratio)

        x = torch.from_numpy(candidates[None, ...]).float().to(device)
        y = torch.from_numpy(labels[None, ...]).float().to(device)
        theta_hat_t = torch.from_numpy(theta_hat[None, None, :]).float().to(device)

        # No filter update (biased Ridge estimator)
        theta_t_no = theta_hat

        # Pointwise + Batch Stats baseline
        x_flat = x.squeeze(0)
        batch_mean = x_flat.mean(dim=0, keepdim=True)
        batch_var = x_flat.var(dim=0, unbiased=False, keepdim=True)
        feat_batch = torch.cat([x_flat, batch_mean.expand_as(x_flat), batch_var.expand_as(x_flat)], dim=1)
        feat_batch = feat_batch.unsqueeze(0)
        w_batch = batch_model(feat_batch)
        theta_batch_new = StandardFilter.weighted_estimate(x, w_batch)
        loss_batch = (
            args.lambda_class * classification_loss(w_batch, y)
            + args.lambda_contract * contraction_loss(theta_batch_new, theta_star_t)
            + args.lambda_ess * ess_loss(w_batch, tau=args.tau)
        )
        opt_batch.zero_grad()
        loss_batch.backward()
        opt_batch.step()
        theta_t_batch = theta_t_batch + args.ours_contraction * (
            theta_batch_new.detach().cpu().numpy().squeeze(0) - theta_t_batch
        )

        # DST baseline: bias head fits biased estimate; main head downweights bias-heavy samples.
        w_bias = dst_bias(x)
        theta_bias = StandardFilter.weighted_estimate(x, w_bias)
        loss_bias = contraction_loss(theta_bias, theta_hat_t) + args.lambda_ess * ess_loss(w_bias, tau=args.tau)

        w_main = dst_main(x)
        bias_mask = (1.0 - w_bias.detach()).clamp_min(0.05)
        w_dst = (w_main * bias_mask).clamp_min(1e-4)
        theta_dst_new = StandardFilter.weighted_estimate(x, w_dst)
        loss_dst = (
            args.lambda_class * classification_loss(w_main, y)
            + args.lambda_contract * contraction_loss(theta_dst_new, theta_star_t)
            + args.lambda_ess * ess_loss(w_main, tau=args.tau)
        )
        opt_dst.zero_grad()
        (loss_bias + loss_dst).backward()
        opt_dst.step()
        theta_t_dst = theta_t_dst + args.ours_contraction * (
            theta_dst_new.detach().cpu().numpy().squeeze(0) - theta_t_dst
        )

        # L2AC-style baseline: meta-aligned weighting using clean reference direction.
        w_l2ac_raw = l2ac_model(x)
        theta_l2ac_t = torch.from_numpy(theta_t_l2ac[None, None, :]).float().to(device)
        theta_l2ac_vec = theta_l2ac_t.squeeze()
        delta_dir = (theta_star_t - theta_l2ac_t).squeeze()
        align_score = torch.relu((x_flat - theta_l2ac_vec) @ delta_dir).unsqueeze(0)
        w_l2ac = (w_l2ac_raw * (align_score + 1e-6)).clamp_min(1e-4)
        theta_l2ac_new = StandardFilter.weighted_estimate(x, w_l2ac)
        loss_l2ac = (
            args.lambda_class * classification_loss(w_l2ac_raw, y)
            + args.lambda_contract * contraction_loss(theta_l2ac_new, theta_star_t)
            + args.lambda_ess * ess_loss(w_l2ac_raw, tau=args.tau)
        )
        opt_l2ac.zero_grad()
        loss_l2ac.backward()
        opt_l2ac.step()
        theta_t_l2ac = theta_t_l2ac + args.ours_contraction * (
            theta_l2ac_new.detach().cpu().numpy().squeeze(0) - theta_t_l2ac
        )

        # Ours: correction only (weights for aux supervision), estimate = delta
        w, delta = model(x)
        theta_new = delta  # estimation uses correction only; weights aux
        loss = (
            args.lambda_class * classification_loss(w, y)
            + args.lambda_contract * contraction_loss(theta_new, theta_star_t)
            + args.lambda_ess * ess_loss(w, tau=args.tau)
            + args.lambda_reg * correction_reg(delta)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        theta_t_ours = theta_t_ours + args.ours_contraction * (theta_new.detach().cpu().numpy().squeeze(0) - theta_t_ours)

        # Evaluation on real test set
        for key, theta in [
            ("no_filter", theta_t_no),
            ("batch_stats", theta_t_batch),
            ("dst", theta_t_dst),
            ("l2ac", theta_t_l2ac),
            ("ours", theta_t_ours),
        ]:
            theta_vec = np.asarray(theta).reshape(-1)
            y_pred = X_test @ theta_vec
            mse = float(np.mean((y_pred - y_test_true) ** 2))
            mse_series[key].append(mse)
            norm_series[key].append(float(np.linalg.norm(theta_vec)))

    return {"mse": mse_series, "norm": norm_series}


def aggregate(seed_runs: List[Dict[str, Dict[str, List[float]]]]) -> Dict[str, Dict[str, List[float]]]:
    keys = seed_runs[0].keys()
    out: Dict[str, Dict[str, List[float]]] = {}
    for k in keys:
        methods = seed_runs[0][k].keys()
        out[k] = {}
        for m in methods:
            arr = np.array([run[k][m] for run in seed_runs])
            out[k][m] = {"mean": arr.mean(axis=0).tolist(), "std": arr.std(axis=0).tolist()}
    return out


def save_csv(stats: Dict[str, Dict[str, List[float]]], out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gens = len(next(iter(stats["mse"].values()))["mean"])
    with (out_dir / "exp7_trajectories.csv").open("w") as f:
        header = ["generation"]
        for m in stats["mse"].keys():
            header.extend([f"{m}_mse_mean", f"{m}_mse_std", f"{m}_norm_mean", f"{m}_norm_std"])
        f.write(",".join(header) + "\n")
        for i in range(gens):
            row = [str(i + 1)]
            for m in stats["mse"].keys():
                row.append(f"{stats['mse'][m]['mean'][i]:.6f}")
                row.append(f"{stats['mse'][m]['std'][i]:.6f}")
                row.append(f"{stats['norm'][m]['mean'][i]:.6f}")
                row.append(f"{stats['norm'][m]['std'][i]:.6f}")
            f.write(",".join(row) + "\n")


def prepare_data(test_size: float, seed: int) -> Dict[str, np.ndarray]:
    data = fetch_california_housing()
    X = data["data"]
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Oracle (OLS)
    theta_star = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
    b_star = float(y_train.mean() - X_train.mean(axis=0) @ theta_star)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test_true": y_test,
        "theta_star": theta_star,
        "b_star": b_star,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp7: Real-world recursive regression with ridge bias.")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--candidate-noise", type=float, default=0.05)
    parser.add_argument("--candidates-per-gen", type=int, default=64)
    parser.add_argument("--alpha-ridge", type=float, default=10.0)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--top-ratio", type=float, default=0.3)
    parser.add_argument("--lambda-class", type=float, default=0.05)
    parser.add_argument("--lambda-contract", type=float, default=1.0)
    parser.add_argument("--lambda-ess", type=float, default=0.01)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=50.0)
    parser.add_argument("--ours-contraction", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1088, 2195, 4960])
    parser.add_argument("--out-dir", type=pathlib.Path, default=TABLES_DIR, help="Directory to store CSV/JSON outputs.")
    parser.add_argument("--fig-dir", type=pathlib.Path, default=FIGURES_DIR, help="Directory to store figures.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    data = prepare_data(test_size=args.test_size, seed=args.seeds[0])
    seed_runs: List[Dict[str, Dict[str, List[float]]]] = []
    for s in args.seeds:
        seed_runs.append(run_recursive(args, device=device, seed=s, data=data))

    stats = aggregate(seed_runs)
    plot_series(stats, args.fig_dir, n_seeds=len(args.seeds))
    save_csv(stats, args.out_dir)

    runtime = {
        "device": str(device),
        "seeds": args.seeds,
        "total_time_sec": time.time() - t0,
        "note": "Exp7 California Housing recursive ridge; methods: no filter vs MLP+Corr (weights aux only).",
    }
    with (args.out_dir / "runtime_exp7.json").open("w") as f:
        json.dump(runtime, f, indent=2)
    print(f"Saved Exp7 results to {args.out_dir} (tables) and {args.fig_dir} (figures)")


if __name__ == "__main__":
    main()
