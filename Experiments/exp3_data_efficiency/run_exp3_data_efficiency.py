import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import optim

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BASE_TABLES_DIR = ROOT / "Total_results" / "Tables" / SCRIPT_DIR.name
BASE_FIGURES_DIR = ROOT / "Total_results" / "Figures" / SCRIPT_DIR.name
TABLES_DIR = BASE_TABLES_DIR / "results"
FIGURES_DIR = BASE_FIGURES_DIR / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bias.bias_sources import bayesian_map_estimator, const_bias_estimator, ridge_estimator
from filter.data import make_bias_vector, sample_clean_reference, set_seed
from filter.losses import classification_loss, contraction_loss, correction_reg, ess_loss
from filter.set_aware.model import SetAwareBiasRobustFilter
from filter.standard.model import StandardFilter
from make_figures_exp3 import plot_results  # noqa: E402


def build_labels(candidates: np.ndarray, theta_true: np.ndarray, top_ratio: float) -> np.ndarray:
    dists = np.linalg.norm(candidates - theta_true, axis=1)
    k = max(1, int(len(candidates) * top_ratio))
    thresh = np.partition(dists, k - 1)[k - 1]
    return (dists <= thresh).astype(np.float32)


def generate_observations(rng: np.random.Generator, theta_true: np.ndarray, n: int, noise_std: float) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(size=(n, theta_true.shape[0]))
    y = X @ theta_true + noise_std * rng.normal(size=n)
    return X, y


def least_squares_estimator(X: np.ndarray, y: np.ndarray, ridge_eps: float = 1e-6) -> np.ndarray:
    """
    Simple closed-form least squares used as the unbiased core for the hard-bias scenario.
    """
    d = X.shape[1]
    xtx = X.T @ X
    return np.linalg.solve(xtx + ridge_eps * np.eye(d), X.T @ y)


def run_scenario(
    args: argparse.Namespace,
    theta_true: np.ndarray,
    scenario: str,
    device: torch.device,
    seed: int,
    small_n: int,
    bias_vec: np.ndarray,
) -> Dict[str, List[float]]:
    rng = set_seed(seed)
    theta_good = sample_clean_reference(rng, theta_true, n=args.calibration_size, noise_std=args.noise_std)
    theta_good_t = torch.from_numpy(theta_good[None, ...]).float().to(device)

    std_model = StandardFilter(dim=args.dim, hidden=args.hidden, dropout=args.dropout).to(device)
    sa_model = SetAwareBiasRobustFilter(
        dim=args.dim,
        hidden=args.hidden,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    opt_std = optim.Adam(std_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_sa = optim.Adam(sa_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    theta_std = theta_good.copy()
    theta_sa = theta_good.copy()

    series: Dict[str, List[float]] = {"baseline_big": [], "ours_small": [], "small_no_filter": []}

    for _ in range(args.generations):
        # Big-data baseline (no filter)
        X_big, y_big = generate_observations(rng, theta_true, n=args.big_n, noise_std=args.noise_std)
        if scenario == "ridge":
            theta_hat_big = ridge_estimator(X_big, y_big, alpha=args.ridge_alpha)
        elif scenario == "bayes":
            theta_hat_big = bayesian_map_estimator(
                X_big,
                y_big,
                mu_prior=args.mu_prior * np.ones(args.dim),
                sigma_prior=args.sigma_prior,
                noise_var=args.noise_std**2,
            )
        elif scenario == "const":
            theta_hat_big_unbiased = least_squares_estimator(X_big, y_big)
            theta_hat_big = const_bias_estimator(theta_hat_big_unbiased, bias_vec)
        else:
            raise ValueError(f"Unknown scenario {scenario}")
        series["baseline_big"].append(float(np.linalg.norm(theta_hat_big - theta_true)))

        # Small data candidates
        X_small, y_small = generate_observations(rng, theta_true, n=small_n, noise_std=args.noise_std)
        if scenario == "ridge":
            theta_hat_small = ridge_estimator(X_small, y_small, alpha=args.ridge_alpha)
        elif scenario == "bayes":
            theta_hat_small = bayesian_map_estimator(
                X_small,
                y_small,
                mu_prior=args.mu_prior * np.ones(args.dim),
                sigma_prior=args.sigma_prior,
                noise_var=args.noise_std**2,
            )
        elif scenario == "const":
            theta_hat_small_unbiased = least_squares_estimator(X_small, y_small)
            theta_hat_small = const_bias_estimator(theta_hat_small_unbiased, bias_vec)
        else:
            raise ValueError(f"Unknown scenario {scenario}")
        series["small_no_filter"].append(float(np.linalg.norm(theta_hat_small - theta_true)))

        # Build candidates around small estimator (add jitter) to feed filter.
        candidates = theta_hat_small + args.candidate_noise * rng.normal(size=(args.candidates_per_gen, args.dim))
        labels = build_labels(candidates, theta_true, top_ratio=args.top_ratio)

        x = torch.from_numpy(candidates[None, ...]).float().to(device)
        y = torch.from_numpy(labels[None, ...]).float().to(device)

        # Standard filter
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

        # Set-aware filter: weights as auxiliary, estimation uses correction only (w=1)
        w_sa, delta_phi = sa_model(x)
        theta_new_sa = delta_phi
        loss_sa = (
            args.lambda_class * classification_loss(w_sa, y)
            + args.lambda_contract * contraction_loss(theta_new_sa, theta_good_t)
            + args.lambda_ess * ess_loss(w_sa, tau=args.tau)
            + args.lambda_reg * correction_reg(delta_phi)
        )
        opt_sa.zero_grad()
        loss_sa.backward()
        opt_sa.step()

        theta_sa = theta_sa + args.ours_contraction * (
            theta_new_sa.detach().cpu().numpy().squeeze(0) - theta_sa
        )
        series["ours_small"].append(float(np.linalg.norm(theta_sa - theta_true)))

    return series


def aggregate_seed_series(seed_series: List[Dict[str, List[float]]]) -> Dict[str, Dict[str, List[float]]]:
    methods = seed_series[0].keys()
    stats: Dict[str, Dict[str, List[float]]] = {}
    for m in methods:
        arr = np.array([s[m] for s in seed_series])
        stats[m] = {"mean": arr.mean(axis=0).tolist(), "std": arr.std(axis=0).tolist()}
    return stats


def save_csv(all_series: Dict[str, Dict[str, Dict[str, List[float]]]], out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, series in all_series.items():
        gens = np.arange(1, len(next(iter(series.values()))["mean"]) + 1)
        path = out_dir / f"exp3_data_eff_{name}.csv"
        with path.open("w") as f:
            header = ["generation"]
            for k in series:
                header.extend([f"{k}_mean", f"{k}_std"])
            f.write(",".join(header) + "\n")
            for i in range(len(gens)):
                row = [str(gens[i])]
                for k in series:
                    row.append(f"{series[k]['mean'][i]:.6f}")
                    row.append(f"{series[k]['std'][i]:.6f}")
                f.write(",".join(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Exp3: Data efficiency under ridge, bayes, and hard biases.")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--ridge-alpha", type=float, default=5.0)
    parser.add_argument("--mu-prior", type=float, default=0.0)
    parser.add_argument("--sigma-prior", type=float, default=0.5)
    parser.add_argument("--bias-norm", type=float, default=0.5, help="L2 norm of the hard-coded bias vector.")
    parser.add_argument("--big-n", type=int, default=10000)
    parser.add_argument("--small-n", type=int, default=100)
    parser.add_argument(
        "--small-n-bayes-extreme",
        type=int,
        default=5,
        help="Small sample size for the extreme Bayes scarcity scenario (Exp3.3).",
    )
    parser.add_argument("--candidates-per-gen", type=int, default=64)
    parser.add_argument("--candidate-noise", type=float, default=0.1)
    parser.add_argument("--calibration-size", type=int, default=200)
    parser.add_argument("--standard-contraction", type=float, default=0.2)
    parser.add_argument("--ours-contraction", type=float, default=0.5)
    parser.add_argument("--top-ratio", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=50.0)
    parser.add_argument("--lambda-class", type=float, default=0.05)
    parser.add_argument("--lambda-contract", type=float, default=1.0)
    parser.add_argument("--lambda-ess", type=float, default=0.1)
    parser.add_argument("--lambda-reg", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1088, 2195, 4960, 1545, 3549, 1440, 3050, 5414],
        help="Seeds for mean/std shading.",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out-dir", type=pathlib.Path, default=TABLES_DIR, help="Directory to store CSV/JSON outputs.")
    parser.add_argument("--fig-dir", type=pathlib.Path, default=FIGURES_DIR, help="Directory to store figures.")
    parser.add_argument("--save-csv", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    start_time = time.time()
    seeds: Sequence[int] = args.seeds if args.seeds is not None else [args.seed]
    theta_ridge = np.ones(args.dim)
    theta_bayes = np.ones(args.dim) * args.mu_prior
    theta_const = np.ones(args.dim)
    bias_vec_const = make_bias_vector(args.dim, args.bias_norm)

    all_series: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    titles: Dict[str, str] = {}
    scenario_cfgs = [
        {
            "name": "ridge",
            "scenario_type": "ridge",
            "theta": theta_ridge,
            "small_n": args.small_n,
            "bias_vec": np.zeros(args.dim),
            "title": f"Ridge alpha={args.ridge_alpha}, n={args.small_n}",
        },
        {
            "name": "bayes",
            "scenario_type": "bayes",
            "theta": theta_bayes,
            "small_n": args.small_n,
            "bias_vec": np.zeros(args.dim),
            "title": f"Bayes mu_prior={args.mu_prior}, sigma={args.sigma_prior}, n={args.small_n}",
        },
        {
            "name": "bayes_n5",
            "scenario_type": "bayes",
            "theta": theta_bayes,
            "small_n": args.small_n_bayes_extreme,
            "bias_vec": np.zeros(args.dim),
            "title": f"Bayes (extreme) mu_prior={args.mu_prior}, sigma={args.sigma_prior}, n={args.small_n_bayes_extreme}",
        },
        {
            "name": "const",
            "scenario_type": "const",
            "theta": theta_const,
            "small_n": args.small_n,
            "bias_vec": bias_vec_const,
            "title": f"Const bias ||b||={args.bias_norm}, n={args.small_n}",
        },
    ]

    for cfg in scenario_cfgs:
        titles[cfg["name"]] = cfg["title"]
        seed_series: List[Dict[str, List[float]]] = []
        for s in seeds:
            seed_series.append(
                run_scenario(
                    args,
                    theta_true=cfg["theta"],
                    scenario=cfg["scenario_type"],
                    device=device,
                    seed=s,
                    small_n=cfg["small_n"],
                    bias_vec=cfg["bias_vec"],
                )
            )
        all_series[cfg["name"]] = aggregate_seed_series(seed_series)

    fig_dir = args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_results(all_series, fig_dir / "exp3_data_efficiency.png", args, n_seeds=len(seeds), titles=titles)
    if args.save_csv:
        save_csv(all_series, args.out_dir)
    runtime = {
        "device": str(device),
        "seeds": list(seeds),
        "total_time_sec": time.time() - start_time,
        "scenarios": [
            {"name": cfg["name"], "small_n": cfg["small_n"], "scenario_type": cfg["scenario_type"]}
            for cfg in scenario_cfgs
        ],
    }
    with (args.out_dir / "runtime_exp3_data_efficiency.json").open("w") as f:
        json.dump(runtime, f, indent=2)
    print(f"Saved Exp3 data efficiency results to {args.out_dir} (tables) and {fig_dir} (figures)")


if __name__ == "__main__":
    main()
