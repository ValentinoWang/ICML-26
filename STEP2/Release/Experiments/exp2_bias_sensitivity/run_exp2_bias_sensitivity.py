import argparse
import json
import pathlib
import sys
import time
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
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

from bias.bias_sources import bayesian_map_estimator, ridge_estimator
from filter.data import make_bias_vector, sample_candidates, sample_clean_reference, set_seed
from filter.losses import classification_loss, contraction_loss, correction_reg, ess_loss
from filter.set_aware.model import SetAwareBiasRobustFilter
from filter.standard.model import StandardFilter
from plot_exp2 import display_label, plot_by_method, plot_param_curve, plot_trajectories_grid  # noqa: E402

CandidateGenerator = Callable[[np.random.Generator], Tuple[np.ndarray, np.ndarray]]
METHODS = ["no_filter", "standard_filter", "dst", "k_center", "ours"]


def build_labels(candidates: np.ndarray, theta_true: np.ndarray, top_ratio: float) -> np.ndarray:
    dists = np.linalg.norm(candidates - theta_true, axis=1)
    k = max(1, int(len(candidates) * top_ratio))
    thresh = np.partition(dists, k - 1)[k - 1]
    return (dists <= thresh).astype(np.float32)


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


def mean_tail(values: List[float], tail: int) -> float:
    return float(np.mean(values[-tail:])) if values else float("nan")


def clamp_delta_phi(delta_phi: torch.Tensor, clip: float) -> torch.Tensor:
    if clip <= 0:
        return delta_phi
    norm = delta_phi.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    factor = torch.clamp(clip / norm, max=1.0)
    return delta_phi * factor


def aggregate_seed_series(seed_series: List[Dict[str, List[float]]]) -> Dict[str, Dict[str, List[float]]]:
    methods = seed_series[0].keys()
    stats: Dict[str, Dict[str, List[float]]] = {}
    for m in methods:
        arr = np.array([s[m] for s in seed_series])
        stats[m] = {"mean": arr.mean(axis=0).tolist(), "std": arr.std(axis=0).tolist()}
    return stats


def aggregate_tail_stats(seed_series: List[Dict[str, List[float]]], tail: int) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for m in seed_series[0]:
        vals = np.array([mean_tail(s[m], tail=tail) for s in seed_series])
        stats[m] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return stats


def run_filter_process(
    generator: CandidateGenerator,
    args: argparse.Namespace,
    theta_true: np.ndarray,
    theta_good: np.ndarray,
    device: torch.device,
    seed: int,
    std_contraction: float | None = None,
    ours_contraction: float | None = None,
) -> Dict[str, List[float]]:
    rng = set_seed(seed)
    std_c = args.standard_contraction if std_contraction is None else std_contraction
    ours_c = args.ours_contraction if ours_contraction is None else ours_contraction

    std_model = StandardFilter(dim=args.dim, hidden=args.hidden, dropout=args.dropout).to(device)
    dst_bias = StandardFilter(dim=args.dim, hidden=args.hidden, dropout=args.dropout).to(device)
    dst_main = StandardFilter(dim=args.dim, hidden=args.hidden, dropout=args.dropout).to(device)
    sa_model = SetAwareBiasRobustFilter(
        dim=args.dim,
        hidden=args.hidden,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    opt_std = optim.Adam(std_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_dst = optim.Adam(
        list(dst_bias.parameters()) + list(dst_main.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    opt_sa = optim.Adam(sa_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    theta_std = theta_good.copy()
    theta_dst = theta_good.copy()
    theta_kcenter = theta_good.copy()
    theta_sa = theta_good.copy()
    theta_good_t = torch.from_numpy(theta_good[None, ...]).float().to(device)
    errors: Dict[str, List[float]] = {k: [] for k in METHODS}

    warmup_steps = int(args.generations * args.warmup_frac)

    for step in range(args.generations):
        candidates, theta_hat = generator(rng)
        labels = build_labels(candidates, theta_true, top_ratio=args.top_ratio)

        x = torch.from_numpy(candidates[None, ...]).float().to(device)
        y = torch.from_numpy(labels[None, ...]).float().to(device)

        errors["no_filter"].append(float(np.linalg.norm(theta_hat - theta_true)))

        # k-Center (coreset) baseline: diverse subset average.
        k = max(1, int(candidates.shape[0] * args.coreset_ratio))
        k_idx = k_center_greedy(candidates, k, rng)
        theta_kc_new = candidates[k_idx].mean(axis=0)
        theta_kcenter = theta_kcenter + std_c * (theta_kc_new - theta_kcenter)
        errors["k_center"].append(float(np.linalg.norm(theta_kcenter - theta_true)))

        in_warmup = step < warmup_steps
        curr_lambda_class = 0.0 if in_warmup else args.lambda_class
        curr_lambda_ess = 0.0 if in_warmup else args.lambda_ess

        # Standard filter.
        w_std = std_model(x)
        w_std_eff = torch.ones_like(w_std) if in_warmup else w_std
        theta_w_std = StandardFilter.weighted_estimate(x, w_std_eff)
        loss_std = (
            curr_lambda_class * classification_loss(w_std, y)
            + args.lambda_contract * contraction_loss(theta_w_std, theta_good_t)
            + curr_lambda_ess * ess_loss(w_std, tau=args.tau)
        )
        opt_std.zero_grad()
        loss_std.backward()
        opt_std.step()
        theta_std = theta_std + std_c * (theta_w_std.detach().cpu().numpy().squeeze(0) - theta_std)
        errors["standard_filter"].append(float(np.linalg.norm(theta_std - theta_true)))

        # DST baseline: bias head learns biased estimate, main head downweights bias-heavy samples.
        theta_hat_t = torch.from_numpy(theta_hat[None, ...]).float().to(device)
        w_bias = dst_bias(x)
        theta_bias = StandardFilter.weighted_estimate(x, w_bias)
        loss_bias = contraction_loss(theta_bias, theta_hat_t) + curr_lambda_ess * ess_loss(w_bias, tau=args.tau)

        w_main = dst_main(x)
        bias_mask = (1.0 - w_bias.detach()).clamp_min(0.05)
        w_dst = (w_main * bias_mask).clamp_min(1e-4)
        theta_dst_new = StandardFilter.weighted_estimate(x, w_dst)
        loss_dst = (
            curr_lambda_class * classification_loss(w_main, y)
            + args.lambda_contract * contraction_loss(theta_dst_new, theta_good_t)
            + curr_lambda_ess * ess_loss(w_main, tau=args.tau)
        )
        opt_dst.zero_grad()
        (loss_bias + loss_dst).backward()
        opt_dst.step()
        theta_dst = theta_dst + std_c * (theta_dst_new.detach().cpu().numpy().squeeze(0) - theta_dst)
        errors["dst"].append(float(np.linalg.norm(theta_dst - theta_true)))

        # Set-aware filter with auxiliary weighting head (used only for weak supervision; final estimate uses correction only).
        w_sa, delta_phi = sa_model(x)
        delta_phi = clamp_delta_phi(delta_phi, args.correction_clip)
        theta_new_sa = delta_phi  # weights guide encoder via losses, not the final estimate
        loss_sa = (
            curr_lambda_class * classification_loss(w_sa, y)
            + args.lambda_contract * contraction_loss(theta_new_sa, theta_good_t)
            + curr_lambda_ess * ess_loss(w_sa, tau=args.tau)
            + args.lambda_reg * correction_reg(delta_phi)
        )
        opt_sa.zero_grad()
        loss_sa.backward()
        opt_sa.step()

        theta_sa = theta_sa + ours_c * (theta_new_sa.detach().cpu().numpy().squeeze(0) - theta_sa)
        errors["ours"].append(float(np.linalg.norm(theta_sa - theta_true)))

    return errors


def make_const_generator(
    theta_true: np.ndarray,
    bias_norm: float,
    args: argparse.Namespace,
    n_candidates: int,
) -> CandidateGenerator:
    bias_vec = make_bias_vector(args.dim, bias_norm)

    def _generator(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        cands = sample_candidates(
            rng,
            theta_true=theta_true,
            bias=bias_vec,
            n=n_candidates,
            noise_std=args.noise_std,
        )
        return cands, cands.mean(axis=0)

    return _generator


def make_ridge_generator(
    theta_true: np.ndarray,
    alpha: float,
    args: argparse.Namespace,
    n_candidates: int,
    data_size: int,
) -> CandidateGenerator:
    def _generator(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        cands: List[np.ndarray] = []
        for _ in range(n_candidates):
            X = rng.normal(size=(data_size, args.dim))
            y = X @ theta_true + args.noise_std * rng.normal(size=data_size)
            cands.append(ridge_estimator(X, y, alpha=alpha))
        cand_arr = np.stack(cands, axis=0)
        return cand_arr, cand_arr.mean(axis=0)

    return _generator


def make_bayes_generator(
    theta_true: np.ndarray,
    mu_prior: np.ndarray,
    sigma_prior: float,
    args: argparse.Namespace,
    n_candidates: int,
    data_size: int,
) -> CandidateGenerator:
    def _generator(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        cands: List[np.ndarray] = []
        for _ in range(n_candidates):
            X = rng.normal(size=(data_size, args.dim))
            y = X @ theta_true + args.noise_std * rng.normal(size=data_size)
            cands.append(
                bayesian_map_estimator(
                    X,
                    y,
                    mu_prior=mu_prior,
                    sigma_prior=sigma_prior,
                    noise_var=args.noise_std**2,
                )
            )
        cand_arr = np.stack(cands, axis=0)
        return cand_arr, cand_arr.mean(axis=0)

    return _generator


def save_summary_csv(
    param_values: List[float],
    finals: Dict[str, Dict[str, List[float]]],
    param_name: str,
    path: pathlib.Path,
) -> None:
    with path.open("w") as f:
        header = [param_name]
        for m in METHODS:
            header.extend([f"{m}_mean", f"{m}_std"])
        f.write(",".join(header) + "\n")
        for idx, val in enumerate(param_values):
            row = [str(val)]
            for m in METHODS:
                row.append(f"{finals[m]['mean'][idx]:.6f}")
                row.append(f"{finals[m]['std'][idx]:.6f}")
            f.write(",".join(row) + "\n")


def save_trajectories_csv(series: Dict[str, Dict[str, List[float]]], path: pathlib.Path) -> None:
    with path.open("w") as f:
        header = ["generation"]
        for m in METHODS:
            header.extend([f"{m}_mean", f"{m}_std"])
        f.write(",".join(header) + "\n")
        length = len(next(iter(series.values()))["mean"])
        for i in range(length):
            row = [str(i + 1)]
            for m in METHODS:
                row.append(f"{series[m]['mean'][i]:.6f}")
                row.append(f"{series[m]['std'][i]:.6f}")
            f.write(",".join(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Sensitivity to bias magnitude, ridge alpha, and prior strength.")
    # Exp 2.1
    parser.add_argument("--bias-list", nargs="+", type=float, default=[0.1, 0.5, 1.0, 2.0], help="||b_const|| grid for Exp 2.1.")
    parser.add_argument("--samples-const", type=int, default=200, help="Candidates per generation for Exp 2.1.")
    # Exp 2.2
    parser.add_argument(
        "--ridge-alpha-list",
        nargs="+",
        type=float,
        default=[0.1, 1.0, 10.0, 50.0, 100.0, 200.0],
        help="Alpha grid for ridge sensitivity (Exp 2.2).",
    )
    parser.add_argument("--samples-ridge", type=int, default=120, help="Candidates per generation for Exp 2.2.")
    parser.add_argument("--ridge-data-size", type=int, default=50, help="Data size used to fit each ridge estimator.")
    # Exp 2.3
    parser.add_argument(
        "--prior-offset-list",
        nargs="+",
        type=float,
        default=[0.0, 5.0, 10.0, 20.0, 30.0, 40.0],
        help="||delta|| between prior mean and truth for Exp 2.3.",
    )
    parser.add_argument(
        "--bayes-sample-sizes",
        nargs="+",
        type=int,
        default=[50, 5, 3],
        help="Data sizes n for Bayesian MAP draws (Exp 2.3).",
    )
    parser.add_argument("--samples-bayes", type=int, default=120, help="Candidates per generation for Exp 2.3.")
    parser.add_argument("--sigma-prior", type=float, default=0.2, help="Prior std for MAP (small => strong prior).")
    parser.add_argument("--mu-true-bayes", type=float, default=5.0, help="Ground-truth mean for Exp 2.3.")
    # Shared hyper-parameters
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--calibration-size", type=int, default=120)
    parser.add_argument("--standard-contraction", type=float, default=0.4)
    parser.add_argument("--ours-contraction", type=float, default=0.4)
    parser.add_argument("--top-ratio", type=float, default=0.2)
    parser.add_argument(
        "--coreset-ratio",
        type=float,
        default=0.2,
        help="Subset ratio for diversity-only baselines (k-center).",
    )
    parser.add_argument("--tau", type=float, default=50.0)
    parser.add_argument("--lambda-class", type=float, default=0.05, help="Scale classification loss.")
    parser.add_argument("--lambda-contract", type=float, default=1.0)
    parser.add_argument("--lambda-ess", type=float, default=0.01)
    parser.add_argument("--lambda-reg", type=float, default=1e-5)
    parser.add_argument("--warmup-frac", type=float, default=0.5, help="Fraction of steps to warm up with weights ~ uniform and no class/ESS.")
    parser.add_argument(
        "--correction-clip",
        type=float,
        default=0.0,
        help="Clamp ||delta_phi||_2 to limit over-correction (<=0 disables clipping).",
    )
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--tail-window", type=int, default=50, help="Window size for final error averaging.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1088, 2195, 4960],
        help="Multiple seeds for mean/std shading.",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--save-trajectories-csv",
        action="store_true",
        help="If set, save per-config trajectory CSVs for all experiments.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=TABLES_DIR,
        help="Directory to store CSV/JSON outputs.",
    )
    parser.add_argument(
        "--fig-dir",
        type=pathlib.Path,
        default=FIGURES_DIR,
        help="Directory to store figures.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    seeds = args.seeds if args.seeds is not None else [args.seed]
    start_all = time.time()

    # Shared references.
    theta_const = np.ones(args.dim)
    theta_ridge = np.ones(args.dim)
    theta_prior = np.ones(args.dim) * args.mu_true_bayes

    theta_good_const = sample_clean_reference(set_seed(args.seed + 11), theta_const, n=args.calibration_size, noise_std=args.noise_std)
    theta_good_ridge = sample_clean_reference(set_seed(args.seed + 23), theta_ridge, n=args.calibration_size, noise_std=args.noise_std)
    theta_good_prior = sample_clean_reference(set_seed(args.seed + 35), theta_prior, n=args.calibration_size, noise_std=args.noise_std)

    out_dir: pathlib.Path = args.out_dir
    fig_dir: pathlib.Path = args.fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ----------------
    # Exp 2.1: bias magnitude
    t_exp21 = time.time()
    exp21_series: Dict[float, Dict[str, Dict[str, List[float]]]] = {}
    exp21_finals: Dict[str, Dict[str, List[float]]] = {k: {"mean": [], "std": []} for k in METHODS}
    for idx, b in enumerate(args.bias_list):
        gen = make_const_generator(theta_const, b, args, n_candidates=args.samples_const)
        seed_series: List[Dict[str, List[float]]] = []
        for seed in seeds:
            seed_series.append(
                run_filter_process(
                    generator=gen,
                    args=args,
                    theta_true=theta_const,
                    theta_good=theta_good_const,
                    device=device,
                    seed=seed + idx * 31,
                )
            )
        stats = aggregate_seed_series(seed_series)
        exp21_series[b] = stats
        tail_stats = aggregate_tail_stats(seed_series, tail=args.tail_window)
        for m in METHODS:
            exp21_finals[m]["mean"].append(tail_stats[m]["mean"])
            exp21_finals[m]["std"].append(tail_stats[m]["std"])
        if args.save_trajectories_csv:
            bias_dir = out_dir / f"exp2_2.1_bias_{b}"
            bias_dir.mkdir(parents=True, exist_ok=True)
            save_trajectories_csv(stats, bias_dir / "trajectories.csv")

    save_summary_csv(args.bias_list, exp21_finals, "bias", out_dir / "exp2_2.1_bias_summary.csv")
    plot_param_curve(
        args.bias_list,
        exp21_finals,
        xlabel=r"$\|\mathbf{b}_{const}\|_2$",
        title="Exp 2.1: Bias magnitude sensitivity",
        out_path=fig_dir / "exp2_2.1_bias_vs_error.png",
        n_seeds=len(seeds),
    )
    plot_trajectories_grid(exp21_series, param_label="||b||", out_path=fig_dir / "exp2_2.1_trajs_by_bias.png", n_seeds=len(seeds))
    plot_by_method(exp21_series, args.bias_list, param_label="||b||", out_path=fig_dir / "exp2_2.1_trajs_by_method.png", n_seeds=len(seeds))
    dur21 = time.time() - t_exp21
    print(f"Exp 2.1 completed on {device} in {dur21:.1f}s")

    # ----------------
    # Exp 2.2: ridge alpha sensitivity
    t_exp22 = time.time()
    exp22_series: Dict[float, Dict[str, Dict[str, List[float]]]] = {}
    exp22_finals: Dict[str, Dict[str, List[float]]] = {k: {"mean": [], "std": []} for k in METHODS}
    for idx, alpha in enumerate(args.ridge_alpha_list):
        gen = make_ridge_generator(
            theta_true=theta_ridge,
            alpha=alpha,
            args=args,
            n_candidates=args.samples_ridge,
            data_size=args.ridge_data_size,
        )
        seed_series: List[Dict[str, List[float]]] = []
        for seed in seeds:
            seed_series.append(
                run_filter_process(
                    generator=gen,
                    args=args,
                    theta_true=theta_ridge,
                    theta_good=theta_good_ridge,
                    device=device,
                    seed=seed + 500 + idx * 29,
                )
            )
        stats = aggregate_seed_series(seed_series)
        exp22_series[alpha] = stats
        tail_stats = aggregate_tail_stats(seed_series, tail=args.tail_window)
        for m in METHODS:
            exp22_finals[m]["mean"].append(tail_stats[m]["mean"])
            exp22_finals[m]["std"].append(tail_stats[m]["std"])
        if args.save_trajectories_csv:
            alpha_dir = out_dir / f"exp2_2.2_alpha_{alpha}"
            alpha_dir.mkdir(parents=True, exist_ok=True)
            save_trajectories_csv(stats, alpha_dir / "trajectories.csv")

    save_summary_csv(args.ridge_alpha_list, exp22_finals, "alpha", out_dir / "exp2_2.2_ridge_summary.csv")
    plot_param_curve(
        args.ridge_alpha_list,
        exp22_finals,
        xlabel=r"$\alpha$ (ridge)",
        title="Exp 2.2: Ridge regularization sensitivity",
        out_path=fig_dir / "exp2_2.2_alpha_vs_error.png",
        log_x=True,
        n_seeds=len(seeds),
    )
    plot_trajectories_grid(exp22_series, param_label="alpha", out_path=fig_dir / "exp2_2.2_trajs_by_alpha.png", n_seeds=len(seeds))
    plot_by_method(exp22_series, args.ridge_alpha_list, param_label="alpha", out_path=fig_dir / "exp2_2.2_trajs_by_method.png", n_seeds=len(seeds))
    dur22 = time.time() - t_exp22
    print(f"Exp 2.2 completed on {device} in {dur22:.1f}s")

    # ----------------
    # Exp 2.3: prior misspecification x data scarcity
    t_exp23 = time.time()
    exp23_series: Dict[int, Dict[float, Dict[str, Dict[str, List[float]]]]] = {n: {} for n in args.bayes_sample_sizes}
    exp23_finals: Dict[int, Dict[str, Dict[str, List[float]]]] = {
        n: {k: {"mean": [], "std": []} for k in METHODS} for n in args.bayes_sample_sizes
    }
    for n_idx, n_data in enumerate(args.bayes_sample_sizes):
        for delta_idx, delta in enumerate(args.prior_offset_list):
            offset_vec = make_bias_vector(args.dim, delta)
            mu_prior = theta_prior + offset_vec
            gen = make_bayes_generator(
                theta_true=theta_prior,
                mu_prior=mu_prior,
                sigma_prior=args.sigma_prior,
                args=args,
                n_candidates=args.samples_bayes,
                data_size=n_data,
            )
            seed_series: List[Dict[str, List[float]]] = []
            for seed in seeds:
                seed_series.append(
                    run_filter_process(
                        generator=gen,
                        args=args,
                        theta_true=theta_prior,
                        theta_good=theta_good_prior,
                        device=device,
                        seed=seed + 1000 + n_idx * 101 + delta_idx * 7,
                    )
                )
            stats = aggregate_seed_series(seed_series)
            exp23_series[n_data][delta] = stats
            tail_stats = aggregate_tail_stats(seed_series, tail=args.tail_window)
            for m in METHODS:
                exp23_finals[n_data][m]["mean"].append(tail_stats[m]["mean"])
                exp23_finals[n_data][m]["std"].append(tail_stats[m]["std"])
            if args.save_trajectories_csv:
                delta_dir = out_dir / f"exp2_2.3_n{n_data}_delta{delta}"
                delta_dir.mkdir(parents=True, exist_ok=True)
                save_trajectories_csv(stats, delta_dir / "trajectories.csv")

    # Summary plots for prior experiment.
    fig, axes = plt.subplots(1, len(args.bayes_sample_sizes), figsize=(6 * len(args.bayes_sample_sizes), 4), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    for ax, n_data in zip(axes, args.bayes_sample_sizes):
        for m in METHODS:
            means = exp23_finals[n_data][m]["mean"]
            stds = exp23_finals[n_data][m]["std"]
            ci = 1.96 * np.array(stds) / np.sqrt(len(seeds))
            ax.errorbar(
                args.prior_offset_list,
                means,
                yerr=ci,
                marker="o",
                linewidth=2,
                capsize=4,
                label=display_label(m),
            )
        ax.set_title(f"n={n_data}")
        ax.set_xlabel(r"$\|\delta\|$")
        ax.set_ylabel("Final error (mean tail)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Exp 2.3: Prior misspecification x data scarcity")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "exp2_2.3_prior_offset_curves.png", dpi=200)
    plt.close(fig)

    # Trajectories by delta for the hardest regime (smallest n).
    hard_n = min(args.bayes_sample_sizes)
    plot_trajectories_grid(
        exp23_series[hard_n],
        param_label=r"$\delta$",
        out_path=fig_dir / f"exp2_2.3_trajs_n{hard_n}.png",
        suptitle=f"Exp 2.3 trajectories (n={hard_n})",
        n_seeds=len(seeds),
    )

    # CSV summaries for prior experiment.
    for n_data in args.bayes_sample_sizes:
        save_summary_csv(
            args.prior_offset_list,
            exp23_finals[n_data],
            param_name="delta",
            path=out_dir / f"exp2_2.3_prior_summary_n{n_data}.csv",
        )
    dur23 = time.time() - t_exp23
    print(f"Exp 2.3 completed on {device} in {dur23:.1f}s")

    total_time = time.time() - start_all
    runtime = {
        "device": str(device),
        "seeds": list(seeds),
        "exp2_1_time_sec": dur21,
        "exp2_2_time_sec": dur22,
        "exp2_3_time_sec": dur23,
        "total_time_sec": total_time,
        "note": "Exp 2 prior scenario uses theta_true = mu_true_bayes * ones (default 5·1_d).",
    }
    with (out_dir / "runtime_exp2.json").open("w") as f:
        json.dump(runtime, f, indent=2)
    print(f"Saved Experiment 2 results to {out_dir} (tables) and {fig_dir} (figures)")
    print(f"Total runtime on {device}: {total_time:.1f}s")
    print(runtime["note"])


if __name__ == "__main__":
    main()
