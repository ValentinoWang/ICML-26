import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List, Tuple, Sequence

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

from bias.bias_sources import ridge_estimator  # noqa: E402
from filter.data import make_bias_vector, sample_candidates, sample_clean_reference, set_seed  # noqa: E402
from filter.losses import classification_loss, contraction_loss, correction_reg, ess_loss  # noqa: E402
from filter.set_aware.model import SetAwareBiasRobustFilter  # noqa: E402
from plot_exp4 import plot_exp41, plot_exp42, plot_exp43  # noqa: E402


def build_labels(candidates: np.ndarray, theta_true: np.ndarray, top_ratio: float) -> np.ndarray:
    dists = np.linalg.norm(candidates - theta_true, axis=1)
    k = max(1, int(len(candidates) * top_ratio))
    thresh = np.partition(dists, k - 1)[k - 1]
    return (dists <= thresh).astype(np.float32)


def clip_delta(delta_phi: torch.Tensor, clip_val: float) -> torch.Tensor:
    if clip_val is None or clip_val <= 0:
        return delta_phi
    norm = delta_phi.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    clamp = torch.clamp(clip_val / norm, max=1.0)
    return delta_phi * clamp


def run_exp41_base(args: argparse.Namespace, device: torch.device, seed: int) -> Tuple[Dict[str, List[float]], float]:
    """
    Exp 4.1: visualize Δφ → -b_const.
    """
    rng = set_seed(seed)
    theta_true = np.ones(args.dim)
    bias = make_bias_vector(args.dim, args.bias_norm)
    target_neg_bias = -bias
    target_norm = float(np.linalg.norm(target_neg_bias))

    model = SetAwareBiasRobustFilter(
        dim=args.dim,
        hidden=args.hidden,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    theta_good = sample_clean_reference(rng, theta_true, n=args.calibration_size, noise_std=args.noise_std)
    theta_sa = theta_good.copy()
    theta_good_t = torch.from_numpy(theta_good[None, ...]).float().to(device)

    metrics: Dict[str, List[float]] = {"dist": [], "cos": [], "delta_norm": [], "error": [], "ess": [], "w_var": []}

    for _ in range(args.generations):
        candidates = sample_candidates(
            rng,
            theta_true=theta_true,
            bias=bias,
            n=args.samples_per_gen,
            noise_std=args.noise_std,
        )
        labels = build_labels(candidates, theta_true, top_ratio=args.top_ratio)

        x = torch.from_numpy(candidates[None, ...]).float().to(device)
        y = torch.from_numpy(labels[None, ...]).float().to(device)

        w_sa, delta_phi = model(x)
        delta_phi = clip_delta(delta_phi, args.correction_clip)

        theta_new_sa = delta_phi

        loss_sa = (
            args.lambda_class * classification_loss(w_sa, y)
            + args.lambda_contract * contraction_loss(theta_new_sa, theta_good_t)
            + args.lambda_ess * ess_loss(w_sa, tau=args.tau)
            + args.lambda_reg * correction_reg(delta_phi)
        )
        optimizer.zero_grad()
        loss_sa.backward()
        optimizer.step()

        theta_sa = theta_sa + args.ours_contraction * (
            theta_new_sa.detach().cpu().numpy().squeeze(0) - theta_sa
        )
        metrics["error"].append(float(np.linalg.norm(theta_sa - theta_true)))

        delta_np = delta_phi.detach().cpu().numpy().squeeze(0)
        metrics["delta_norm"].append(float(np.linalg.norm(delta_np)))
        metrics["dist"].append(float(np.linalg.norm(delta_np - target_neg_bias)))
        cos = float(np.dot(delta_np, target_neg_bias) / (np.linalg.norm(delta_np) * target_norm + 1e-8))
        metrics["cos"].append(cos)
        w_np = w_sa.detach().cpu().numpy().squeeze(0)
        ess = (w_np.sum() ** 2) / (np.sum(w_np**2) + 1e-8)
        metrics["ess"].append(float(ess))
        metrics["w_var"].append(float(np.var(w_np)))

    return metrics, target_norm


def save_exp41_csv(metrics: Dict[str, Dict[str, List[float]]], path: pathlib.Path) -> None:
    gens = np.arange(1, len(metrics["delta_norm"]["mean"]) + 1)
    with path.open("w") as f:
        header = [
            "generation",
            "error_mean",
            "error_std",
            "delta_norm_mean",
            "delta_norm_std",
            "dist_to_minus_b_mean",
            "dist_to_minus_b_std",
            "cos_to_minus_b_mean",
            "cos_to_minus_b_std",
            "ess_mean",
            "ess_std",
            "w_variance_mean",
            "w_variance_std",
        ]
        f.write(",".join(header) + "\n")
        for i in range(len(gens)):
            row = [
                str(gens[i]),
                f"{metrics['error']['mean'][i]:.6f}",
                f"{metrics['error']['std'][i]:.6f}",
                f"{metrics['delta_norm']['mean'][i]:.6f}",
                f"{metrics['delta_norm']['std'][i]:.6f}",
                f"{metrics['dist']['mean'][i]:.6f}",
                f"{metrics['dist']['std'][i]:.6f}",
                f"{metrics['cos']['mean'][i]:.6f}",
                f"{metrics['cos']['std'][i]:.6f}",
                f"{metrics['ess']['mean'][i]:.6f}",
                f"{metrics['ess']['std'][i]:.6f}",
                f"{metrics['w_var']['mean'][i]:.6f}",
                f"{metrics['w_var']['std'][i]:.6f}",
            ]
            f.write(",".join(row) + "\n")


def run_exp42_ridge(args: argparse.Namespace, device: torch.device, seed: int) -> Dict[str, List[float]]:
    """
    Exp 4.2: Ridge-style state-dependent bias. Track how Δφ changes with θ_t.
    """
    rng = set_seed(seed)
    model = SetAwareBiasRobustFilter(
        dim=args.dim,
        hidden=args.hidden,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    theta_sa = np.ones(args.dim) * args.ridge_theta_min

    scales = np.linspace(args.ridge_theta_min, args.ridge_theta_max, args.generations)
    metrics: Dict[str, List[float]] = {
        "theta_norm": [],
        "delta_norm": [],
        "bias_norm": [],
        "cos_to_theta": [],
        "cos_to_correction": [],
        "error": [],
        "ess": [],
        "w_var": [],
    }

    for scale in scales:
        theta_t = np.ones(args.dim) * scale
        theta_good = sample_clean_reference(rng, theta_t, n=args.calibration_size, noise_std=args.noise_std)
        theta_good_t = torch.from_numpy(theta_good[None, ...]).float().to(device)

        X = rng.normal(size=(args.ridge_samples, args.dim))
        y = X @ theta_t + args.noise_std * rng.normal(size=args.ridge_samples)
        theta_hat = ridge_estimator(X, y, alpha=args.ridge_alpha_dyn)

        candidates = theta_hat + args.ridge_candidate_noise * rng.normal(size=(args.samples_per_gen, args.dim))
        labels = build_labels(candidates, theta_t, top_ratio=args.top_ratio)

        x = torch.from_numpy(candidates[None, ...]).float().to(device)
        y_t = torch.from_numpy(labels[None, ...]).float().to(device)

        w_sa, delta_phi = model(x)
        delta_phi = clip_delta(delta_phi, args.correction_clip)
        theta_new_sa = delta_phi

        loss_sa = (
            args.lambda_class * classification_loss(w_sa, y_t)
            + args.lambda_contract * contraction_loss(theta_new_sa, theta_good_t)
            + args.lambda_ess * ess_loss(w_sa, tau=args.tau)
            + args.lambda_reg * correction_reg(delta_phi)
        )
        optimizer.zero_grad()
        loss_sa.backward()
        optimizer.step()

        theta_sa = theta_sa + args.ridge_ours_contraction * (
            theta_new_sa.detach().cpu().numpy().squeeze(0) - theta_sa
        )

        delta_np = delta_phi.detach().cpu().numpy().squeeze(0)
        bias_vec = theta_hat - theta_t
        correction_target = theta_t - theta_hat
        metrics["theta_norm"].append(float(np.linalg.norm(theta_t)))
        metrics["delta_norm"].append(float(np.linalg.norm(delta_np)))
        metrics["bias_norm"].append(float(np.linalg.norm(bias_vec)))
        metrics["cos_to_theta"].append(
            float(np.dot(delta_np, theta_t) / (np.linalg.norm(delta_np) * np.linalg.norm(theta_t) + 1e-8))
        )
        metrics["cos_to_correction"].append(
            float(
                np.dot(delta_np, correction_target)
                / (np.linalg.norm(delta_np) * np.linalg.norm(correction_target) + 1e-8)
            )
        )
        metrics["error"].append(float(np.linalg.norm(theta_sa - theta_t)))
        w_np = w_sa.detach().cpu().numpy().squeeze(0)
        ess = (w_np.sum() ** 2) / (np.sum(w_np**2) + 1e-8)
        metrics["ess"].append(float(ess))
        metrics["w_var"].append(float(np.var(w_np)))

    return metrics


def save_exp42_csv(metrics: Dict[str, Dict[str, List[float]]], path: pathlib.Path) -> None:
    gens = np.arange(1, len(metrics["theta_norm"]["mean"]) + 1)
    with path.open("w") as f:
        header = [
            "generation",
            "theta_norm_mean",
            "theta_norm_std",
            "delta_norm_mean",
            "delta_norm_std",
            "bias_norm_mean",
            "bias_norm_std",
            "cos_to_theta_mean",
            "cos_to_theta_std",
            "cos_to_correction_mean",
            "cos_to_correction_std",
            "error_mean",
            "error_std",
            "ess_mean",
            "ess_std",
            "w_variance_mean",
            "w_variance_std",
        ]
        f.write(",".join(header) + "\n")
        for i in range(len(gens)):
            row = [
                str(gens[i]),
                f"{metrics['theta_norm']['mean'][i]:.6f}",
                f"{metrics['theta_norm']['std'][i]:.6f}",
                f"{metrics['delta_norm']['mean'][i]:.6f}",
                f"{metrics['delta_norm']['std'][i]:.6f}",
                f"{metrics['bias_norm']['mean'][i]:.6f}",
                f"{metrics['bias_norm']['std'][i]:.6f}",
                f"{metrics['cos_to_theta']['mean'][i]:.6f}",
                f"{metrics['cos_to_theta']['std'][i]:.6f}",
                f"{metrics['cos_to_correction']['mean'][i]:.6f}",
                f"{metrics['cos_to_correction']['std'][i]:.6f}",
                f"{metrics['error']['mean'][i]:.6f}",
                f"{metrics['error']['std'][i]:.6f}",
                f"{metrics['ess']['mean'][i]:.6f}",
                f"{metrics['ess']['std'][i]:.6f}",
                f"{metrics['w_var']['mean'][i]:.6f}",
                f"{metrics['w_var']['std'][i]:.6f}",
            ]
            f.write(",".join(row) + "\n")


def sample_bayes_candidates(
    rng: np.random.Generator,
    n: int,
    true_mean: float,
    prior_mean: float,
    noise_std: float,
    prior_frac: float,
) -> np.ndarray:
    n_prior = int(n * prior_frac)
    n_true = n - n_prior
    prior_block = prior_mean + noise_std * rng.normal(size=(n_prior, 1))
    true_block = true_mean + noise_std * rng.normal(size=(n_true, 1))
    return np.concatenate([prior_block, true_block], axis=0)


def run_exp43_bayes(
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    eval_candidates: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exp 4.3: scatter of x vs weight under wrong prior.
    """
    rng = set_seed(seed)
    dim = 1
    theta_true = np.array([args.bayes_true_mean])
    theta_good = sample_clean_reference(rng, theta_true, n=args.calibration_size, noise_std=args.bayes_noise_std)

    model = SetAwareBiasRobustFilter(
        dim=dim,
        hidden=args.hidden,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    theta_state = theta_good.copy()

    for _ in range(args.bayes_generations):
        candidates = sample_bayes_candidates(
            rng,
            n=args.bayes_candidates,
            true_mean=args.bayes_true_mean,
            prior_mean=args.mu_prior,
            noise_std=args.bayes_noise_std,
            prior_frac=args.bayes_prior_frac,
        )
        labels = build_labels(candidates, theta_true, top_ratio=args.bayes_top_ratio)
        x = torch.from_numpy(candidates[None, ...]).float().to(device)
        y = torch.from_numpy(labels[None, ...]).float().to(device)

        w_sa, delta_phi = model(x)
        delta_phi = clip_delta(delta_phi, args.correction_clip)
        theta_new = delta_phi
        theta_good_t = torch.from_numpy(theta_good[None, ...]).float().to(device)

        loss_sa = (
            args.lambda_class * classification_loss(w_sa, y)
            + args.lambda_contract * contraction_loss(theta_new, theta_good_t)
            + args.lambda_ess * ess_loss(w_sa, tau=args.tau)
            + args.lambda_reg * correction_reg(delta_phi)
        )
        optimizer.zero_grad()
        loss_sa.backward()
        optimizer.step()

        theta_state = theta_state + args.ours_contraction * (
            theta_new.detach().cpu().numpy().squeeze(0) - theta_state
        )

    model.eval()
    with torch.no_grad():
        eval_cands = (
            eval_candidates
            if eval_candidates is not None
            else sample_bayes_candidates(
                rng,
                n=args.bayes_eval_candidates,
                true_mean=args.bayes_true_mean,
                prior_mean=args.mu_prior,
                noise_std=args.bayes_noise_std,
                prior_frac=args.bayes_prior_frac,
            )
        )
        eval_x = torch.from_numpy(eval_cands[None, ...]).float().to(device)
        w_eval, _ = model(eval_x)
        weights = w_eval.detach().cpu().numpy().reshape(-1)
    return (eval_cands.reshape(-1), weights)


def save_exp43_csv(x_eval: np.ndarray, w_mean: np.ndarray, w_std: np.ndarray, path: pathlib.Path) -> None:
    with path.open("w") as f:
        f.write("x,weight_mean,weight_std\n")
        for x, m, s in zip(x_eval, w_mean, w_std):
            f.write(f"{float(x):.6f},{float(m):.6f},{float(s):.6f}\n")


def aggregate_metrics_list(metrics_list: List[Dict[str, List[float]]]) -> Dict[str, Dict[str, List[float]]]:
    keys = metrics_list[0].keys()
    agg: Dict[str, Dict[str, List[float]]] = {}
    for k in keys:
        arr = np.array([m[k] for m in metrics_list])
        agg[k] = {"mean": arr.mean(axis=0).tolist(), "std": arr.std(axis=0).tolist()}
    return agg


def main():
    parser = argparse.ArgumentParser(description="Experiment 4.x: bias correction visualizations.")
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--bias-norm", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--samples-per-gen", type=int, default=160)
    parser.add_argument("--calibration-size", type=int, default=120)
    parser.add_argument("--ours-contraction", type=float, default=0.35)
    parser.add_argument("--ridge-ours-contraction", type=float, default=0.4)
    parser.add_argument("--top-ratio", type=float, default=0.3)
    parser.add_argument("--tau", type=float, default=50.0)
    parser.add_argument("--lambda-contract", type=float, default=1.0)
    parser.add_argument("--lambda-class", type=float, default=0.05)
    parser.add_argument("--lambda-ess", type=float, default=0.1)
    parser.add_argument("--lambda-reg", type=float, default=1e-5)
    parser.add_argument("--correction-clip", type=float, default=0.5)
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
    parser.add_argument("--eval-seed", type=int, default=123, help="Seed to generate shared eval candidates for Exp4.3.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-csv", action="store_true")

    # Ridge dynamics specific
    parser.add_argument("--ridge-alpha-dyn", type=float, default=6.0)
    parser.add_argument("--ridge-theta-min", type=float, default=0.4)
    parser.add_argument("--ridge-theta-max", type=float, default=2.5)
    parser.add_argument("--ridge-samples", type=int, default=80)
    parser.add_argument("--ridge-candidate-noise", type=float, default=0.08)

    # Bayesian weighting specific
    parser.add_argument("--mu-prior", type=float, default=0.0)
    parser.add_argument("--sigma-prior", type=float, default=0.2, help="Kept for consistency; not directly used here.")
    parser.add_argument("--bayes-true-mean", type=float, default=5.0)
    parser.add_argument("--bayes-noise-std", type=float, default=0.35)
    parser.add_argument("--bayes-prior-frac", type=float, default=0.45)
    parser.add_argument("--bayes-top-ratio", type=float, default=0.5)
    parser.add_argument("--bayes-generations", type=int, default=250)
    parser.add_argument("--bayes-candidates", type=int, default=80)
    parser.add_argument("--bayes-eval-candidates", type=int, default=200)

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
    start_time = time.time()

    out_dir: pathlib.Path = args.out_dir
    fig_dir: pathlib.Path = args.fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    seeds: Sequence[int] = args.seeds if args.seeds is not None else [args.seed]

    # Exp 4.1
    metrics_41_list: List[Dict[str, List[float]]] = []
    for s in seeds:
        m, target_norm = run_exp41_base(args, device, seed=s)
        metrics_41_list.append(m)
    metrics_41 = aggregate_metrics_list(metrics_41_list)
    plot_exp41(metrics_41, target_norm, fig_dir, n_seeds=len(seeds))
    if args.save_csv:
        save_exp41_csv(metrics_41, out_dir / "exp4_4.1_base.csv")

    # Exp 4.2
    metrics_42_list: List[Dict[str, List[float]]] = []
    for s in seeds:
        metrics_42_list.append(run_exp42_ridge(args, device, seed=s))
    metrics_42 = aggregate_metrics_list(metrics_42_list)
    plot_exp42(metrics_42, fig_dir, n_seeds=len(seeds))
    if args.save_csv:
        save_exp42_csv(metrics_42, out_dir / "exp4_4.2_ridge.csv")

    # Exp 4.3 (aggregate weights across seeds on a shared eval set)
    rng_eval = np.random.default_rng(args.eval_seed)
    shared_eval = sample_bayes_candidates(
        rng_eval,
        n=args.bayes_eval_candidates,
        true_mean=args.bayes_true_mean,
        prior_mean=args.mu_prior,
        noise_std=args.bayes_noise_std,
        prior_frac=args.bayes_prior_frac,
    )
    weights_list = []
    for s in seeds:
        _, w_eval = run_exp43_bayes(args, device, seed=s, eval_candidates=shared_eval)
        weights_list.append(w_eval)
    weights_arr = np.stack(weights_list, axis=0)
    weights_mean = weights_arr.mean(axis=0)
    weights_std = weights_arr.std(axis=0)
    plot_exp43(shared_eval.reshape(-1), weights_mean, args, fig_dir)
    if args.save_csv:
        save_exp43_csv(shared_eval.reshape(-1), weights_mean, weights_std, out_dir / "exp4_4.3_bayes_scatter.csv")

    runtime = {"device": str(device), "seeds": list(seeds), "total_time_sec": time.time() - start_time}
    with (out_dir / "runtime_exp4.json").open("w") as f:
        json.dump(runtime, f, indent=2)

    print(f"Saved Exp4.1–4.3 visualizations to {out_dir} (tables) and {fig_dir} (figures)")


if __name__ == "__main__":
    main()
