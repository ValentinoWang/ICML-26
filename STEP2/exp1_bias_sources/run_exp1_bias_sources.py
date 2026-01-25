import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
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
from filter.data import make_bias_vector, sample_candidates, sample_clean_reference, set_seed
from filter.losses import classification_loss, contraction_loss, correction_reg, ess_loss
from filter.set_aware.model import SetAwareBiasRobustFilter
from filter.standard.model import StandardFilter
from plot_exp1 import plot_scenarios  # noqa: E402


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


def rbf_kernel(points: np.ndarray, sigma: float | None, jitter: float) -> np.ndarray:
    diffs = points[:, None, :] - points[None, :, :]
    dist2 = np.sum(diffs**2, axis=2)
    if sigma is None or sigma <= 0:
        nonzero = dist2[dist2 > 0]
        if nonzero.size == 0:
            sigma = 1.0
        else:
            sigma = float(np.sqrt(np.median(nonzero)))
            if sigma <= 0:
                sigma = 1.0
    kernel = np.exp(-dist2 / (2.0 * sigma**2))
    if jitter > 0:
        kernel = kernel + jitter * np.eye(kernel.shape[0])
    return kernel


def dpp_greedy(L: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = L.shape[0]
    if k >= n:
        return np.arange(n, dtype=int)
    cis = np.zeros((k, n), dtype=float)
    di2s = np.clip(np.diag(L).copy(), 0.0, None)
    di2s = di2s + 1e-12 * rng.random(n)
    selected: List[int] = []
    for it in range(k):
        idx = int(np.argmax(di2s))
        if di2s[idx] <= 1e-12:
            break
        selected.append(idx)
        if it == k - 1:
            break
        if it == 0:
            eis = L[idx, :] / np.sqrt(di2s[idx])
        else:
            proj = cis[:it, idx] @ cis[:it, :]
            eis = (L[idx, :] - proj) / np.sqrt(di2s[idx])
        cis[it, :] = eis
        di2s = di2s - eis**2
        di2s[idx] = -np.inf
    if len(selected) < k:
        remaining = np.argsort(di2s)[::-1]
        for idx in remaining:
            if idx not in selected:
                selected.append(int(idx))
            if len(selected) == k:
                break
    return np.array(selected, dtype=int)


def run_bias_scenario(
    args: argparse.Namespace,
    scenario: str,
    theta_true: np.ndarray,
    bias_vec: np.ndarray,
    samples_per_gen: int,
    std_contraction: float,
    ours_contraction: float,
    ridge_alpha: float,
    sigma_prior: float,
    mu_prior: float,
    device: torch.device,
    seed: int,
) -> Dict[str, List[float]]:
    rng = set_seed(seed)

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
    dst_bias = StandardFilter(dim=args.dim, hidden=args.hidden, dropout=args.dropout).to(device)
    dst_main = StandardFilter(dim=args.dim, hidden=args.hidden, dropout=args.dropout).to(device)
    opt_dst = optim.Adam(
        list(dst_bias.parameters()) + list(dst_main.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    l2ac_model = StandardFilter(dim=args.dim, hidden=args.hidden, dropout=args.dropout).to(device)
    opt_l2ac = optim.Adam(l2ac_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    theta_good = sample_clean_reference(rng, theta_true, n=args.calibration_size, noise_std=args.noise_std)
    theta_std = theta_good.copy()
    theta_hard = theta_good.copy()
    theta_ema = theta_good.copy()
    theta_sa = theta_good.copy()
    theta_dst = theta_good.copy()
    theta_l2ac = theta_good.copy()
    theta_kcenter = theta_good.copy()
    theta_dpp = theta_good.copy()
    theta_good_t = torch.from_numpy(theta_good[None, ...]).float().to(device)

    errors: Dict[str, List[float]] = {
        "no_filter": [],
        "ema": [],
        "standard_filter": [],
        "hard_threshold": [],
        "k_center": [],
        "dpp": [],
        "dst": [],
        "l2ac": [],
        "ours": [],
    }

    for _ in range(args.generations):
        candidates = sample_candidates(
            rng,
            theta_true=theta_true,
            bias=bias_vec,
            n=samples_per_gen,
            noise_std=args.noise_std,
        )
        labels = build_labels(candidates, theta_true, top_ratio=args.top_ratio)

        x = torch.from_numpy(candidates[None, ...]).float().to(device)
        y = torch.from_numpy(labels[None, ...]).float().to(device)

        # Biased estimator depending on scenario
        if scenario == "const":
            theta_hat = const_bias_estimator(candidates.mean(axis=0), bias_vec)
        elif scenario == "ridge":
            X = rng.normal(size=(samples_per_gen, args.dim))
            y_reg = X @ theta_true + args.noise_std * rng.normal(size=samples_per_gen)
            theta_hat = ridge_estimator(X, y_reg, alpha=ridge_alpha)
        elif scenario == "bayes":
            X = rng.normal(size=(samples_per_gen, args.dim))
            y_reg = X @ theta_true + args.noise_std * rng.normal(size=samples_per_gen)
            theta_hat = bayesian_map_estimator(
                X,
                y_reg,
                mu_prior=mu_prior * np.ones(args.dim),
                sigma_prior=sigma_prior,
                noise_var=args.noise_std**2,
            )
        else:
            raise ValueError("unknown scenario")

        errors["no_filter"].append(float(np.linalg.norm(theta_hat - theta_true)))
        theta_ema = args.ema_gamma * theta_ema + (1 - args.ema_gamma) * theta_hat
        errors["ema"].append(float(np.linalg.norm(theta_ema - theta_true)))

        # k-Center (coreset) baseline: select diverse subset and average.
        k = max(1, int(samples_per_gen * args.coreset_ratio))
        k_idx = k_center_greedy(candidates, k, rng)
        theta_kc_new = candidates[k_idx].mean(axis=0)
        theta_kcenter = theta_kcenter + std_contraction * (theta_kc_new - theta_kcenter)
        errors["k_center"].append(float(np.linalg.norm(theta_kcenter - theta_true)))

        # DPP (MAP) baseline: diversity-only selection via kernel determinant.
        L = rbf_kernel(candidates, args.dpp_sigma, args.dpp_jitter)
        dpp_idx = dpp_greedy(L, k, rng)
        theta_dpp_new = candidates[dpp_idx].mean(axis=0)
        theta_dpp = theta_dpp + std_contraction * (theta_dpp_new - theta_dpp)
        errors["dpp"].append(float(np.linalg.norm(theta_dpp - theta_true)))

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
        theta_std = theta_std + std_contraction * (
            theta_w_std.detach().cpu().numpy().squeeze(0) - theta_std
        )
        errors["standard_filter"].append(float(np.linalg.norm(theta_std - theta_true)))

        # Hard-thresholded estimation: binarize top-k weights, still without correction.
        with torch.no_grad():
            k = max(1, int(w_std.shape[1] * args.hard_top_ratio))
            top_idx = torch.topk(w_std, k, dim=1).indices
            w_hard = torch.zeros_like(w_std)
            w_hard.scatter_(1, top_idx, 1.0)
            theta_w_hard = StandardFilter.weighted_estimate(x, w_hard)
        theta_hard = theta_hard + std_contraction * (
            theta_w_hard.detach().cpu().numpy().squeeze(0) - theta_hard
        )
        errors["hard_threshold"].append(float(np.linalg.norm(theta_hard - theta_true)))

        # DST baseline: bias head learns biased estimate, main head downweights bias-heavy samples.
        theta_hat_t = torch.from_numpy(theta_hat[None, ...]).float().to(device)
        w_bias = dst_bias(x)
        theta_bias = StandardFilter.weighted_estimate(x, w_bias)
        loss_bias = contraction_loss(theta_bias, theta_hat_t) + args.lambda_ess * ess_loss(w_bias, tau=args.tau)

        w_main = dst_main(x)
        bias_mask = (1.0 - w_bias.detach()).clamp_min(0.05)
        w_dst = (w_main * bias_mask).clamp_min(1e-4)
        theta_dst_new = StandardFilter.weighted_estimate(x, w_dst)
        loss_dst = (
            args.lambda_class * classification_loss(w_main, y)
            + args.lambda_contract * contraction_loss(theta_dst_new, theta_good_t)
            + args.lambda_ess * ess_loss(w_main, tau=args.tau)
        )
        opt_dst.zero_grad()
        (loss_bias + loss_dst).backward()
        opt_dst.step()
        theta_dst = theta_dst + std_contraction * (
            theta_dst_new.detach().cpu().numpy().squeeze(0) - theta_dst
        )
        errors["dst"].append(float(np.linalg.norm(theta_dst - theta_true)))

        # L2AC-style baseline: meta-aligned weighting using the clean reference direction.
        w_l2ac_raw = l2ac_model(x)
        theta_l2ac_t = torch.from_numpy(theta_l2ac[None, ...]).float().to(device)
        x_flat = x.squeeze(0)
        theta_l2ac_vec = theta_l2ac_t.squeeze()
        delta_dir = (theta_good_t - theta_l2ac_t).squeeze()
        align_score = F.relu((x_flat - theta_l2ac_vec) @ delta_dir).unsqueeze(0)
        w_l2ac = (w_l2ac_raw * (align_score + 1e-6)).clamp_min(1e-4)
        theta_l2ac_new = StandardFilter.weighted_estimate(x, w_l2ac)
        loss_l2ac = (
            args.lambda_class * classification_loss(w_l2ac_raw, y)
            + args.lambda_contract * contraction_loss(theta_l2ac_new, theta_good_t)
            + args.lambda_ess * ess_loss(w_l2ac_raw, tau=args.tau)
        )
        opt_l2ac.zero_grad()
        loss_l2ac.backward()
        opt_l2ac.step()
        theta_l2ac = theta_l2ac + std_contraction * (
            theta_l2ac_new.detach().cpu().numpy().squeeze(0) - theta_l2ac
        )
        errors["l2ac"].append(float(np.linalg.norm(theta_l2ac - theta_true)))

        # Set-aware filter: train weights as auxiliary, estimation uses correction only (w=1)
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
        theta_sa = theta_sa + ours_contraction * (
            theta_new_sa.detach().cpu().numpy().squeeze(0) - theta_sa
        )
        errors["ours"].append(float(np.linalg.norm(theta_sa - theta_true)))

    return errors


def save_series(series: Dict[str, Dict], path: pathlib.Path) -> None:
    cfg_names = list(series.keys())
    first_cfg = next(iter(series.values()))
    first_series_len = len(next(iter(first_cfg["series"].values()))["mean"])
    gens = np.arange(1, first_series_len + 1)
    with path.open("w") as f:
        header = ["generation"]
        for cfg in cfg_names:
            for k in series[cfg]["series"]:
                header.extend([f"{k}_{cfg}_mean", f"{k}_{cfg}_std"])
        f.write(",".join(header) + "\n")
        for i in range(len(gens)):
            row = [str(gens[i])]
            for cfg in cfg_names:
                for k in series[cfg]["series"]:
                    row.append(f"{series[cfg]['series'][k]['mean'][i]:.6f}")
                    row.append(f"{series[cfg]['series'][k]['std'][i]:.6f}")
            f.write(",".join(row) + "\n")


def aggregate_seed_series(seed_series: List[Dict[str, List[float]]]) -> Dict[str, Dict[str, List[float]]]:
    methods = seed_series[0].keys()
    stats: Dict[str, Dict[str, List[float]]] = {}
    for m in methods:
        arr = np.array([s[m] for s in seed_series])
        stats[m] = {"mean": arr.mean(axis=0).tolist(), "std": arr.std(axis=0).tolist()}
    return stats


def main():
    parser = argparse.ArgumentParser(description="Experiment 1 variants: different bias sources.")
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--bias-norm", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--samples-per-gen", type=int, default=200)
    parser.add_argument("--calibration-size", type=int, default=200)
    parser.add_argument("--standard-contraction", type=float, default=0.2)
    parser.add_argument("--ours-contraction", type=float, default=0.5)
    parser.add_argument("--top-ratio", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=50.0)
    parser.add_argument("--lambda-class", type=float, default=0.05)
    parser.add_argument("--lambda-contract", type=float, default=1.0)
    parser.add_argument("--lambda-ess", type=float, default=0.1)
    parser.add_argument("--lambda-reg", type=float, default=1e-5)
    parser.add_argument("--ema-gamma", type=float, default=0.9, help="EMA momentum on the biased estimator.")
    parser.add_argument(
        "--hard-top-ratio",
        type=float,
        default=0.5,
        help="Fraction of candidates kept by hard-thresholding (Top-K) baseline.",
    )
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--coreset-ratio",
        type=float,
        default=0.2,
        help="Subset ratio for diversity-only baselines (k-center/DPP).",
    )
    parser.add_argument(
        "--dpp-sigma",
        type=float,
        default=0.0,
        help="RBF kernel bandwidth for DPP; 0 uses median pairwise distance.",
    )
    parser.add_argument(
        "--dpp-jitter",
        type=float,
        default=1e-6,
        help="Diagonal jitter for DPP kernel stability.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=20.0)
    parser.add_argument("--mu-prior", type=float, default=0.0)
    parser.add_argument("--sigma-prior", type=float, default=0.2)
    parser.add_argument("--mu-true-bayes", type=float, default=5.0, help="True mean for wrong prior scenario.")
    parser.add_argument("--samples-const", type=int, default=200)
    parser.add_argument("--samples-ridge", type=int, default=100)
    parser.add_argument("--samples-bayes", type=int, default=20, help="Smaller n to emphasize prior impact.")
    parser.add_argument(
        "--scenarios",
        type=str,
        default="const,ridge,bayes",
        help="Comma-separated list of scenarios to run: const,ridge,bayes.",
    )
    # For Exp1.1 keep a single bias value; list kept for extensibility.
    parser.add_argument("--const-bias-list", type=float, nargs="+", default=None)
    parser.add_argument("--ridge-alpha-list", type=float, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1088, 2195, 4960, 1545, 3549, 1440, 3050, 5414],
        help="Multiple seeds for mean/std bands.",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-csv", action="store_true")
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
    seeds: Sequence[int] = args.seeds if args.seeds is not None else [args.seed]
    start_time = time.time()

    # Scenario-specific configs
    theta_const = np.ones(args.dim)
    theta_ridge = np.ones(args.dim)
    theta_bayes = np.ones(args.dim) * args.mu_true_bayes

    bias_zero = np.zeros(args.dim)

    summaries: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    # Multiple configs per scenario to plot several curves.
    const_bias_list = args.const_bias_list if args.const_bias_list is not None else [args.bias_norm]
    const_cfgs = {
        f"b{b}": {
            "samples": args.samples_const,
            "std_c": args.standard_contraction,
            "ours_c": args.ours_contraction,
            "ridge_alpha": args.ridge_alpha,
            "sigma_prior": args.sigma_prior,
            "bias_norm": b,
            "label": f"b={b}",
        }
        for b in const_bias_list
    }
    ridge_alpha_list = args.ridge_alpha_list if args.ridge_alpha_list is not None else [args.ridge_alpha]
    ridge_cfgs = {
        f"alpha{int(a)}": {
            "samples": args.samples_ridge,
            "std_c": args.standard_contraction,
            "ours_c": args.ours_contraction,
            "ridge_alpha": a,
            "sigma_prior": args.sigma_prior,
            "bias_norm": args.bias_norm,
            "label": f"alpha={a}",
        }
        for a in ridge_alpha_list
    }
    bayes_cfgs = {
        f"sig{args.sigma_prior}_n{args.samples_bayes}": {
            "samples": args.samples_bayes,
            "std_c": args.standard_contraction,
            "ours_c": args.ours_contraction,
            "ridge_alpha": args.ridge_alpha,
            "sigma_prior": args.sigma_prior,
            "bias_norm": args.bias_norm,
            "label": f"sigma={args.sigma_prior}, n={args.samples_bayes}",
        }
    }

    scenario_set = {s.strip() for s in args.scenarios.split(",") if s.strip()}

    if "const" in scenario_set:
        summaries["const"] = {}
        for cfg_name, cfg in const_cfgs.items():
            bias_const = make_bias_vector(args.dim, cfg["bias_norm"])
            seed_series: List[Dict[str, List[float]]] = []
            for s in seeds:
                seed_series.append(
                    run_bias_scenario(
                        args,
                        scenario="const",
                        theta_true=theta_const,
                        bias_vec=bias_const,
                        samples_per_gen=cfg["samples"],
                        std_contraction=cfg["std_c"],
                        ours_contraction=cfg["ours_c"],
                        ridge_alpha=cfg["ridge_alpha"],
                        sigma_prior=cfg["sigma_prior"],
                        mu_prior=args.mu_prior,
                        device=device,
                        seed=s,
                    )
                )
            summaries["const"][cfg_name] = {"series": aggregate_seed_series(seed_series), "label": cfg["label"]}

    if "ridge" in scenario_set:
        summaries["ridge"] = {}
        for cfg_name, cfg in ridge_cfgs.items():
            seed_series = []
            for s in seeds:
                seed_series.append(
                    run_bias_scenario(
                        args,
                        scenario="ridge",
                        theta_true=theta_ridge,
                        bias_vec=bias_zero,
                        samples_per_gen=cfg["samples"],
                        std_contraction=cfg["std_c"],
                        ours_contraction=cfg["ours_c"],
                        ridge_alpha=cfg["ridge_alpha"],
                        sigma_prior=cfg["sigma_prior"],
                        mu_prior=args.mu_prior,
                        device=device,
                        seed=s,
                    )
                )
            summaries["ridge"][cfg_name] = {"series": aggregate_seed_series(seed_series), "label": cfg["label"]}

    if "bayes" in scenario_set:
        summaries["bayes"] = {}
        for cfg_name, cfg in bayes_cfgs.items():
            seed_series = []
            for s in seeds:
                seed_series.append(
                    run_bias_scenario(
                        args,
                        scenario="bayes",
                        theta_true=theta_bayes,
                        bias_vec=bias_zero,
                        samples_per_gen=cfg["samples"],
                        std_contraction=cfg["std_c"],
                        ours_contraction=cfg["ours_c"],
                        ridge_alpha=cfg["ridge_alpha"],
                        sigma_prior=cfg["sigma_prior"],
                        mu_prior=args.mu_prior,
                        device=device,
                        seed=s,
                    )
                )
            summaries["bayes"][cfg_name] = {"series": aggregate_seed_series(seed_series), "label": cfg["label"]}

    out_dir: pathlib.Path = args.out_dir
    fig_dir: pathlib.Path = args.fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    subtitles = {
        "const": "Exp 1.1 Hard-coded bias",
        "ridge": "Exp 1.2 Ridge bias",
        "bayes": "Exp 1.3 Wrong prior",
    }
    plot_scenarios(summaries, fig_dir, subtitles, n_seeds=len(seeds))

    if args.save_csv:
        name_map = {"const": "exp1_1.1_const", "ridge": "exp1_1.2_ridge", "bayes": "exp1_1.3_bayes"}
        for name, series in summaries.items():
            save_series(series, out_dir / f"{name_map.get(name, name)}.csv")

    runtime = {"device": str(device), "seeds": list(seeds), "total_time_sec": time.time() - start_time}
    with (out_dir / "runtime_exp1.json").open("w") as f:
        json.dump(runtime, f, indent=2)

    print(f"Saved bias source experiments to {out_dir} (tables) and {fig_dir} (figures)")


if __name__ == "__main__":
    main()
