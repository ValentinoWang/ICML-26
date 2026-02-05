import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List, Sequence, Tuple

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

from bias.bias_sources import bayesian_map_estimator, ridge_estimator  # noqa: E402
from filter.data import sample_clean_reference, set_seed  # noqa: E402
from filter.losses import classification_loss, contraction_loss, correction_reg, ess_loss  # noqa: E402
from filter.set_aware.model import SetAwareBiasRobustFilter  # noqa: E402
from filter.standard.model import StandardFilter  # noqa: E402
from plot_exp6_gated import plot_tail_bar, plot_trajectories  # noqa: E402

METHODS = ["only_weight", "only_correction", "mlp_correction", "ours_full"]


def build_labels(candidates: np.ndarray, theta_true: np.ndarray, top_ratio: float) -> np.ndarray:
    dists = np.linalg.norm(candidates - theta_true, axis=1)
    k = max(1, int(len(candidates) * top_ratio))
    thresh = np.partition(dists, k - 1)[k - 1]
    return (dists <= thresh).astype(np.float32)


class CorrectionMLP(nn.Module):
    """Simple MLP backbone (no set interaction) to predict global correction."""

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
        # x shape: [B, N, D]; pool by mean
        pooled = x.mean(dim=1)
        return self.net(pooled).unsqueeze(1)  # [B,1,D] for broadcast consistency


def generate_complex_bias_candidates(
    rng: np.random.Generator,
    theta_true: np.ndarray,
    samples_per_gen: int,
    base_noise_std: float,
    noise_scale_std: float,
    beta: float,
    outlier_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complex structural bias: bias magnitude grows with dispersion of the candidate set.
    Construct a dense core plus biased outliers whose shift scales with the core dispersion:
    shift = beta * log(1 + Var(core)).
    """
    d = theta_true.shape[0]
    n_out = max(1, int(samples_per_gen * outlier_ratio))
    n_core = max(1, samples_per_gen - n_out)
    n_out = samples_per_gen - n_core
    noise_scale = rng.lognormal(mean=0.0, sigma=noise_scale_std)
    core_noise = base_noise_std * noise_scale
    core = theta_true + core_noise * rng.normal(size=(n_core, d))
    core_mean = core.mean(axis=0)
    core_var = float(np.mean(np.sum((core - core_mean) ** 2, axis=1)))
    shift = beta * np.log1p(core_var)
    out_noise = core_noise * (1.5 + noise_scale)
    out = theta_true + shift * np.ones(d) + out_noise * rng.normal(size=(n_out, d))
    cands = np.concatenate([core, out], axis=0)
    rng.shuffle(cands)
    return cands, cands.mean(axis=0)


def generate_candidates(
    rng: np.random.Generator,
    scenario: str,
    theta_true: np.ndarray,
    samples_per_gen: int,
    noise_std: float,
    ridge_alpha: float,
    bayes_mu_prior: float,
    bayes_sigma_prior: float,
    bayes_n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    d = theta_true.shape[0]
    if scenario == "ridge":
        cands = []
        for _ in range(samples_per_gen):
            X = rng.normal(size=(bayes_n, d))  # reuse bayes_n as data size, small to emphasize bias
            y = X @ theta_true + noise_std * rng.normal(size=bayes_n)
            cands.append(ridge_estimator(X, y, alpha=ridge_alpha))
        cands = np.stack(cands, axis=0)
    elif scenario == "bayes":
        cands = []
        mu_prior = np.ones(d) * bayes_mu_prior
        for _ in range(samples_per_gen):
            X = rng.normal(size=(bayes_n, d))
            y = X @ theta_true + noise_std * rng.normal(size=bayes_n)
            cands.append(
                bayesian_map_estimator(
                    X,
                    y,
                    mu_prior=mu_prior,
                    sigma_prior=bayes_sigma_prior,
                    noise_var=noise_std**2,
                )
            )
        cands = np.stack(cands, axis=0)
    else:
        raise ValueError("unknown scenario")
    return cands, cands.mean(axis=0)


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


def run_single_scenario(args: argparse.Namespace, scenario: str, device: torch.device, seed: int) -> Dict[str, List[float]]:
    rng = set_seed(seed)
    if scenario == "ridge":
        dim = args.dim_ridge
        theta_true = np.ones(dim)
        noise_std = args.noise_std
        samples_per_gen = args.samples_per_gen
    elif scenario == "bayes":
        dim = args.dim_bayes
        theta_true = np.ones(dim) * args.mu_true_bayes
        noise_std = args.noise_std
        samples_per_gen = args.samples_per_gen
    elif scenario == "complex":
        dim = args.dim_complex
        theta_true = np.ones(dim) * args.mu_true_complex
        noise_std = args.complex_noise_std
        samples_per_gen = args.samples_per_gen_complex or args.samples_per_gen
    else:
        raise ValueError(f"unknown scenario {scenario}")

    # Models
    std_model = StandardFilter(dim=dim, hidden=args.hidden, dropout=args.dropout).to(device)
    sa_model_corr = SetAwareBiasRobustFilter(dim=dim, hidden=args.hidden, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout).to(device)
    sa_model_full = SetAwareBiasRobustFilter(dim=dim, hidden=args.hidden, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout).to(device)
    corr_mlp = CorrectionMLP(dim=dim, hidden=args.hidden, dropout=args.dropout).to(device)

    opt_std = optim.Adam(std_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_sa_corr = optim.Adam(sa_model_corr.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_sa_full = optim.Adam(sa_model_full.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_corr = optim.Adam(corr_mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    theta_good = sample_clean_reference(rng, theta_true, n=args.calibration_size, noise_std=noise_std)
    theta_good_t = torch.from_numpy(theta_good[None, ...]).float().to(device)
    theta_std = theta_good.copy()
    theta_only_corr_sa = theta_good.copy()
    theta_mlp_corr = theta_good.copy()
    theta_ours = theta_good.copy()

    series: Dict[str, List[float]] = {k: [] for k in METHODS}

    for _ in range(args.generations):
        if scenario in {"ridge", "bayes"}:
            cands, _ = generate_candidates(
                rng,
                scenario=scenario,
                theta_true=theta_true,
                samples_per_gen=samples_per_gen,
                noise_std=noise_std,
                ridge_alpha=args.ridge_alpha,
                bayes_mu_prior=args.mu_prior,
                bayes_sigma_prior=args.sigma_prior,
                bayes_n=args.bayes_n,
            )
        else:
            cands, _ = generate_complex_bias_candidates(
                rng,
                theta_true=theta_true,
                samples_per_gen=samples_per_gen,
                base_noise_std=args.complex_noise_std,
                noise_scale_std=args.complex_noise_scale,
                beta=args.complex_beta,
                outlier_ratio=args.complex_outlier_ratio,
            )
        labels = build_labels(cands, theta_true, top_ratio=args.top_ratio)
        x = torch.from_numpy(cands[None, ...]).float().to(device)
        y = torch.from_numpy(labels[None, ...]).float().to(device)

        # Variant A: only weighting (set-aware weights, delta=0)
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
        theta_std = theta_std + args.std_contraction * (theta_w_std.detach().cpu().numpy().squeeze(0) - theta_std)
        series["only_weight"].append(float(np.linalg.norm(theta_std - theta_true)))

        # Variant B: only correction (set-aware delta, weights=1)
        w_sa, delta_phi_sa = sa_model_corr(x)
        theta_w_sa = SetAwareBiasRobustFilter.weighted_estimate(x, w_sa)
        theta_new_sa = theta_w_sa * 0 + delta_phi_sa  # ignore weighting, use correction only
        loss_only_corr = (
            args.lambda_class * classification_loss(w_sa, y)
            + args.lambda_contract * contraction_loss(theta_new_sa, theta_good_t)
            + args.lambda_ess * ess_loss(w_sa, tau=args.tau)
            + args.lambda_reg * correction_reg(delta_phi_sa)
        )
        opt_sa_corr.zero_grad()
        loss_only_corr.backward()
        opt_sa_corr.step()
        theta_only_corr_sa = theta_only_corr_sa + args.ours_contraction * (
            theta_new_sa.detach().cpu().numpy().squeeze(0) - theta_only_corr_sa
        )
        series["only_correction"].append(float(np.linalg.norm(theta_only_corr_sa - theta_true)))

        # Variant C: MLP backbone + correction (no set interaction), weights=1
        delta_phi_mlp = corr_mlp(x)  # pooled inside
        theta_new_mlp = delta_phi_mlp
        loss_mlp = args.lambda_contract * contraction_loss(theta_new_mlp, theta_good_t) + args.lambda_reg * correction_reg(delta_phi_mlp)
        opt_corr.zero_grad()
        loss_mlp.backward()
        opt_corr.step()
        theta_mlp_corr = theta_mlp_corr + args.ours_contraction * (
            theta_new_mlp.detach().cpu().numpy().squeeze(0) - theta_mlp_corr
        )
        series["mlp_correction"].append(float(np.linalg.norm(theta_mlp_corr - theta_true)))

        # Ours full: set-aware weights + correction + gate
        w_full, delta_full, gate_full = sa_model_full.forward_with_gate(x)
        theta_mean = torch.from_numpy(cands.mean(axis=0)[None, None, :]).float().to(device)
        theta_w_full = SetAwareBiasRobustFilter.weighted_estimate(x, w_full)
        gate_expanded = gate_full.view(gate_full.shape[0], 1, 1)
        theta_new_full = (1 - gate_expanded) * theta_mean + gate_expanded * theta_w_full + delta_full
        loss_full = (
            args.lambda_class * classification_loss(w_full, y)
            + args.lambda_contract * contraction_loss(theta_new_full, theta_good_t)
            + args.lambda_ess * ess_loss(w_full, tau=args.tau)
            + args.lambda_reg * correction_reg(delta_full)
            + args.lambda_gate * torch.abs(gate_full).mean()
        )
        opt_sa_full.zero_grad()
        loss_full.backward()
        opt_sa_full.step()
        theta_ours = theta_ours + args.ours_contraction * (theta_new_full.detach().cpu().numpy().squeeze(0) - theta_ours)
        series["ours_full"].append(float(np.linalg.norm(theta_ours - theta_true)))

    return series


def save_csv(all_series: Dict[str, Dict[str, Dict[str, List[float]]]], tails: Dict[str, Dict[str, Dict[str, float]]], out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, series in all_series.items():
        gens = len(next(iter(series.values()))["mean"])
        with (out_dir / f"exp6_{name}_trajectories.csv").open("w") as f:
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
        with (out_dir / f"exp6_{name}_tail_summary.csv").open("w") as f:
            header = []
            for m in METHODS:
                header.extend([f"{m}_mean", f"{m}_std"])
            f.write(",".join(header) + "\n")
            row = []
            for m in METHODS:
                row.append(f"{tails[name][m]['mean']:.6f}")
                row.append(f"{tails[name][m]['std']:.6f}")
            f.write(",".join(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Exp 6: Architecture ablation (weighting vs correction).")
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--samples-per-gen", type=int, default=120)
    parser.add_argument("--samples-per-gen-complex", type=int, default=60, help="Optional override for complex bias scenario.")
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--calibration-size", type=int, default=120)
    parser.add_argument("--std-contraction", type=float, default=0.3)
    parser.add_argument("--ours-contraction", type=float, default=1.0)
    parser.add_argument("--top-ratio", type=float, default=0.3)
    parser.add_argument("--tau", type=float, default=50.0)
    parser.add_argument("--lambda-class", type=float, default=0.05)
    parser.add_argument("--lambda-contract", type=float, default=1.0)
    parser.add_argument("--lambda-ess", type=float, default=0.1)
    parser.add_argument("--lambda-reg", type=float, default=0.0)
    parser.add_argument("--lambda-gate", type=float, default=1e-3, help="L1 penalty on gate for ours_full.")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--dim_ridge", type=int, default=5)
    parser.add_argument("--dim_bayes", type=int, default=5)
    parser.add_argument("--mu-prior", type=float, default=0.0)
    parser.add_argument("--sigma-prior", type=float, default=0.2)
    parser.add_argument("--mu-true-bayes", type=float, default=5.0)
    parser.add_argument("--bayes-n", type=int, default=5)
    parser.add_argument("--dim-complex", type=int, default=10)
    parser.add_argument("--mu-true-complex", type=float, default=1.0)
    parser.add_argument("--complex-beta", type=float, default=2.0)
    parser.add_argument("--complex-noise-std", type=float, default=0.5)
    parser.add_argument("--complex-noise-scale", type=float, default=0.75, help="Lognormal sigma for dispersion heterogeneity.")
    parser.add_argument("--complex-outlier-ratio", type=float, default=0.35, help="Fraction of samples treated as biased outliers.")
    parser.add_argument("--scenarios", type=str, nargs="+", choices=["bayes", "ridge", "complex"])
    parser.add_argument("--tail-window", type=int, default=50)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1088, 2195, 4960, 1545, 3549, 1440, 3050, 5414],
        help="Seeds for CI/aggregation.",
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

    scenarios = args.scenarios if args.scenarios is not None else ["bayes", "ridge", "complex"]
    all_series: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    tails: Dict[str, Dict[str, Dict[str, float]]] = {}
    per_scenario_time: Dict[str, float] = {}
    start_all = time.time()

    for name in scenarios:
        t0 = time.time()
        seed_series: List[Dict[str, List[float]]] = []
        for s in seeds:
            seed_series.append(run_single_scenario(args, scenario=name, device=device, seed=s))
        all_series[name] = aggregate_seed_series(seed_series)
        tails[name] = aggregate_tail(seed_series, tail=args.tail_window)
        per_scenario_time[name] = time.time() - t0
        print(f"Scenario {name} done in {per_scenario_time[name]:.1f}s on {device}")

    plot_trajectories(all_series, fig_dir / "exp6_trajs.png", n_seeds=len(seeds))
    plot_tail_bar(tails, fig_dir / "exp6_tail_bar.png", n_seeds=len(seeds))
    save_csv(all_series, tails, out_dir)

    runtime = {
        "device": str(device),
        "seeds": list(seeds),
        "per_scenario_time_sec": per_scenario_time,
        "total_time_sec": time.time() - start_all,
        "note": "Variants: only weighting, only correction, MLP+correction, full set-aware. Scenarios: bayes prior, ridge, complex structural bias.",
    }
    with (out_dir / "runtime_exp6.json").open("w") as f:
        json.dump(runtime, f, indent=2)
    print(f"Saved Exp6 ablation to {out_dir} (tables) and {fig_dir} (figures)")


if __name__ == "__main__":
    main()
