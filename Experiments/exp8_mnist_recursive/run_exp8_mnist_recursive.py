import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List, Tuple, Sequence

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from scipy.ndimage import rotate
from sklearn.decomposition import PCA
from torch import optim
import matplotlib.pyplot as plt

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
from plot_exp8 import plot_series, save_grid_multi  # noqa: E402


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def build_labels(candidates: np.ndarray, theta_true: np.ndarray, top_ratio: float) -> np.ndarray:
    dists = np.linalg.norm(candidates - theta_true, axis=1)
    k = max(1, int(len(candidates) * top_ratio))
    thresh = np.partition(dists, k - 1)[k - 1]
    return (dists <= thresh).astype(np.float32)


def load_digits(seed: int, pca_dim: int) -> Dict[str, np.ndarray | PCA]:
    torch.manual_seed(seed)
    transform = T.Compose([T.ToTensor()])
    train_set = torchvision.datasets.MNIST(root=ROOT / "data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root=ROOT / "data", train=False, download=True, transform=transform)

    def collect(ds):
        per_class: Dict[int, List[np.ndarray]] = {i: [] for i in range(10)}
        for img, label in ds:
            per_class[label].append(np.asarray(img).reshape(-1))
        return {k: np.stack(v, axis=0) for k, v in per_class.items()}

    train_by_class = collect(train_set)
    test_by_class = collect(test_set)
    # Fit PCA on all training images for a shared basis
    all_train = np.concatenate(list(train_by_class.values()), axis=0)
    pca = PCA(n_components=pca_dim, random_state=seed)
    pca.fit(all_train)
    theta_star = {k: v.mean(axis=0).astype(np.float32) for k, v in train_by_class.items()}
    return {"train_by_class": train_by_class, "test_by_class": test_by_class, "theta_star": theta_star, "pca": pca}


def rotate_batch(imgs: np.ndarray, angle: float) -> np.ndarray:
    rotated = [rotate(img.reshape(28, 28), angle=angle, reshape=False, order=1, mode="constant", cval=0.0).reshape(-1) for img in imgs]
    return np.stack(rotated, axis=0)


def make_candidates(rng: np.random.Generator, theta_hat: np.ndarray, n: int, noise: float) -> np.ndarray:
    return theta_hat + noise * rng.normal(size=(n, theta_hat.shape[0]))


def weighted_estimate(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w_exp = w.unsqueeze(-1)
    num = (w_exp * x).sum(dim=1)
    den = w_exp.sum(dim=1, keepdim=True).clamp_min(eps)
    return num / den


def _to_numpy_vec(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().numpy()
    return arr.squeeze()


def save_delta_viz(dst_bias_img: np.ndarray, delta_img: np.ndarray, out_path: pathlib.Path, title_suffix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.2))
    for ax, img, title in zip(
        axes,
        [dst_bias_img, delta_img],
        ["DST Bias Head", "Set-Aware $\\Delta\\phi$"],
    ):
        ax.imshow(img.reshape(28, 28), cmap="gray")
        ax.axis("off")
        ax.set_title(title, fontsize=9)
    fig.suptitle(title_suffix, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def run_single_seed(
    args: argparse.Namespace,
    data: Dict[str, np.ndarray | PCA],
    seed: int,
    device: torch.device,
    digit: int,
) -> Dict[str, List]:
    rng = set_seed(seed)
    theta_star = data["theta_star"][digit]
    pca: PCA = data["pca"]  # type: ignore
    theta_no = pca.transform(theta_star[None, ...]).astype(np.float32).squeeze(0)
    theta_ours = theta_no.copy()
    theta_dst = theta_no.copy()
    theta_l2ac = theta_no.copy()

    model = SetAwareBiasRobustFilter(dim=args.pca_dim, hidden=args.hidden, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    mse = {
        "no_filter": [],
        "mlp_filter": [],
        "batch_stats": [],
        "tent": [],
        "dst": [],
        "l2ac": [],
        "ours": [],
    }
    norm = {
        "no_filter": [],
        "mlp_filter": [],
        "batch_stats": [],
        "tent": [],
        "dst": [],
        "l2ac": [],
        "ours": [],
    }
    theta_no_hist: List[np.ndarray] = []
    theta_mlp_hist: List[np.ndarray] = []
    theta_batch_hist: List[np.ndarray] = []
    theta_tent_hist: List[np.ndarray] = []
    theta_ours_hist: List[np.ndarray] = []
    # Generation 0: start from the unbiased reference (for visualization)
    theta_init_img = pca.inverse_transform(theta_no)
    theta_no_hist.append(theta_init_img)
    theta_mlp_hist.append(theta_init_img)
    theta_batch_hist.append(theta_init_img)
    theta_tent_hist.append(theta_init_img)
    theta_ours_hist.append(theta_init_img)

    # Point-wise MLP for weighting only
    mlp_weight = torch.nn.Sequential(
        torch.nn.Linear(args.pca_dim, args.hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout),
        torch.nn.Linear(args.hidden, args.hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout),
        torch.nn.Linear(args.hidden, 1),
    ).to(device)
    opt_mlp = optim.Adam(mlp_weight.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Point-wise MLP with batch statistics (mean/variance) concatenated to each sample.
    mlp_batch = torch.nn.Sequential(
        torch.nn.Linear(args.pca_dim * 3, args.hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout),
        torch.nn.Linear(args.hidden, args.hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout),
        torch.nn.Linear(args.hidden, 1),
    ).to(device)
    opt_batch = optim.Adam(mlp_batch.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # TENT: entropy minimization on softmax over candidate weights (test-time adaptation)
    mlp_weight_tent = torch.nn.Sequential(
        torch.nn.Linear(args.pca_dim, args.hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout),
        torch.nn.Linear(args.hidden, args.hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout),
        torch.nn.Linear(args.hidden, 1),
    ).to(device)
    opt_tent = optim.Adam(mlp_weight_tent.parameters(), lr=args.lr * 5.0, weight_decay=args.weight_decay)

    # DST baseline: bias head + main head (pointwise)
    dst_bias = torch.nn.Sequential(
        torch.nn.Linear(args.pca_dim, args.hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout),
        torch.nn.Linear(args.hidden, args.hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout),
        torch.nn.Linear(args.hidden, 1),
    ).to(device)
    dst_main = torch.nn.Sequential(
        torch.nn.Linear(args.pca_dim, args.hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout),
        torch.nn.Linear(args.hidden, args.hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout),
        torch.nn.Linear(args.hidden, 1),
    ).to(device)
    opt_dst = optim.Adam(
        list(dst_bias.parameters()) + list(dst_main.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # L2AC-style baseline: meta-aligned weighting using clean reference direction.
    l2ac_head = torch.nn.Sequential(
        torch.nn.Linear(args.pca_dim, args.hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout),
        torch.nn.Linear(args.hidden, args.hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout),
        torch.nn.Linear(args.hidden, 1),
    ).to(device)
    opt_l2ac = optim.Adam(l2ac_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    theta_star_t = torch.from_numpy(theta_no[None, None, :]).float().to(device)

    dst_bias_img = None
    delta_img = None
    for _ in range(args.generations):
        # Reconstruct current mean to pixel, apply drift (rotate)
        theta_no_img = pca.inverse_transform(theta_no)
        imgs = np.tile(theta_no_img, (args.batch_size, 1))
        imgs_rot = rotate_batch(imgs, angle=args.drift_deg)
        # Biased estimate: mean of rotated imgs projected to PCA
        theta_hat = imgs_rot.mean(axis=0)
        theta_hat_proj = pca.transform(theta_hat[None, ...]).astype(np.float32).squeeze(0)
        theta_no = theta_hat_proj

        # Candidates around biased estimate
        candidates_img = make_candidates(rng, theta_hat, args.candidates_per_gen, args.candidate_noise)
        labels = build_labels(candidates_img, theta_star, top_ratio=args.top_ratio)
        c_proj = pca.transform(candidates_img).astype(np.float32)
        x = torch.from_numpy(c_proj[None, ...]).float().to(device)
        y = torch.from_numpy(labels[None, ...]).float().to(device)
        theta_hat_t = torch.from_numpy(theta_hat_proj[None, None, :]).float().to(device)

        # MLP filter (reweighting only)
        w_mlp = torch.sigmoid(mlp_weight(torch.from_numpy(c_proj).float().to(device))).squeeze(-1)  # [N]
        theta_w_mlp = weighted_estimate(x.squeeze(0), w_mlp.unsqueeze(0))
        loss_mlp = (
            args.lambda_class * classification_loss(w_mlp.unsqueeze(0), y)
            + args.lambda_contract * contraction_loss(theta_w_mlp.unsqueeze(1), theta_star_t)
            + args.lambda_ess * ess_loss(w_mlp.unsqueeze(0), tau=args.tau)
        )
        opt_mlp.zero_grad()
        loss_mlp.backward()
        opt_mlp.step()
        theta_mlp = theta_w_mlp.detach().cpu().numpy().squeeze(0)
        theta_mlp_hist.append(pca.inverse_transform(theta_mlp))

        # Pointwise + Batch Stats baseline (mean/variance concatenated to each sample)
        x_flat = x.squeeze(0)
        batch_mean = x_flat.mean(dim=0, keepdim=True)
        batch_var = x_flat.var(dim=0, unbiased=False, keepdim=True)
        feat_batch = torch.cat([x_flat, batch_mean.expand_as(x_flat), batch_var.expand_as(x_flat)], dim=1)
        w_batch = torch.sigmoid(mlp_batch(feat_batch)).squeeze(-1)
        theta_w_batch = weighted_estimate(x_flat, w_batch.unsqueeze(0))
        loss_batch = (
            args.lambda_class * classification_loss(w_batch.unsqueeze(0), y)
            + args.lambda_contract * contraction_loss(theta_w_batch.unsqueeze(1), theta_star_t)
            + args.lambda_ess * ess_loss(w_batch.unsqueeze(0), tau=args.tau)
        )
        opt_batch.zero_grad()
        loss_batch.backward()
        opt_batch.step()
        theta_batch = theta_w_batch.detach().cpu().numpy().squeeze(0)

        # TENT adaptation: entropy minimization over candidate weights (softmax)
        for _ in range(5):  # multiple TENT steps to induce movement
            logits_tent = mlp_weight_tent(torch.from_numpy(c_proj).float().to(device)).squeeze(-1)  # [N]
            prob_tent = torch.softmax(logits_tent, dim=0)
            entropy = -(prob_tent * torch.log(prob_tent + 1e-8)).sum()
            opt_tent.zero_grad()
            entropy.backward()
            opt_tent.step()
        with torch.no_grad():
            logits_tent = mlp_weight_tent(torch.from_numpy(c_proj).float().to(device)).squeeze(-1)
            prob_tent = torch.softmax(logits_tent, dim=0)
            theta_w_tent = weighted_estimate(x.squeeze(0), prob_tent.unsqueeze(0))
            theta_tent = theta_w_tent.detach().cpu().numpy().squeeze(0)
        theta_tent_hist.append(pca.inverse_transform(theta_tent))

        # DST: bias head fits biased estimate; main head downweights bias-heavy samples.
        w_bias = torch.sigmoid(dst_bias(torch.from_numpy(c_proj).float().to(device))).squeeze(-1)
        theta_bias = weighted_estimate(x.squeeze(0), w_bias.unsqueeze(0))
        loss_bias = contraction_loss(theta_bias.unsqueeze(1), theta_hat_t) + args.lambda_ess * ess_loss(
            w_bias.unsqueeze(0), tau=args.tau
        )

        w_main = torch.sigmoid(dst_main(torch.from_numpy(c_proj).float().to(device))).squeeze(-1)
        bias_mask = (1.0 - w_bias.detach()).clamp_min(0.05)
        w_dst = (w_main * bias_mask).clamp_min(1e-4)
        theta_w_dst = weighted_estimate(x.squeeze(0), w_dst.unsqueeze(0))
        loss_dst = (
            args.lambda_class * classification_loss(w_main.unsqueeze(0), y)
            + args.lambda_contract * contraction_loss(theta_w_dst.unsqueeze(1), theta_star_t)
            + args.lambda_ess * ess_loss(w_main.unsqueeze(0), tau=args.tau)
        )
        opt_dst.zero_grad()
        (loss_bias + loss_dst).backward()
        opt_dst.step()
        theta_dst = theta_w_dst.detach().cpu().numpy().squeeze(0)

        # L2AC-style meta weighting: align samples with clean reference direction.
        theta_l2ac_t = torch.from_numpy(theta_l2ac[None, None, :]).float().to(device)
        x_flat = x.squeeze(0)
        theta_l2ac_vec = theta_l2ac_t.squeeze()
        delta_dir = (theta_star_t - theta_l2ac_t).squeeze()
        align_score = torch.relu((x_flat - theta_l2ac_vec) @ delta_dir).unsqueeze(0)
        w_l2ac_raw = torch.sigmoid(l2ac_head(torch.from_numpy(c_proj).float().to(device))).squeeze(-1)
        w_l2ac = (w_l2ac_raw * (align_score.squeeze(0) + 1e-6)).clamp_min(1e-4)
        theta_w_l2ac = weighted_estimate(x.squeeze(0), w_l2ac.unsqueeze(0))
        loss_l2ac = (
            args.lambda_class * classification_loss(w_l2ac_raw.unsqueeze(0), y)
            + args.lambda_contract * contraction_loss(theta_w_l2ac.unsqueeze(1), theta_star_t)
            + args.lambda_ess * ess_loss(w_l2ac_raw.unsqueeze(0), tau=args.tau)
        )
        opt_l2ac.zero_grad()
        loss_l2ac.backward()
        opt_l2ac.step()
        theta_l2ac = theta_w_l2ac.detach().cpu().numpy().squeeze(0)

        w, delta = model(x)
        theta_new = delta  # estimation uses correction only
        loss = (
            args.lambda_class * classification_loss(w, y)
            + args.lambda_contract * contraction_loss(theta_new, theta_star_t)
            + args.lambda_ess * ess_loss(w, tau=args.tau)
            + args.lambda_reg * correction_reg(delta)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        theta_ours = theta_ours + args.ours_contraction * (theta_new.detach().cpu().numpy().squeeze(0) - theta_ours)

        if args.save_delta_viz:
            dst_bias_img = pca.inverse_transform(_to_numpy_vec(theta_bias))
            delta_img = pca.inverse_transform(_to_numpy_vec(delta))

        # Metrics in pixel space
        for key, t_proj in [
            ("no_filter", theta_no),
            ("mlp_filter", theta_mlp),
            ("batch_stats", theta_batch),
            ("tent", theta_tent),
            ("dst", theta_dst),
            ("l2ac", theta_l2ac),
            ("ours", theta_ours),
        ]:
            t_img = pca.inverse_transform(t_proj)
            mse[key].append(float(np.mean((t_img - theta_star) ** 2)))
            norm[key].append(float(np.linalg.norm(t_img)))

        theta_no_hist.append(pca.inverse_transform(theta_no))
        theta_mlp_hist.append(pca.inverse_transform(theta_mlp))
        theta_batch_hist.append(pca.inverse_transform(theta_batch))
        theta_tent_hist.append(pca.inverse_transform(theta_tent))
        theta_ours_hist.append(pca.inverse_transform(theta_ours))

    return {
        "mse": mse,
        "norm": norm,
        "theta_no_hist": theta_no_hist,
        "theta_mlp_hist": theta_mlp_hist,
        "theta_batch_hist": theta_batch_hist,
        "theta_tent_hist": theta_tent_hist,
        "theta_ours_hist": theta_ours_hist,
        "dst_bias_img": dst_bias_img,
        "delta_img": delta_img,
    }


def aggregate(seed_runs: List[Dict[str, Dict[str, List[float]]]]) -> Dict[str, Dict[str, List[float]]]:
    keys = seed_runs[0].keys()
    out: Dict[str, Dict[str, List[float]]] = {}
    for k in keys:
        if k.endswith("_hist"):
            continue
        methods = seed_runs[0][k].keys()
        out[k] = {}
        for m in methods:
            arr = np.array([run[k][m] for run in seed_runs])
            out[k][m] = {"mean": arr.mean(axis=0).tolist(), "std": arr.std(axis=0).tolist()}
    return out


def save_csv(stats: Dict[str, Dict[str, List[float]]], out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gens = len(next(iter(stats["mse"].values()))["mean"])
    with (out_dir / "exp8_trajectories.csv").open("w") as f:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp8: MNIST digit-3 rotational drift with set-aware correction.")
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--candidate-noise", type=float, default=0.1)
    parser.add_argument("--candidates-per-gen", type=int, default=128)
    parser.add_argument("--drift-deg", type=float, default=5.0, help="Per-generation rotation bias (degrees, clockwise).")
    parser.add_argument("--top-ratio", type=float, default=0.3)
    parser.add_argument("--lambda-class", type=float, default=0.05)
    parser.add_argument("--lambda-contract", type=float, default=1.0)
    parser.add_argument("--lambda-ess", type=float, default=0.01)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=50.0)
    parser.add_argument("--ours-contraction", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1088, 2195, 4960])
    parser.add_argument("--out-dir", type=pathlib.Path, default=TABLES_DIR, help="Directory to store CSV/JSON outputs.")
    parser.add_argument("--fig-dir", type=pathlib.Path, default=FIGURES_DIR, help="Directory to store figures.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-delta-viz", action="store_true", help="Save DST bias head vs Set-Aware delta phi visualization.")
    parser.add_argument("--viz-seed", type=int, default=1088)
    parser.add_argument("--viz-digit", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    t0 = time.time()

    data = load_digits(seed=args.seeds[0], pca_dim=args.pca_dim)
    seed_runs = []
    theta_no_hist = []
    theta_mlp_hist = []
    theta_batch_hist = []
    theta_ours_hist = []
    viz_payload = None
    for s in args.seeds:
        run = run_single_seed(args, data, seed=s, device=device, digit=3)
        seed_runs.append({"mse": run["mse"], "norm": run["norm"]})
        if s == args.seeds[0]:
            theta_no_hist = run["theta_no_hist"]
            theta_mlp_hist = run["theta_mlp_hist"]
            theta_batch_hist = run["theta_batch_hist"]
            theta_ours_hist = run["theta_ours_hist"]
        if args.save_delta_viz and s == args.viz_seed:
            viz_payload = run

    stats = aggregate(seed_runs)
    plot_series(stats, args.fig_dir, n_seeds=len(args.seeds))
    save_csv(stats, args.out_dir)

    # Multi-digit grid using first seed
    grid_rows = []
    digit_labels = list(range(10))
    for digit in digit_labels:
        run_d = run_single_seed(args, data, seed=args.seeds[0], device=device, digit=digit)
        grid_rows.append(
            {
                "theta_star": data["theta_star"][digit],
                "theta_no_hist": run_d["theta_no_hist"],
                "theta_mlp_hist": run_d["theta_mlp_hist"],
                "theta_batch_hist": run_d["theta_batch_hist"],
                "theta_ours_hist": run_d["theta_ours_hist"],
            }
        )
    gens_vis = [0, 1, 20, 40, 60, 80, 100, 150, args.generations]
    save_grid_multi(grid_rows, gens=gens_vis, out_path=args.fig_dir / "exp8_visual_grid_digits.png", digit_labels=digit_labels)

    if args.save_delta_viz and viz_payload is not None:
        dst_bias_img = viz_payload.get("dst_bias_img")
        delta_img = viz_payload.get("delta_img")
        if dst_bias_img is not None and delta_img is not None:
            out_path = args.fig_dir / "exp8_dst_vs_delta_phi.png"
            save_delta_viz(dst_bias_img, delta_img, out_path, f"Seed {args.viz_seed}, Digit {args.viz_digit}, G{args.generations}")

    runtime = {
        "device": str(device),
        "seeds": args.seeds,
        "total_time_sec": time.time() - t0,
        "note": "Rotational drift + set-aware correction (PCA=50). Ours uses correction only; weights aux.",
    }
    with (args.out_dir / "runtime_exp8.json").open("w") as f:
        json.dump(runtime, f, indent=2)
    print(f"Saved Exp8 results to {args.out_dir} (tables) and {args.fig_dir} (figures)")


if __name__ == "__main__":
    main()
