"""
Render Exp7 combined figure using saved results (response curve + t-SNE).
Outputs a single PNG (default: results/exp7_combined.png) aligned with the paper.
"""

import argparse
import pathlib
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BASE_TABLES_DIR = ROOT / "Total_results" / "Tables" / SCRIPT_DIR.name
BASE_FIGURES_DIR = ROOT / "Total_results" / "Figures" / SCRIPT_DIR.name
TABLES_DIR = BASE_TABLES_DIR / "results"
FIGURES_DIR = BASE_FIGURES_DIR / "results"

# ICML-styled typography and spines
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 13,
        "figure.constrained_layout.use": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Color palette
COLORS = {
    "gt": "black",
    "baseline": "#c0392b",
    "pointwise": "#8e44ad",
    "batch_stats": "#ff7f0e",
    "ours": "#2ca02c",
    "stable_txt": "#1f4e79",
}

def load_response_curve(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    x = df["metric"].to_numpy()
    gt = df["gt_norm"].to_numpy()
    baseline = df["mlp_pred_norm"].to_numpy()
    pointwise = df["pointwise_pred_norm"].to_numpy()
    batch_stats = df["batch_pred_norm"].to_numpy()
    ours = df["sa_pred_norm"].to_numpy()
    return x, gt, baseline, pointwise, batch_stats, ours


def load_latents(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrs = dict(np.load(path))
    variances = arrs["variances"]
    feats_pw = arrs["feats_pw"]
    feats_sa = arrs["feats_sa"]
    return variances, feats_pw, feats_sa


def embed_tsne(features: np.ndarray, seed: int) -> np.ndarray:
    if features.shape[1] <= 2:
        return features
    tsne = TSNE(
        n_components=2,
        perplexity=35,
        init="random",
        random_state=seed,
        learning_rate="auto",
        n_iter=1000,
    )
    return tsne.fit_transform(features)


def plot_response_panel(
    ax: plt.Axes,
    x: np.ndarray,
    gt: np.ndarray,
    baseline: np.ndarray,
    pointwise: np.ndarray,
    batch_stats: np.ndarray,
    ours: np.ndarray,
) -> None:
    ax.plot(
        x,
        gt,
        linestyle="--",
        color=COLORS["gt"],
        linewidth=2.0,
        dashes=(5, 2),
        label="Ground Truth",
        zorder=1,
    )
    ax.scatter(x, baseline, color=COLORS["baseline"], s=42, alpha=0.9, label="Correction MLP", zorder=2.5)
    ax.scatter(
        x,
        pointwise,
        color=COLORS["pointwise"],
        s=42,
        alpha=0.9,
        label="Pointwise (MLP)",
        zorder=2.0,
    )
    ax.scatter(
        x,
        batch_stats,
        color=COLORS["batch_stats"],
        s=42,
        alpha=0.9,
        label="Pointwise + Batch Stats",
        zorder=2.2,
    )
    ax.scatter(
        x,
        ours,
        color=COLORS["ours"],
        s=62,
        alpha=0.95,
        edgecolors="black",
        linewidths=0.6,
        label="Set-Aware (Ours)",
        zorder=3,
    )

    idx_best = int(np.argmin(np.abs(ours - gt)))
    ax.annotate(
        "Perfect Alignment",
        xy=(x[idx_best], ours[idx_best]),
        xytext=(x[idx_best], ours[idx_best] + 5.5),
        fontsize=13,
        color=COLORS["ours"],
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=COLORS["ours"], lw=1.6),
    )

    ax.set_xlabel("Pair Gap (Anomaly Distance)", fontsize=16, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"Correction Norm $\|\Delta \phi\|_2$", fontsize=16, fontweight="bold", labelpad=6)
    ax.set_title("(a) Quantitative: Geometric Recovery", loc="left", pad=12, fontsize=17)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=True, framealpha=0.9, loc="upper right")


def _scatter_tsne(
    ax: plt.Axes,
    pts: np.ndarray,
    values: np.ndarray,
    title: str,
    annotate_text: str,
    pick_fn,
    text_color: str,
    offset: Tuple[float, float],
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
) -> None:
    sc = ax.scatter(pts[:, 0], pts[:, 1], s=11, c=values, cmap="coolwarm", alpha=0.82, linewidths=0)
    idx = pick_fn(values)
    ax.annotate(
        annotate_text,
        xy=(pts[idx, 0], pts[idx, 1]),
        xytext=(pts[idx, 0] + offset[0], pts[idx, 1] + offset[1]),
        fontsize=12,
        color=text_color,
        ha="center",
        va="center",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=text_color, lw=1.4),
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.8),
    )
    ax.set_title(title, fontsize=16, pad=8, loc="center")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_aspect("equal")
    (xmin, xmax), (ymin, ymax) = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    return sc


def plot_tsne_panels(fig: plt.Figure, grid: GridSpec, base_pts: np.ndarray, ours_pts: np.ndarray, variances: np.ndarray) -> None:
    # Shared limits to align the two panels
    all_pts = np.vstack([base_pts, ours_pts])
    pad_x = 0.08 * (all_pts[:, 0].max() - all_pts[:, 0].min() + 1e-8)
    pad_y = 0.08 * (all_pts[:, 1].max() - all_pts[:, 1].min() + 1e-8)
    bounds = (
        (all_pts[:, 0].min() - pad_x, all_pts[:, 0].max() + pad_x),
        (all_pts[:, 1].min() - pad_y, all_pts[:, 1].max() + pad_y),
    )

    norm = Normalize(vmin=float(variances.min()), vmax=float(variances.max()))
    cmap = "coolwarm"

    ax_base = fig.add_subplot(grid[0, 1])
    sc_base = _scatter_tsne(
        ax_base,
        base_pts,
        variances,
        title="(b) Baseline: Manifold Collapse",
        annotate_text="High Variance\n(Collapse)",
        pick_fn=np.argmax,
        text_color=COLORS["baseline"],
        offset=(4.0, 4.0),
        bounds=bounds,
    )

    ax_ours = fig.add_subplot(grid[1, 1])
    sc_ours = _scatter_tsne(
        ax_ours,
        ours_pts,
        variances,
        title="(c) Ours: Structure Preserved",
        annotate_text="Low Variance\n(Stable)",
        pick_fn=np.argmin,
        text_color=COLORS["stable_txt"],
        offset=(-4.0, -4.0),
        bounds=bounds,
    )

    cax = fig.add_axes([0.91, 0.20, 0.018, 0.55])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Batch Variance / Gap", labelpad=6, fontsize=14)
    cb.ax.tick_params(labelsize=11, width=0.8)
    for spine in cb.ax.spines.values():
        spine.set_visible(False)
    sc_base.set_cmap(cmap)
    sc_ours.set_cmap(cmap)


def plot_exp7_combined(
    response_csv: pathlib.Path,
    tsne_npz: pathlib.Path,
    out_path: pathlib.Path,
    seed: int = 0,
) -> None:
    x, gt, baseline, pointwise, batch_stats, ours = load_response_curve(response_csv)
    variances, feats_pw, feats_sa = load_latents(tsne_npz)

    base_pts = embed_tsne(feats_pw, seed=seed)
    ours_pts = embed_tsne(feats_sa, seed=seed + 1)

    fig = plt.figure(figsize=(13.5, 6.8))
    gs = GridSpec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.32, figure=fig)

    ax_resp = fig.add_subplot(gs[:, 0])
    plot_response_panel(ax_resp, x, gt, baseline, pointwise, batch_stats, ours)
    plot_tsne_panels(fig, gs, base_pts, ours_pts, variances)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved Exp7 combined figure to {out_path}")
    plt.close(fig)


def save_exp7_visuals(
    records: dict[str, np.ndarray],
    attn_low: np.ndarray,
    attn_high: np.ndarray,
    variances: np.ndarray,
    feats_mlp: np.ndarray,
    feats_pw: np.ndarray,
    feats_sa: np.ndarray,
    mode: str,
    seed: int,
    table_dir: pathlib.Path,
    figure_dir: pathlib.Path,
) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    response_csv = table_dir / "exp7_response_curve.csv"
    with response_csv.open("w", encoding="utf-8") as f:
        f.write("metric,gt_norm,mlp_pred_norm,pointwise_pred_norm,batch_pred_norm,sa_pred_norm\n")
        for m, gt, mlp, pw, bs, sa in zip(
            records["metric"],
            records["gt"],
            records["mlp"],
            records["pw"],
            records["bs"],
            records["sa"],
        ):
            f.write(
                f"{float(m):.6f},{float(gt):.6f},{float(mlp):.6f},{float(pw):.6f},{float(bs):.6f},{float(sa):.6f}\n"
            )

    tsne_npz = table_dir / "exp7_tsne_latent.npz"
    np.savez(
        tsne_npz,
        variances=variances,
        feats_mlp=feats_mlp,
        feats_pw=feats_pw,
        feats_sa=feats_sa,
    )

    attn_path = table_dir / "exp7_attention_maps.npz"
    np.savez(attn_path, attn_low=np.asarray(attn_low), attn_high=np.asarray(attn_high), mode=mode)

    plot_exp7_combined(response_csv, tsne_npz, figure_dir / "exp7_combined.png", seed=seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Exp7 response + t-SNE combined figure.")
    parser.add_argument("--response-csv", type=pathlib.Path, default=TABLES_DIR / "exp7_response_curve.csv")
    parser.add_argument("--tsne-npz", type=pathlib.Path, default=TABLES_DIR / "exp7_tsne_latent.npz")
    parser.add_argument("--out", type=pathlib.Path, default=FIGURES_DIR / "exp7_combined.png")
    parser.add_argument("--seed", type=int, default=0, help="Seed for t-SNE embeddings.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_exp7_combined(args.response_csv, args.tsne_npz, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
