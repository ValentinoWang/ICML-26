"""
Regenerate the digit-3 visualization grid with tighter spacing and arrow cues for recursion.

Reads MNIST, runs a single-seed Exp8 rollout for digit=3, and saves a 7xN grid:
rows = [GT, No, MLP, Batch-Stats, Ours, |GT-Batch|, |GT-Ours|],
cols = [GT, t=1,20,40,60,80,100,150,200].
Outputs go to:
  - Total_results/Figures/exp8_mnist_recursive/results/exp8_visual_grid.png
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_exp8_mnist_recursive import load_digits, run_single_seed

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
FIGURES_DIR = ROOT / "Total_results" / "Figures" / SCRIPT_DIR.name / "results"
RESULTS_DIR = ROOT / "exp8_mnist_recursive" / "results"


def main() -> None:
    # Match paper fonts (Times New Roman)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
        }
    )

    # Match main defaults, but generations=200 to expose t=200 frame
    args = SimpleNamespace(
        generations=200,
        batch_size=256,
        candidate_noise=0.1,
        candidates_per_gen=128,
        drift_deg=5.0,
        top_ratio=0.3,
        lambda_class=0.05,
        lambda_contract=1.0,
        lambda_ess=0.01,
        lambda_reg=1e-4,
        tau=50.0,
        ours_contraction=0.5,
        hidden=128,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        lr=1e-3,
        weight_decay=0.0,
        pca_dim=50,
        seeds=[1088, 2195, 4960],
        save_delta_viz=False,
        viz_seed=1088,
        viz_digit=3,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_digits(seed=args.seeds[0], pca_dim=args.pca_dim)

    # Single-seed rollout for digit=3
    run = run_single_seed(args, data, seed=args.seeds[0], device=device, digit=3)

    gens = [g for g in [0, 1, 20, 40, 60, 80, 100, 150, args.generations] if g <= args.generations]
    col_labels = [f"t={g}" for g in gens]  # 8 columns, no extra GT column
    n_cols = len(col_labels)
    n_rows = 7  # Ground Truth / No / MLP / Batch / Ours / Diff Batch / Diff Ours
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.5, 10.6))
    axes = axes.reshape(n_rows, n_cols)

    row_entries = [
        ("Ground Truth", None),
        ("No Filter", run["theta_no_hist"]),
        ("Pointwise (MLP)", run["theta_mlp_hist"]),
        ("Pointwise + Batch Stats", run["theta_batch_hist"]),
        ("Ours (Set-Aware)", run["theta_ours_hist"]),
        ("|GT - Batch|", run["theta_batch_hist"]),
        ("|GT - Ours|", run["theta_ours_hist"]),
    ]

    # Shared scale for difference maps (percentile to boost contrast)
    diff_values = []
    for _, hist in row_entries[-2:]:
        for g in gens:
            diff = np.abs(hist[g] - data["theta_star"][3])
            diff_values.append(diff.reshape(-1))
    if diff_values:
        diff_stack = np.concatenate(diff_values, axis=0)
        diff_max = float(np.percentile(diff_stack, 99.0))
    else:
        diff_max = 0.0
    diff_max = max(diff_max, 1e-6)

    for r, (name, hist) in enumerate(row_entries):
        for c in range(n_cols):
            ax = axes[r, c]
            g = gens[c]
            if hist is None:
                img = data["theta_star"][3]
                ax.imshow(img.reshape(28, 28), cmap="gray")
            else:
                img = hist[g]
                if name.startswith("|GT"):
                    diff = np.abs(img - data["theta_star"][3])
                    ax.imshow(diff.reshape(28, 28), cmap="inferno", vmin=0.0, vmax=diff_max)
                else:
                    ax.imshow(img.reshape(28, 28), cmap="gray")
            ax.axis("off")
            if r == 0:
                ax.set_title(col_labels[c], fontsize=13, pad=6)
            if c == 0:
                color = "#555555" if "Standard" in name else "#111111"
                ax.text(
                    -0.3,
                    0.5,
                    name,
                    fontsize=12,
                    fontweight="bold" if "Ours" in name else "normal",
                    color=color,
                    va="center",
                    ha="right",
                    transform=ax.transAxes,
                )

    # Highlight boxes: orange for Batch-Stats diff, green for Ours diff (last column).
    last_col_ax_batch = axes[5, -1]
    last_col_ax_ours = axes[6, -1]
    for target_ax, color in [(last_col_ax_batch, "orange"), (last_col_ax_ours, "green")]:
        rect = plt.Rectangle(
            (0, 0),
            1,
            1,
            fill=False,
            lw=2,
            ec=color,
            transform=target_ax.transAxes,
            clip_on=False,
        )
        target_ax.add_patch(rect)

    # Single, subtle arrow above columns to indicate progression.
    arrow_ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    arrow_ax.set_xlim(0, 1)
    arrow_ax.set_ylim(0, 1)
    arrow_ax.axis("off")
    y_arrow = 0.9
    arrow_ax.annotate(
        "",
        xy=(0.92, y_arrow),
        xytext=(0.08, y_arrow),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.8),
    )
    arrow_ax.text(0.5, y_arrow + 0.006, "Generation Process", ha="center", va="bottom", fontsize=18)

    fig.subplots_adjust(left=0.18, right=0.99, top=0.84, bottom=0.06, wspace=0.05, hspace=0.10)

    out_paths = [
        FIGURES_DIR / "exp8_visual_grid.png",
        RESULTS_DIR / "exp8_visual_grid.png",
    ]
    for out in out_paths:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)
    plt.close(fig)

    for out in out_paths:
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
