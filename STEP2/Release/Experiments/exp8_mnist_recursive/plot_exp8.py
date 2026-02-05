import pathlib
import shutil
import sys
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Common_Utils.plot_style import METHOD_COLOR, METHOD_LINESTYLE

# Font/LaTeX settings
# Keep plots robust: many environments ship a partial TeX install that breaks Matplotlib's usetex.
USE_TEX = False
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "mathtext.fontset": "cm",
        "text.usetex": USE_TEX,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


def plot_series(
    stats: Dict[str, Dict[str, List[float]]],
    out_dir: pathlib.Path,
    n_seeds: int,
    grid_path: pathlib.Path | None = None,
) -> None:
    gens = np.arange(1, len(next(iter(stats["mse"].values()))["mean"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    legend_handles = []
    legend_labels = []
    # Enforce a consistent method→color/linestyle mapping for the core baselines.
    style_overrides = {
        "no_filter": {"color": METHOD_COLOR["no_filter"], "linestyle": METHOD_LINESTYLE["no_filter"]},
        "mlp_filter": {"color": METHOD_COLOR["pointwise"], "linestyle": METHOD_LINESTYLE["pointwise"]},
        "ours": {"color": METHOD_COLOR["set_aware"], "linestyle": METHOD_LINESTYLE["set_aware"]},
    }
    for m, label in [
        ("no_filter", "No Filter"),
        ("mlp_filter", "MLP Filter (reweight only)"),
        ("batch_stats", "Pointwise + Batch Stats"),
        ("tent", "TENT (entropy min)"),
        ("dst", "DST (decoupled)"),
        ("l2ac", "L2AC (meta-weight)"),
        ("ours", "Ours (Set-Aware+PCA)"),
    ]:
        mean = np.array(stats["mse"][m]["mean"])
        std = np.array(stats["mse"][m]["std"])
        ci = 1.96 * std / np.sqrt(n_seeds)
        style = style_overrides.get(m, {})
        line, = axes[0].plot(gens, mean, label=label, linewidth=2, **style)
        axes[0].fill_between(gens, mean - ci, mean + ci, alpha=0.15)
        legend_handles.append(line)
        legend_labels.append(label)
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel(r"MSE to $\theta$")
    axes[0].set_title("Rotational Drift: MSE vs Generations")
    axes[0].grid(alpha=0.3)

    for m, label in [
        ("no_filter", "No Filter"),
        ("mlp_filter", "MLP Filter (reweight only)"),
        ("batch_stats", "Pointwise + Batch Stats"),
        ("tent", "TENT (entropy min)"),
        ("dst", "DST (decoupled)"),
        ("l2ac", "L2AC (meta-weight)"),
        ("ours", "Ours (Set-Aware+PCA)"),
    ]:
        mean = np.array(stats["norm"][m]["mean"])
        std = np.array(stats["norm"][m]["std"])
        ci = 1.96 * std / np.sqrt(n_seeds)
        style = style_overrides.get(m, {})
        axes[1].plot(gens, mean, label=label, linewidth=2, **style)
        axes[1].fill_between(gens, mean - ci, mean + ci, alpha=0.15)
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel(r"$\|\theta_t\|_2$")
    axes[1].set_title("Rotational Drift: Norm vs Generations")
    axes[1].grid(alpha=0.3)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=7,
        frameon=False,
        fontsize=7.2,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out_dir.mkdir(parents=True, exist_ok=True)
    curves_path = out_dir / "exp8_curves.png"
    plt.savefig(curves_path, dpi=200)
    plt.close()


def plot_series_main(
    stats: Dict[str, Dict[str, List[float]]],
    out_path: pathlib.Path,
    n_seeds: int,
) -> None:
    gens = np.arange(1, len(next(iter(stats["mse"].values()))["mean"]) + 1)
    methods = [
        ("mlp_filter", "Pointwise (MLP)", METHOD_COLOR["pointwise"], "-.", 2),
        ("batch_stats", "Pointwise + Batch Stats (\u2248 MLP)", "#ff7f0e", ":", 2),
        ("ours", "Set-Aware", METHOD_COLOR["set_aware"], "-", 3),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.2))

    for m, label, color, ls, zorder in methods:
        mean = np.array(stats["mse"][m]["mean"])
        std = np.array(stats["mse"][m]["std"])
        ci = 1.96 * std / np.sqrt(n_seeds)
        axes[0].plot(
            gens,
            mean,
            label=label,
            linewidth=1.8,
            color=color,
            linestyle=ls,
            zorder=zorder,
        )
        axes[0].fill_between(
            gens,
            mean - ci,
            mean + ci,
            alpha=0.12,
            color=color,
            zorder=zorder - 1,
        )
    axes[0].set_xlabel("Gen")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("MNIST Drift: MSE")
    axes[0].set_ylim(0.0, 0.04)
    axes[0].set_xlim(gens[0], gens[-1])
    tick_list = [1, 10, 20, 30, 40, 50]
    axes[0].set_xticks([t for t in tick_list if t <= gens[-1]])
    axes[0].grid(alpha=0.2)

    axins = inset_axes(axes[0], width="45%", height="45%", loc="upper right", borderpad=0.8)
    mean = np.array(stats["mse"]["ours"]["mean"])
    axins.plot(gens, mean, linewidth=1.2, color=METHOD_COLOR["set_aware"])
    axins.set_ylim(0.0, 0.001)
    axins.set_xlim(20, gens[-1])
    axins.set_xticks([20, gens[-1]])
    axins.set_yticks([0.0, 0.0005, 0.001])
    axins.set_yticklabels(["0", "5e-4", "1e-3"])
    axins.tick_params(axis="x", labelsize=7, pad=2)
    axins.tick_params(axis="y", labelsize=7)
    axins.text(0.02, 0.95, "Tail zoom", transform=axins.transAxes, fontsize=7, va="top", ha="left")
    final_val = float(mean[-1])
    axins.plot([gens[-1]], [final_val], marker="o", color="#1f77b4", markersize=3)
    label_val = final_val / 1e-4
    axins.text(
        gens[-1] - 7,
        final_val,
        rf"${label_val:.1f}\times10^{{-4}}$",
        fontsize=7,
        color=METHOD_COLOR["set_aware"],
        va="center",
        ha="left",
    )
    for spine in axins.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("#777777")

    for m, label, color, ls, zorder in methods:
        mean = np.array(stats["norm"][m]["mean"])
        std = np.array(stats["norm"][m]["std"])
        ci = 1.96 * std / np.sqrt(n_seeds)
        axes[1].plot(
            gens,
            mean,
            label=label,
            linewidth=1.8,
            color=color,
            linestyle=ls,
            zorder=zorder,
        )
        axes[1].fill_between(
            gens,
            mean - ci,
            mean + ci,
            alpha=0.12,
            color=color,
            zorder=zorder - 1,
        )
    axes[1].set_xlabel("Gen")
    axes[1].set_ylabel(r"$\|\theta_t\|_2$")
    axes[1].set_title("MNIST Drift: Norm")
    axes[1].set_xlim(gens[0], gens[-1])
    axes[1].set_xticks([t for t in tick_list if t <= gens[-1]])
    axes[1].grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=False,
        fontsize=8,
        columnspacing=1.0,
        handletextpad=0.6,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()



def save_grid_multi(
    rows: List[Dict[str, List[np.ndarray] | np.ndarray]],
    gens: List[int],
    out_path: pathlib.Path,
    digit_labels: Sequence[int] | None = None,
) -> None:
    col_labels = ["GT"] + [f"t={g}" for g in gens]
    n_cols = len(col_labels)
    n_rows = len(rows) * 5  # per digit: GT / No / MLP / Batch / Ours
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.2 * n_cols, 1.2 * n_rows))
    axes = axes.reshape(n_rows, n_cols)
    label_list = list(digit_labels) if digit_labels is not None else list(range(len(rows)))

    for d_idx, row in enumerate(rows):
        theta_star = row["theta_star"]
        no_hist = row["theta_no_hist"]
        mlp_hist = row["theta_mlp_hist"]
        batch_hist = row["theta_batch_hist"]
        ours_hist = row["theta_ours_hist"]
        digit_label = label_list[d_idx] if d_idx < len(label_list) else d_idx
        for m_idx, (name, hist) in enumerate(
            [("GT", None), ("No", no_hist), ("MLP", mlp_hist), ("Batch", batch_hist), ("Ours", ours_hist)]
        ):
            r = d_idx * 5 + m_idx
            for c in range(n_cols):
                ax = axes[r, c]
                if c == 0:
                    img = theta_star  # 第一列固定为 GT
                else:
                    g = gens[c - 1]
                    if m_idx == 0 or hist is None:
                        img = theta_star
                    else:
                        idx = min(g, len(hist) - 1)
                        img = hist[idx]
                ax.imshow(img.reshape(28, 28), cmap="gray")
                ax.axis("off")
                if d_idx == 0:
                    ax.set_title(col_labels[c], fontsize=9)
                if c == 0:
                    ax.text(
                        -0.28,
                        0.5,
                        f"{name}",
                        fontsize=9,
                        va="center",
                        ha="right",
                        transform=ax.transAxes,
                    )
                if c == 0 and m_idx == 0 and len(label_list) > 1:
                    ax.text(
                        -0.28,
                        1.08,
                        f"Digit {digit_label}",
                        fontsize=9,
                        va="bottom",
                        ha="left",
                        transform=ax.transAxes,
                    )
    fig.subplots_adjust(left=0.04, right=0.99, top=0.96, bottom=0.02, wspace=0.04, hspace=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
