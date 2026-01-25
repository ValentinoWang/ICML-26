#!/usr/bin/env python3
"""
Refined plots for Exp12 (Vendi + t-SNE KDE) with publication styling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Publication style (match Figure 1)
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Colors aligned with Figure 1
COLOR_RED = "#C44E52"  # Pointwise
COLOR_BLUE = "#4C72B0"  # Set-Aware
COLOR_ORANGE = "#E69F00"  # No Filter
COLOR_GREEN = "#55A868"  # Dispersion
COLOR_GRAY = "#BBBBBB"  # Reference


def plot_vendi_curve(csv_path: Path, output_path: Path) -> None:
    df = pd.read_csv(csv_path)
    # Expect columns: method, generation, vendi, vendi_std
    df = df.rename(columns={"generation": "Generation"})
    fig, ax = plt.subplots(figsize=(5, 4))

    styles = {
        "no_filter": {"color": COLOR_ORANGE, "marker": "s", "label": "No Filter"},
        "pointwise": {"color": COLOR_RED, "marker": "^", "label": "Pointwise"},
        "dispersion": {"color": COLOR_GREEN, "marker": "D", "label": "Dispersion"},
        "set_aware": {"color": COLOR_BLUE, "marker": "o", "label": "Set-Aware"},
    }
    order = ["no_filter", "pointwise", "dispersion", "set_aware"]
    for method in order:
        subset = df[df["method"] == method].sort_values("Generation")
        if subset.empty:
            continue
        style = styles[method]
        ax.fill_between(
            subset["Generation"],
            subset["vendi"] - subset["vendi_std"],
            subset["vendi"] + subset["vendi_std"],
            color=style["color"],
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            subset["Generation"],
            subset["vendi"],
            color=style["color"],
            lw=2.4,
            marker=style["marker"],
            label=style["label"],
        )

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Vendi Score (Semantic Volume)", fontsize=12)
    ax.set_title("Topological Volume Stability", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(sorted(df["Generation"].unique()))
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(
        frameon=False,
        fontsize=9,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        columnspacing=1.0,
        handletextpad=0.6,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(top=0.82)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)


def plot_tsne_contours(csv_path: Path, output_path: Path) -> None:
    df = pd.read_csv(csv_path)
    # Expect columns: label, x, y
    df = df.rename(columns={"label": "method"})
    df["method"] = df["method"].replace(
        {
            "reference": "Reference",
            "no_filter": "No Filter",
            "pointwise": "Pointwise",
            "dispersion": "Dispersion",
            "set_aware": "Set-Aware",
        }
    )

    fig, ax = plt.subplots(figsize=(5, 4))

    # Reference background
    subset_ref = df[df["method"] == "Reference"]
    sns.kdeplot(
        data=subset_ref,
        x="x",
        y="y",
        color=COLOR_GRAY,
        fill=True,
        alpha=0.3,
        levels=5,
        thresh=0.05,
        warn_singular=False,
        ax=ax,
    )
    ax.scatter(subset_ref["x"], subset_ref["y"], c=COLOR_GRAY, s=5, alpha=0.2, label="_nolegend_")

    method_styles = {
        "No Filter": {"color": COLOR_ORANGE, "levels": 3, "lw": 1.4, "alpha": 0.25},
        "Pointwise": {"color": COLOR_RED, "levels": 4, "lw": 1.6, "alpha": 1.0},
        "Dispersion": {"color": COLOR_GREEN, "levels": 3, "lw": 1.4, "alpha": 0.25},
        "Set-Aware": {"color": COLOR_BLUE, "levels": 4, "lw": 2.0, "alpha": 1.0},
    }
    for name in ["No Filter", "Pointwise", "Dispersion", "Set-Aware"]:
        subset = df[df["method"] == name]
        if subset.empty:
            continue
        style = method_styles[name]
        sns.kdeplot(
            data=subset,
            x="x",
            y="y",
            color=style["color"],
            levels=style["levels"],
            linewidths=style["lw"],
            alpha=style["alpha"],
            warn_singular=False,
            ax=ax,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE Dim 1", fontsize=12)
    ax.set_ylabel("t-SNE Dim 2", fontsize=12)
    ax.set_title("Latent Manifold Coverage (Gen 4)", fontsize=13, fontweight="bold", pad=15)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=COLOR_GRAY, edgecolor="none", alpha=0.3, label="Reference"),
        Line2D([0], [0], color=COLOR_ORANGE, lw=1.6, label="No Filter"),
        Line2D([0], [0], color=COLOR_RED, lw=1.6, label="Pointwise"),
        Line2D([0], [0], color=COLOR_GREEN, lw=1.6, label="Dispersion"),
        Line2D([0], [0], color=COLOR_BLUE, lw=2.0, label="Set-Aware"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=9,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.26),
        columnspacing=1.0,
        handletextpad=0.6,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(top=0.80)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)


def main() -> None:
    root = Path("/root/autodl-tmp/ICML/STEP2")
    vendi_csv = root / "Total_results" / "Tables" / "exp12_Embedding_model" / "vendi_scores.csv"
    tsne_csv = root / "Total_results" / "Tables" / "exp12_Embedding_model" / "tsne_g4_coords.csv"
    out_dir = root / "Total_results" / "Figures" / "exp12_Embedding_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_vendi_curve(vendi_csv, out_dir / "vendi_scores_refined.png")
    plot_tsne_contours(tsne_csv, out_dir / "tsne_g4_refined.png")


if __name__ == "__main__":
    main()
