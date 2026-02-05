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
COLOR_GRAY = "#BBBBBB"  # Reference


def plot_vendi_curve(csv_path: Path, output_path: Path) -> None:
    df = pd.read_csv(csv_path)
    # Expect columns: method, generation, vendi, vendi_std
    df = df.rename(columns={"generation": "Generation"})
    fig, ax = plt.subplots(figsize=(5, 4))

    pw = df[df["method"] == "pointwise"].sort_values("Generation")
    sa = df[df["method"] == "set_aware"].sort_values("Generation")

    ax.plot(pw["Generation"], pw["vendi"], color=COLOR_RED, lw=2.5, marker="^")
    ax.plot(sa["Generation"], sa["vendi"], color=COLOR_BLUE, lw=2.5, marker="o")

    # Direct labeling (no legend)
    ax.text(
        pw["Generation"].iloc[-1] + 0.1,
        pw["vendi"].iloc[-1],
        "Pointwise\n(Collapse)",
        color=COLOR_RED,
        va="center",
        fontweight="bold",
        fontsize=10,
    )
    ax.text(
        sa["Generation"].iloc[-1] + 0.1,
        sa["vendi"].iloc[-1],
        "Set-Aware\n(Preserved)",
        color=COLOR_BLUE,
        va="center",
        fontweight="bold",
        fontsize=10,
    )

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Vendi Score (Semantic Volume)", fontsize=12)
    ax.set_title("Topological Volume Stability", fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)


def plot_tsne_contours(csv_path: Path, output_path: Path) -> None:
    df = pd.read_csv(csv_path)
    # Expect columns: label, x, y
    df = df.rename(columns={"label": "method"})
    df["method"] = df["method"].replace(
        {"reference": "Reference", "pointwise": "Pointwise", "set_aware": "SetAware"}
    )

    fig, ax = plt.subplots(figsize=(6, 6))

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
    ax.scatter(subset_ref["x"], subset_ref["y"], c=COLOR_GRAY, s=5, alpha=0.1, label="_nolegend_")

    # Pointwise (collapsed)
    subset_pw = df[df["method"] == "Pointwise"]
    sns.kdeplot(
        data=subset_pw,
        x="x",
        y="y",
        color=COLOR_RED,
        levels=4,
        linewidths=1.5,
        warn_singular=False,
        ax=ax,
    )

    # Set-Aware (coverage)
    subset_sa = df[df["method"] == "SetAware"]
    sns.kdeplot(
        data=subset_sa,
        x="x",
        y="y",
        color=COLOR_BLUE,
        levels=4,
        linewidths=2.0,
        warn_singular=False,
        ax=ax,
    )

    ax.text(
        0.05,
        0.95,
        "Reference Distribution (Gray Area)",
        transform=ax.transAxes,
        color="gray",
        fontsize=10,
        style="italic",
    )
    ax.text(
        0.05,
        0.90,
        "Set-Aware (Blue Contours)",
        transform=ax.transAxes,
        color=COLOR_BLUE,
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        0.05,
        0.85,
        "Pointwise (Red Contours)",
        transform=ax.transAxes,
        color=COLOR_RED,
        fontsize=10,
        fontweight="bold",
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_title("Latent Manifold Coverage (Gen 4)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    experiments_root = script_dir.parent
    total_results = experiments_root / "Total_results"

    vendi_csv = total_results / "Tables" / "exp13_Qwen_model" / "vendi_scores.csv"
    tsne_csv = total_results / "Tables" / "exp13_Qwen_model" / "tsne_g4_coords.csv"
    out_dir = total_results / "Figures" / "exp13_Qwen_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_vendi_curve(vendi_csv, out_dir / "vendi_scores_refined.png")
    plot_tsne_contours(tsne_csv, out_dir / "tsne_g4_refined.png")


if __name__ == "__main__":
    main()
