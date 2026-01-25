#!/usr/bin/env python3
"""
Generate the camera-ready hero spiral plot for ICML 2026.
Optimized for readability, print safety, and visual hierarchy.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np


OUT_PAPER = (
    "/root/autodl-tmp/ICML/STEP2/Paper/LaTEX/icml2025/"
    "Figures/hero_spiral_final_polished.png"
)
OUT_RESULTS = (
    "/root/autodl-tmp/ICML/STEP2/Total_results/Figures/hero_spiral_final_polished.png"
)


def configure_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-paper")
    except Exception:
        pass

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "cm",
            "figure.figsize": (6, 5.5),
            "figure.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.bottom": False,
            "axes.spines.left": False,
        }
    )


def plot_final_hero() -> plt.Figure:
    fig, ax = plt.subplots()

    bias_radius = 2.0
    start_radius = 5.8
    t = np.linspace(0, 12, 1200)

    theta = t * 2.0

    np.random.seed(42)
    decay_red = (start_radius - bias_radius) * np.exp(-0.4 * t) + bias_radius
    base_noise = np.random.normal(0, 0.12, t.shape) * (1 - np.exp(-0.55 * t))
    smooth_kernel = np.ones(7) / 7.0
    noise = np.convolve(base_noise, smooth_kernel, mode="same")
    jagged = np.random.normal(0, 0.18, t.shape)
    jagged_kernel = np.ones(3) / 3.0
    jagged = np.convolve(jagged, jagged_kernel, mode="same")
    jagged_scale = np.max(np.abs(jagged)) or 1.0
    jagged = 0.22 * (jagged / jagged_scale) * (1 - np.exp(-0.6 * t))
    low_noise = np.random.normal(0, 0.10, t.shape)
    low_kernel = np.ones(41) / 41.0
    low_noise = np.convolve(low_noise, low_kernel, mode="same")
    low_scale = np.max(np.abs(low_noise)) or 1.0
    tail_gate = np.clip((t - 7.5) / 3.0, 0.0, 1.0)
    tail_osc = 0.16 * (low_noise / low_scale) * tail_gate

    rw = np.cumsum(np.random.normal(0, 0.03, t.shape))
    rw_kernel = np.ones(31) / 31.0
    rw = np.convolve(rw, rw_kernel, mode="same")
    rw_scale = np.max(np.abs(rw)) or 1.0
    tail_wander = 0.30 * (rw / rw_scale) * tail_gate
    tail_sine = 0.10 * np.sin(1.6 * t + 0.7) * tail_gate

    r_red_raw = decay_red + noise + tail_osc + jagged
    floor_target = bias_radius + tail_wander + tail_sine
    r_red = (1 - tail_gate) * r_red_raw + tail_gate * floor_target
    r_red = np.maximum(r_red, bias_radius * 0.80)
    x_red = r_red * np.cos(theta)
    y_red = r_red * np.sin(theta)

    r_blue = start_radius * np.exp(-0.45 * t)
    x_blue = r_blue * np.cos(theta)
    y_blue = r_blue * np.sin(theta)

    x_red[0], y_red[0] = start_radius, 0
    x_blue[0], y_blue[0] = start_radius, 0

    circle = plt.Circle(
        (0, 0),
        bias_radius,
        color="#808080",
        fill=False,
        linestyle=(0, (5, 5)),
        linewidth=1.8,
        alpha=0.85,
        zorder=3,
    )
    ax.add_artist(circle)

    ax.scatter(
        [0],
        [0],
        color="white",
        marker="o",
        s=240,
        zorder=9,
        linewidth=0,
    )
    ax.scatter(
        [0],
        [0],
        color="black",
        marker="+",
        s=220,
        zorder=11,
        linewidth=2.8,
        label=r"$\theta^*$",
    )

    near_mask = r_red <= bias_radius * 1.15
    x_red_far = x_red.copy()
    y_red_far = y_red.copy()
    x_red_far[near_mask] = np.nan
    y_red_far[near_mask] = np.nan
    red_line = "#d62728"
    ax.plot(x_red_far, y_red_far, color=red_line, linewidth=1.6, alpha=0.6, zorder=2)

    x_red_near = x_red.copy()
    y_red_near = y_red.copy()
    x_red_near[~near_mask] = np.nan
    y_red_near[~near_mask] = np.nan
    ax.plot(x_red_near, y_red_near, color=red_line, linewidth=1.6, alpha=0.5, zorder=2)
    ax.plot(x_blue, y_blue, color="#005b96", linewidth=3.5, alpha=1.0, zorder=5)

    mid = 220
    arrow = ax.arrow(
        x_blue[mid],
        y_blue[mid],
        x_blue[mid + 1] - x_blue[mid],
        y_blue[mid + 1] - y_blue[mid],
        shape="full",
        lw=0,
        head_width=0.4,
        head_length=0.5,
        color="#005b96",
        zorder=6,
    )
    arrow.set_path_effects(
        [path_effects.Stroke(linewidth=3.0, foreground="white"), path_effects.Normal()]
    )

    halo = [path_effects.Stroke(linewidth=4.5, foreground="white"), path_effects.Normal()]

    red_text = red_line
    blue_text = "#004c99"

    ax.annotate(
        r"Bias Floor $(\beta/c)$",
        xy=(bias_radius * np.cos(np.pi / 4), bias_radius * np.sin(np.pi / 4)),
        xytext=(2.8, 2.9),
        arrowprops=dict(
            arrowstyle="->",
            color="#333333",
            connectionstyle="arc3,rad=.2",
            linewidth=1.8,
        ),
        fontsize=17,
        color=red_text,
        family="serif",
        path_effects=halo,
        zorder=20,
    )

    ax.text(
        -5.8,
        5.0,
        "Pointwise Filter",
        color=red_text,
        fontsize=17,
        fontweight="bold",
        ha="left",
        path_effects=halo,
        zorder=20,
    )
    ax.text(
        -5.8,
        4.3,
        "Stalled by systematic drift",
        color="#C0392B",
        fontsize=14,
        style="italic",
        ha="left",
        path_effects=halo,
        zorder=20,
    )

    ax.text(
        2.5,
        -4.5,
        "Set-Aware Filter",
        color=blue_text,
        fontsize=17,
        fontweight="bold",
        ha="left",
        path_effects=halo,
        zorder=20,
    )
    ax.text(
        2.6,
        -5.2,
        "Breaks the floor via",
        color="#0066cc",
        fontsize=14,
        style="italic",
        ha="left",
        path_effects=halo,
        zorder=20,
    )
    ax.text(
        2.6,
        -5.9,
        "geometric correction",
        color="#0066cc",
        fontsize=14,
        style="italic",
        ha="left",
        path_effects=halo,
        zorder=20,
    )

    ax.set_aspect("equal")
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-6.5, 6.5)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    return fig


def main() -> None:
    configure_style()
    fig = plot_final_hero()

    out_paths = [Path(OUT_PAPER), Path(OUT_RESULTS)]
    for path in out_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
