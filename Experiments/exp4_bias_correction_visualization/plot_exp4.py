import argparse
import pathlib
import shutil
import subprocess
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional dependency
    sns = None

# --- 视觉样式配置 ---
# 使用 Seaborn 风格作为底座（如果安装了）会让配色更柔和，
# 这里为了保持纯 Matplotlib 依赖，我们手动优化 rcParams
# Use LaTeX if available; otherwise fall back to mathtext.
USE_TEX = shutil.which("latex") is not None

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"],
        "mathtext.fontset": "cm",  # 数学公式字体类似 LaTeX
        "text.usetex": USE_TEX,    # 使用 LaTeX 渲染公式（若系统已安装）
        "axes.labelsize": 18,      # 增大坐标轴标签
        "axes.titlesize": 13,      # 子图标题更紧凑
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.grid": True,         # 默认开启网格
        "grid.alpha": 0.25,        # 网格透明度
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.6,    # 线条加粗
    }
)

if sns is not None:
    _palette = sns.color_palette("colorblind")
else:
    _palette = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#CA9161", "#FBAFE4", "#949494"]

COLORS = {
    "dist": _palette[0],
    "align": "#CC78BC",
    "delta": _palette[2],
    "target": "#000000",
    "ess": _palette[1],
    "var": "#7f7f7f",
}


def _moving_average(values: np.ndarray, window: int = 7) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")

def plot_exp41(metrics: Dict[str, Dict[str, List[float]]], target_norm: float, out_dir: pathlib.Path, n_seeds: int) -> None:
    gens = np.arange(1, len(metrics["delta_norm"]["mean"]) + 1)

    # 2x2 宽屏布局，适配双栏缩放
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.1, top=0.9, wspace=0.16, hspace=0.24)
    axs = axes.flatten()

    for ax in axs:
        ax.tick_params(axis="x", pad=0)
        ax.tick_params(axis="y", pad=2)

    def plot_mean_ci(ax, mean, std, label, color, alpha=0.12, linewidth=None, linestyle="-", zorder=3):
        ci = 1.96 * std / np.sqrt(n_seeds)
        lw = linewidth or plt.rcParams["lines.linewidth"]
        ax.plot(gens, mean, label=label, color=color, linewidth=lw, linestyle=linestyle, zorder=zorder)
        lower = np.clip(mean - ci, 1e-4, None)
        ax.fill_between(gens, lower, mean + ci, alpha=alpha, color=color, zorder=zorder - 1)

    # (a) 距离
    plot_mean_ci(
        axs[0],
        np.array(metrics["dist"]["mean"]),
        np.array(metrics["dist"]["std"]),
        r"$\|\Delta\boldsymbol{\phi} + \mathbf{b}\|_2$",
        color=COLORS["dist"],
    )
    axs[0].set_ylabel(r"$\|\Delta\boldsymbol{\phi} + \mathbf{b}\|_2$")
    axs[0].set_yscale("log")
    axs[0].yaxis.set_major_locator(mticker.FixedLocator([0.05, 0.1, 0.5]))
    axs[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{x:g}"))
    axs[0].yaxis.set_minor_locator(mticker.NullLocator())
    axs[0].legend(loc="upper right", frameon=True, framealpha=0.9)
    axs[0].set_title("(a) Convergence to Bias", fontweight="bold", fontsize=22)

    # (b) 方向
    plot_mean_ci(
        axs[1],
        np.array(metrics["cos"]["mean"]),
        np.array(metrics["cos"]["std"]),
        r"$\cos(\Delta\boldsymbol{\phi}, -\mathbf{b})$",
        color=COLORS["align"],
    )
    axs[1].set_ylabel("Cosine similarity")
    axs[1].set_ylim([-1.05, 1.05])
    axs[1].legend(loc="lower right", frameon=True, framealpha=0.9)
    axs[1].set_title("(b) Directional Alignment", fontweight="bold", fontsize=22)

    # (c) 范数
    plot_mean_ci(
        axs[2],
        np.array(metrics["delta_norm"]["mean"]),
        np.array(metrics["delta_norm"]["std"]),
        r"$\|\Delta\boldsymbol{\phi}\|_2$",
        color=COLORS["delta"],
        alpha=0.1,
        linewidth=2.8,
    )
    axs[2].lines[-1].set_alpha(0.85)
    axs[2].axhline(
        target_norm,
        color=COLORS["target"],
        linestyle=(0, (3, 2)),
        linewidth=1.4,
        label=r"$\|\mathbf{b}\|_2$",
        zorder=6,
    )
    axs[2].set_xlabel("Generation", labelpad=0)
    axs[2].set_ylabel(r"$\|\Delta\boldsymbol{\phi}\|_2$")
    axs[2].legend(loc="lower right", frameon=True, framealpha=0.9)
    axs[2].set_title("(c) Correction Magnitude", fontweight="bold", fontsize=22)

    # (d) ESS + Var
    ess_mean = np.array(metrics["ess"]["mean"], dtype=float)
    ess_std = np.array(metrics["ess"]["std"], dtype=float)
    ess_smooth = _moving_average(ess_mean, window=7)
    ess_ci = 1.96 * ess_std / np.sqrt(n_seeds)

    axs[3].plot(gens, ess_mean, color=COLORS["ess"], alpha=0.25, linewidth=1.2, zorder=1)
    axs[3].plot(gens, ess_smooth, color=COLORS["ess"], linewidth=2.8, label="ESS (smoothed)", zorder=3)
    axs[3].fill_between(gens, ess_mean - ess_ci, ess_mean + ess_ci, alpha=0.08, color=COLORS["ess"], zorder=0)

    ax3_twin = axs[3].twinx()
    var_mean = np.array(metrics["w_var"]["mean"], dtype=float)
    var_smooth = _moving_average(var_mean, window=7)
    ax3_twin.plot(gens, var_mean, color=COLORS["var"], alpha=0.2, linewidth=1.0, zorder=1)
    ax3_twin.plot(
        gens,
        var_smooth,
        color=COLORS["var"],
        linestyle=(0, (3, 2)),
        linewidth=2.0,
        label=r"$\mathrm{Var}(w)$ (smoothed)",
        zorder=2,
    )

    axs[3].set_xlabel("Generation", labelpad=0)
    axs[3].set_ylabel(r"$\mathrm{ESS}$", color=COLORS["ess"])
    ax3_twin.set_ylabel(r"$\mathrm{Var}(w)$", color=COLORS["var"])
    axs[3].tick_params(axis="y", colors=COLORS["ess"])
    ax3_twin.tick_params(axis="y", colors=COLORS["var"])

    lines1, labels1 = axs[3].get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    axs[3].legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.04),
        frameon=True,
        framealpha=0.8,
        edgecolor="black",
        fontsize=10,
        borderpad=0.3,
        labelspacing=0.3,
        handlelength=1.6,
        handletextpad=0.4,
    )
    axs[3].set_title("(d) Stability Metrics", fontweight="bold", fontsize=22)

    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / "exp4_4.1_base.svg"
    fig.savefig(svg_path)
    converter = shutil.which("rsvg-convert")
    if converter is not None:
        subprocess.run(
            [converter, str(svg_path), "-f", "pdf", "-o", str(out_dir / "exp4_4.1_base.pdf")],
            check=False,
        )
    plt.close(fig)


def plot_exp41_enhance(
    metrics: Dict[str, Dict[str, List[float]]], target_norm: float, out_dir: pathlib.Path, n_seeds: int
) -> None:
    """Focused view of Exp4.1 showing only alignment and magnitude panels."""
    gens = np.arange(1, len(metrics["delta_norm"]["mean"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

    # (b) Directional Alignment (matches original subplot b)
    ci_corr = 1.96 * np.array(metrics["cos"]["std"]) / np.sqrt(n_seeds)
    axes[0].plot(gens, metrics["cos"]["mean"], label=r"$\cos(\Delta\boldsymbol{\phi}, -\mathbf{b})$", color=COLORS["align"])
    axes[0].fill_between(
        gens,
        np.array(metrics["cos"]["mean"]) - ci_corr,
        np.array(metrics["cos"]["mean"]) + ci_corr,
        alpha=0.12,
        color=COLORS["align"],
    )
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Cosine similarity")
    axes[0].set_ylim([-1.05, 1.05])
    axes[0].legend(loc="lower right")
    axes[0].set_title("(b) Directional Alignment")

    # (c) Correction Magnitude (matches original subplot c)
    ci_delta = 1.96 * np.array(metrics["delta_norm"]["std"]) / np.sqrt(n_seeds)
    axes[1].plot(gens, metrics["delta_norm"]["mean"], label=r"$\|\Delta\boldsymbol{\phi}\|_2$", color=COLORS["delta"], alpha=0.85)
    axes[1].fill_between(
        gens,
        np.array(metrics["delta_norm"]["mean"]) - ci_delta,
        np.array(metrics["delta_norm"]["mean"]) + ci_delta,
        alpha=0.1,
        color=COLORS["delta"],
    )
    axes[1].axhline(
        target_norm,
        color=COLORS["target"],
        linestyle=(0, (3, 2)),
        linewidth=1.4,
        label=r"$\|\mathbf{b}\|_2$",
        zorder=6,
    )
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel(r"$\|\Delta\boldsymbol{\phi}\|_2$")
    axes[1].legend(loc="lower right")
    axes[1].set_title("(c) Correction Magnitude")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "exp4_4.1_base_enhance.png", dpi=300)
    plt.close(fig)


def plot_exp42(metrics: Dict[str, Dict[str, List[float]]], out_dir: pathlib.Path, n_seeds: int) -> None:
    gens = np.arange(1, len(metrics["theta_norm"]["mean"]) + 1)
    
    # 修改 2：同样改为 2x2 布局
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axs = axes.flatten()

    # --- Subplot 0: Cosine Sims ---
    ci_corr = 1.96 * np.array(metrics["cos_to_correction"]["std"]) / np.sqrt(n_seeds)
    ci_theta = 1.96 * np.array(metrics["cos_to_theta"]["std"]) / np.sqrt(n_seeds)
    
    axs[0].plot(gens, metrics["cos_to_correction"]["mean"], label=r"$\cos(\Delta\phi, \theta_t - \hat{\theta}_{ridge})$", color="tab:blue")
    axs[0].fill_between(gens, np.array(metrics["cos_to_correction"]["mean"]) - ci_corr, np.array(metrics["cos_to_correction"]["mean"]) + ci_corr, alpha=0.2, color="tab:blue")
    
    axs[0].plot(gens, metrics["cos_to_theta"]["mean"], label=r"$\cos(\Delta\phi, \theta_t)$", linestyle="--", color="tab:orange")
    axs[0].fill_between(gens, np.array(metrics["cos_to_theta"]["mean"]) - ci_theta, np.array(metrics["cos_to_theta"]["mean"]) + ci_theta, alpha=0.2, color="tab:orange")
    
    axs[0].set_xlabel("Generation")
    axs[0].set_ylabel(r"$\cos(\Delta\phi)$")
    axs[0].set_ylim([-1.1, 1.1])
    axs[0].legend(loc="best", fontsize=10)
    axs[0].set_title("(a) Orientation")

    # --- Subplot 1: Norms (Dual Axis) ---
    ci_delta = 1.96 * np.array(metrics["delta_norm"]["std"]) / np.sqrt(n_seeds)
    ci_bias = 1.96 * np.array(metrics["bias_norm"]["std"]) / np.sqrt(n_seeds)
    
    l1 = axs[1].plot(gens, metrics["delta_norm"]["mean"], label=r"$\|\Delta \phi\|$", color="tab:blue")[0]
    axs[1].fill_between(gens, np.array(metrics["delta_norm"]["mean"]) - ci_delta, np.array(metrics["delta_norm"]["mean"]) + ci_delta, alpha=0.2, color="tab:blue")
    
    l2 = axs[1].plot(gens, metrics["bias_norm"]["mean"], label=r"$\|\hat{\theta}_{ridge}-\theta_t\|$", linestyle="--", color="tab:red")[0]
    axs[1].fill_between(gens, np.array(metrics["bias_norm"]["mean"]) - ci_bias, np.array(metrics["bias_norm"]["mean"]) + ci_bias, alpha=0.2, color="tab:red")
    
    ax2_twin = axs[1].twinx()
    l3 = ax2_twin.plot(gens, metrics["theta_norm"]["mean"], color="tab:green", alpha=0.7, label=r"$\|\theta_t\|$")[0]
    
    axs[1].set_xlabel("Generation")
    axs[1].set_ylabel(r"$\|\Delta \phi\|,\ \|\hat{\theta}_{ridge}-\theta_t\|$")
    ax2_twin.set_ylabel(r"$\|\theta_t\|$")
    
    # Combine legends
    axs[1].legend([l1, l2, l3], [l1.get_label(), l2.get_label(), l3.get_label()], loc="upper left")
    axs[1].set_title("(b) Magnitude Dynamics")

    # --- Subplot 2: Scatter (Phase Portrait) ---
    sc = axs[2].scatter(metrics["theta_norm"]["mean"], metrics["delta_norm"]["mean"], 
                        c=gens, cmap="viridis", s=25, alpha=0.8, edgecolor='none')
    axs[2].set_xlabel(r"$\|\theta_t\|_2$")
    axs[2].set_ylabel(r"$\|\Delta \phi\|_2$")
    axs[2].set_title("(c) Phase Plot: Norm vs Correction")
    cb = fig.colorbar(sc, ax=axs[2], aspect=30)
    cb.set_label("Generation")

    # --- Subplot 3: ESS ---
    ci_ess = 1.96 * np.array(metrics["ess"]["std"]) / np.sqrt(n_seeds)
    l4 = axs[3].plot(gens, metrics["ess"]["mean"], label="ESS", color="tab:cyan")[0]
    axs[3].fill_between(gens, np.array(metrics["ess"]["mean"]) - ci_ess, np.array(metrics["ess"]["mean"]) + ci_ess, alpha=0.2, color="tab:cyan")
    
    ax4_twin = axs[3].twinx()
    l5 = ax4_twin.plot(gens, metrics["w_var"]["mean"], color="tab:purple", linestyle="--", label="Var(w)")[0]
    
    axs[3].set_xlabel("Generation")
    axs[3].set_ylabel(r"$\mathrm{ESS}$")
    ax4_twin.set_ylabel(r"$\mathrm{Var}(w)$")
    axs[3].legend([l4, l5], [l4.get_label(), l5.get_label()], loc="center right")
    axs[3].set_title("(d) Sampling Efficiency")

    fig.suptitle("Ridge Correction: Detailed Dynamics", fontweight='bold')
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "exp4_4.2_ridge_dynamics.png", dpi=300)
    plt.close(fig)


def plot_exp42_phase(metrics: Dict[str, Dict[str, List[float]]], out_dir: pathlib.Path, n_seeds: int) -> None:
    """Standalone phase portrait (subplot c) for ridge correction."""
    gens = np.arange(1, len(metrics["theta_norm"]["mean"]) + 1)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    sc = ax.scatter(
        metrics["theta_norm"]["mean"],
        metrics["delta_norm"]["mean"],
        c=gens,
        cmap="viridis",
        s=30,
        alpha=0.85,
        edgecolor="none",
    )
    ax.set_xlabel(r"$\|\theta_t\|_2$")
    ax.set_ylabel(r"$\|\Delta \phi\|_2$")
    ax.set_title("Ridge Correction: Norm vs Correction")
    cb = fig.colorbar(sc, ax=ax, aspect=30)
    cb.set_label("Generation")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "exp4_4.2_ridge_NvsC.png", dpi=300)
    plt.close(fig)


def plot_exp43(x_eval: np.ndarray, w_eval: np.ndarray, args: argparse.Namespace, out_dir: pathlib.Path) -> None:
    order = np.argsort(x_eval)
    x_sorted = x_eval[order]
    w_sorted = w_eval[order]

    bins = np.linspace(x_sorted.min(), x_sorted.max(), num=20) # 增加一点bin的数量
    bin_centers = []
    bin_means = []
    for i in range(len(bins) - 1):
        mask = (x_sorted >= bins[i]) & (x_sorted < bins[i + 1])
        if mask.any():
            bin_centers.append(0.5 * (bins[i] + bins[i + 1]))
            bin_means.append(w_sorted[mask].mean())

    # 保持单张图，但调整比例使其更协调
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    
    ax.scatter(x_eval, w_eval, alpha=0.4, s=25, color="gray", label="Sample weights", edgecolors='none')
    if bin_centers:
        ax.plot(bin_centers, bin_means, color="tab:red", linewidth=3, label="Binned mean trend")
        
    ax.axvline(args.mu_prior, color="tab:blue", linestyle="--", linewidth=2, label=f"Prior mean={args.mu_prior}")
    ax.axvline(args.bayes_true_mean, color="tab:green", linestyle="-.", linewidth=2, label=f"True mean={args.bayes_true_mean}")
    
    ax.set_xlabel("Sample value $x$")
    ax.set_ylabel(r"$w(x)$")
    ax.set_title("Weight Distribution Analysis")
    ax.legend(frameon=True, framealpha=0.9, loc='upper left')
    
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "exp4_4.3_bayes_weights.png", dpi=300)
    plt.close()


def plot_exp42_43_combined(
    metrics42: Dict[str, Dict[str, List[float]]],
    x_eval: np.ndarray,
    w_eval: np.ndarray,
    args: argparse.Namespace,
    out_dir: pathlib.Path,
    n_seeds: int,
) -> None:
    """Side-by-side view: Exp4.2 phase plot + Exp4.3 weight analysis."""
    gens = np.arange(1, len(metrics42["theta_norm"]["mean"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)

    # Left: phase portrait
    sc = axes[0].scatter(
        metrics42["theta_norm"]["mean"],
        metrics42["delta_norm"]["mean"],
        c=gens,
        cmap="viridis",
        s=30,
        alpha=0.85,
        edgecolor="none",
    )
    axes[0].set_xlabel(r"$\|\theta_t\|_2$")
    axes[0].set_ylabel(r"$\|\Delta \phi\|_2$")
    axes[0].set_title("Ridge Correction: Norm vs Correction")
    cb = fig.colorbar(sc, ax=axes[0], aspect=30)
    cb.set_label("Generation")

    # Right: weights (reuse plot_exp43 styling)
    order = np.argsort(x_eval)
    x_sorted = x_eval[order]
    w_sorted = w_eval[order]
    bins = np.linspace(x_sorted.min(), x_sorted.max(), num=20)
    bin_centers = []
    bin_means = []
    for i in range(len(bins) - 1):
        mask = (x_sorted >= bins[i]) & (x_sorted < bins[i + 1])
        if mask.any():
            bin_centers.append(0.5 * (bins[i] + bins[i + 1]))
            bin_means.append(w_sorted[mask].mean())

    ax = axes[1]
    ax.scatter(x_eval, w_eval, alpha=0.4, s=25, color="gray", label="Sample weights", edgecolors="none")
    if bin_centers:
        ax.plot(bin_centers, bin_means, color="tab:red", linewidth=3, label="Binned mean trend")

    ax.axvline(args.mu_prior, color="tab:blue", linestyle="--", linewidth=2, label=f"Prior mean={args.mu_prior}")
    ax.axvline(args.bayes_true_mean, color="tab:green", linestyle="-.", linewidth=2, label=f"True mean={args.bayes_true_mean}")

    ax.set_xlabel("Sample value $x$")
    ax.set_ylabel(r"$w(x)$")
    ax.set_title("Weight Distribution Analysis")
    ax.legend(frameon=True, framealpha=0.9, loc="upper left")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "exp4_4.2,3_enhence.png", dpi=300)
    plt.close(fig)
