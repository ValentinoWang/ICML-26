"""
Plot CIFAR-10 recursive self-training results (Overall / Worst-Class Acc + pseudo-label hist + ESS).
Times-style plotting for paper figures.

Usage:
    python exp9_cifar10_setaware/plot_exp9_times.py --results-dir exp9_cifar10_setaware/results
"""

import argparse
import csv
import pathlib
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BASE_TABLES_DIR = ROOT / "Total_results" / "Tables" / SCRIPT_DIR.name
BASE_FIGURES_DIR = ROOT / "Total_results" / "Figures" / SCRIPT_DIR.name
TABLES_DIR = BASE_TABLES_DIR / "results"
FIGURES_DIR = BASE_FIGURES_DIR / "results"

COLORS = {
    "baseline": "#7f8c8d",
    "set_aware": "#1f77b4",
    "k_center": "#ff7f0e",
    "pointwise": "#d62728",
    "dpp": "#2ca02c",
}

def load_merged_results(results_dir: pathlib.Path, method: str) -> Dict[int, List[Dict]]:
    per_gen: Dict[int, List[Dict]] = {}
    for path in sorted(results_dir.glob("exp9_seed*_merged.csv")):
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("method") != method:
                    continue
                g = int(row["generation"])
                m: Dict[str, float | List[float]] = {
                    "generation": g,
                    "train_size": float(row["train_size"]) if row["train_size"] else 0.0,
                    "acc": float(row["acc"]) if row["acc"] else 0.0,
                    "worst_class_acc": float(row["worst_class_acc"]) if row["worst_class_acc"] else 0.0,
                    "ess_score": float(row["ess_score"]) if row.get("ess_score") else np.nan,
                }
                per_acc = [float(row.get(f"acc_c{i}", 0) or 0) for i in range(10)]
                hist = [float(row.get(f"hist_c{i}", 0) or 0) for i in range(10)]
                m["per_class_acc"] = per_acc
                m["pseudo_label_hist"] = hist
                per_gen.setdefault(g, []).append(m)
    return per_gen


def aggregate_metrics(per_gen: Dict[int, List[Dict]]) -> Dict[str, Dict[str, List[float]]]:
    gens = sorted(per_gen.keys())
    acc_mean, acc_std = [], []
    worst_mean, worst_std = [], []
    ess_mean, ess_std = [], []
    hist_frac: List[np.ndarray] = []
    for g in gens:
        ms = per_gen[g]
        acc_vals = [m.get("acc", np.nan) for m in ms]
        worst_vals = [m.get("worst_class_acc", np.nan) for m in ms]
        ess_vals = [m.get("ess_score", np.nan) for m in ms]

        def _safe_stats(vals: List[float]) -> Tuple[float, float]:
            arr = np.asarray(vals, dtype=np.float32)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return 0.0, 0.0
            return float(arr.mean()), float(arr.std())

        acc_m, acc_s = _safe_stats(acc_vals)
        worst_m, worst_s = _safe_stats(worst_vals)
        ess_m, ess_s = _safe_stats(ess_vals)

        acc_mean.append(acc_m)
        acc_std.append(acc_s)
        worst_mean.append(worst_m)
        worst_std.append(worst_s)
        ess_mean.append(ess_m)
        ess_std.append(ess_s)
        hists = []
        for m in ms:
            hist = np.asarray(m.get("pseudo_label_hist", [0] * 10), dtype=np.float32)
            total = hist.sum()
            if total > 0:
                hist = hist / total
            hists.append(hist)
        hist_frac.append(np.mean(hists, axis=0) if hists else np.zeros(10, dtype=np.float32))
    return {
        "gens": gens,
        "acc": {"mean": acc_mean, "std": acc_std},
        "worst": {"mean": worst_mean, "std": worst_std},
        "ess": {"mean": ess_mean, "std": ess_std},
        "hist": hist_frac,
    }


def compute_selection_metrics(per_gen: Dict[int, List[Dict]]) -> Dict[str, List[float]]:
    """
    Compute imbalance metrics per generation using normalized pseudo-label histograms.
    Returns mean/std for selection standard deviation and KL divergence to uniform.
    """
    gens = sorted(per_gen.keys())
    std_mean, std_std = [], []
    kl_mean, kl_std = [], []
    for g in gens:
        hists = []
        for m in per_gen[g]:
            hist = np.asarray(m.get("pseudo_label_hist", [0] * 10), dtype=np.float32)
            total = hist.sum()
            if total <= 0:
                continue
            p = hist / total
            hists.append(p)
        if not hists:
            std_mean.append(np.nan)
            std_std.append(0.0)
            kl_mean.append(np.nan)
            kl_std.append(0.0)
            continue
        std_vals = [float(p.std()) for p in hists]
        kl_vals = [float((p * (np.log(p + 1e-12) - np.log(0.1))).sum()) for p in hists]
        std_mean.append(float(np.mean(std_vals)))
        std_std.append(float(np.std(std_vals)))
        kl_mean.append(float(np.mean(kl_vals)))
        kl_std.append(float(np.std(kl_vals)))
    return {
        "gens": gens,
        "std_mean": std_mean,
        "std_std": std_std,
        "kl_mean": kl_mean,
        "kl_std": kl_std,
    }


def average_hist_counts(per_gen: Dict[int, List[Dict]], generation: int) -> np.ndarray:
    hists = []
    for m in per_gen.get(generation, []):
        hist = np.asarray(m.get("pseudo_label_hist", [0] * 10), dtype=np.float32)
        hists.append(hist)
    if not hists:
        return np.zeros(10, dtype=np.float32)
    return np.mean(hists, axis=0)


def _metric_mean_std(per_gen: Dict[int, List[Dict]], generation: int, key: str) -> Tuple[float, float, int]:
    vals = [float(m.get(key, np.nan)) for m in per_gen.get(generation, [])]
    arr = np.asarray(vals, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0, 0
    return float(arr.mean()), float(arr.std()), int(arr.size)


def _worst_classes_by_accuracy(per_gen: Dict[int, List[Dict]], generation: int, k: int = 3) -> np.ndarray:
    rows = [m.get("per_class_acc", []) for m in per_gen.get(generation, [])]
    if not rows:
        return np.arange(k, dtype=int)
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return np.arange(k, dtype=int)
    mean_acc = np.nanmean(arr, axis=0)
    return np.argsort(mean_acc)[:k]


def plot_worstclass_hist_small(
    per_gen_pointwise: Dict[int, List[Dict]],
    per_gen_setaware: Dict[int, List[Dict]],
    per_gen_diversity: Dict[int, List[Dict]],
    generation: int,
    out_path: pathlib.Path,
    diversity_label: str,
    diversity_color: str,
    minority_classes: Tuple[int, ...] | None = None,
) -> None:
    """Two-panel summary: worst-class accuracy + minority-class histogram."""
    colors = COLORS
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.4, 2.7))

    # (a) Worst-class accuracy bars
    pw_mean, pw_std, pw_n = _metric_mean_std(per_gen_pointwise, generation, "worst_class_acc")
    div_mean, div_std, div_n = _metric_mean_std(per_gen_diversity, generation, "worst_class_acc")
    sa_mean, sa_std, sa_n = _metric_mean_std(per_gen_setaware, generation, "worst_class_acc")
    pw_ci = 1.96 * pw_std / np.sqrt(max(pw_n, 1))
    div_ci = 1.96 * div_std / np.sqrt(max(div_n, 1))
    sa_ci = 1.96 * sa_std / np.sqrt(max(sa_n, 1))

    labels = ["Pointwise", "Diversity", "Ours"]
    means = [pw_mean * 100.0, div_mean * 100.0, sa_mean * 100.0]
    cis = [pw_ci * 100.0, div_ci * 100.0, sa_ci * 100.0]
    x = np.arange(len(labels)) * 1.5
    bars = ax_left.bar(
        x,
        means,
        yerr=cis,
        capsize=3,
        color=[colors["baseline"], diversity_color, colors["set_aware"]],
        edgecolor="#333333",
        linewidth=0.6,
    )
    bars[0].set_hatch("//")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels)
    ax_left.set_ylabel("Worst-class accuracy (%)", fontsize=11)
    ax_left.set_title("(a) Worst-Class Accuracy (Gen 5)", fontsize=11, pad=2)
    ymax = max(means) + max(cis) if means else 0.0
    ax_left.set_ylim(0.0, max(45.0, ymax * 1.35))
    ax_left.grid(axis="y", alpha=0.25)
    ax_left.set_xlim(x[0] - 0.8, x[-1] + 0.8)
    ax_left.margins(x=0)

    # (b) Minority-class histogram (counts)
    hist_pw = average_hist_counts(per_gen_pointwise, generation)
    hist_div = average_hist_counts(per_gen_diversity, generation)
    hist_sa = average_hist_counts(per_gen_setaware, generation)
    if minority_classes is None:
        worst = _worst_classes_by_accuracy(per_gen_pointwise, generation, k=3)
        classes = np.sort(worst).astype(int)
    else:
        classes = np.array(minority_classes, dtype=int)
    width = 0.23
    ax_right.bar(
        classes - width,
        hist_pw[classes],
        width=width,
        color=colors["baseline"],
        edgecolor="#333333",
        linewidth=0.6,
        label="Pointwise",
        hatch="//",
    )
    ax_right.bar(
        classes,
        hist_div[classes],
        width=width,
        color=diversity_color,
        edgecolor="#333333",
        linewidth=0.6,
        label=diversity_label,
    )
    ax_right.bar(
        classes + width,
        hist_sa[classes],
        width=width,
        color=colors["set_aware"],
        edgecolor="#333333",
        linewidth=0.6,
        label="Set-Aware",
    )
    ax_right.set_xticks(classes)
    ax_right.set_ylabel("Pseudo-label count", fontsize=11)
    ax_right.set_title("(b) Class Histogram (Minority)", fontsize=11, pad=2)
    max_hist = max(float(hist_pw[classes].max()), float(hist_div[classes].max()), float(hist_sa[classes].max()))
    ax_right.set_ylim(0.0, max_hist * 1.5 if max_hist > 0 else 1.0)
    ax_right.grid(axis="y", alpha=0.25)

    ax_left.legend(
        [bars[0], bars[1], bars[2]],
        ["Pointwise", diversity_label, "Set-Aware"],
        loc="upper left",
        frameon=False,
        fontsize=9,
    )
    ax_right.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_curves(
    agg_baseline: Dict[str, Dict[str, List[float]]],
    agg_setaware: Dict[str, Dict[str, List[float]]],
    out_path: pathlib.Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    colors = COLORS

    # Overall Acc
    ax = axes[0, 0]
    ax.set_title("Overall Accuracy")
    legend_handles = []
    legend_labels = []
    for name, agg in [("Baseline", agg_baseline), ("Set-Aware", agg_setaware)]:
        line, = ax.plot(
            agg["gens"],
            agg["acc"]["mean"],
            label=name,
            color=colors[name.lower().replace("-", "_")],
        )
        if name not in legend_labels:
            legend_handles.append(line)
            legend_labels.append(name)
        ax.fill_between(
            agg["gens"],
            np.array(agg["acc"]["mean"]) - np.array(agg["acc"]["std"]),
            np.array(agg["acc"]["mean"]) + np.array(agg["acc"]["std"]),
            color=colors[name.lower().replace("-", "_")],
            alpha=0.18,
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Accuracy")
    max_acc = max(
        np.max(np.array(agg_baseline["acc"]["mean"]) + np.array(agg_baseline["acc"]["std"])),
        np.max(np.array(agg_setaware["acc"]["mean"]) + np.array(agg_setaware["acc"]["std"])),
    )
    min_acc = min(
        np.min(np.array(agg_baseline["acc"]["mean"]) - np.array(agg_baseline["acc"]["std"])),
        np.min(np.array(agg_setaware["acc"]["mean"]) - np.array(agg_setaware["acc"]["std"])),
    )
    pad = 0.02
    ax.set_ylim(max(0.2, float(min_acc - pad)), min(1.0, float(max_acc + pad)))
    # Legend moved to figure-level.

    # Worst-Class Acc
    ax = axes[0, 1]
    ax.set_title("Worst-Class Accuracy")
    for name, agg in [("Baseline", agg_baseline), ("Set-Aware", agg_setaware)]:
        ax.plot(agg["gens"], agg["worst"]["mean"], label=name, color=colors[name.lower().replace("-", "_")])
        ax.fill_between(
            agg["gens"],
            np.array(agg["worst"]["mean"]) - np.array(agg["worst"]["std"]),
            np.array(agg["worst"]["mean"]) + np.array(agg["worst"]["std"]),
            color=colors[name.lower().replace("-", "_")],
            alpha=0.18,
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Accuracy")
    max_worst = max(
        np.max(np.array(agg_baseline["worst"]["mean"]) + np.array(agg_baseline["worst"]["std"])),
        np.max(np.array(agg_setaware["worst"]["mean"]) + np.array(agg_setaware["worst"]["std"])),
    )
    min_worst = min(
        np.min(np.array(agg_baseline["worst"]["mean"]) - np.array(agg_baseline["worst"]["std"])),
        np.min(np.array(agg_setaware["worst"]["mean"]) - np.array(agg_setaware["worst"]["std"])),
    )
    pad = 0.02
    ax.set_ylim(max(0.0, float(min_worst - pad)), min(1.0, float(max_worst + pad)))
    # Legend moved to figure-level.

    # ESS
    ax = axes[1, 0]
    ax.set_title("ESS (Set-Aware)")
    ax.plot(agg_setaware["gens"], agg_setaware["ess"]["mean"], color=colors["set_aware"], label="Set-Aware ESS")
    ax.fill_between(
        agg_setaware["gens"],
        np.array(agg_setaware["ess"]["mean"]) - np.array(agg_setaware["ess"]["std"]),
        np.array(agg_setaware["ess"]["mean"]) + np.array(agg_setaware["ess"]["std"]),
        color=colors["set_aware"],
        alpha=0.18,
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("ESS")
    # Legend moved to figure-level.

    # Final generation class distribution (fraction)
    ax = axes[1, 1]
    ax.set_title("Pseudo-Label Class Fraction (Final Gen)")
    classes = np.arange(10)
    width = 0.38
    hist_base = agg_baseline["hist"][-1] if agg_baseline["hist"] else np.zeros(10)
    hist_set = agg_setaware["hist"][-1] if agg_setaware["hist"] else np.zeros(10)
    ax.bar(classes - width / 2, hist_base, width=width, label="Baseline", color=colors["baseline"])
    ax.bar(classes + width / 2, hist_set, width=width, label="Set-Aware", color=colors["set_aware"])
    ax.set_xlabel("Class")
    ax.set_ylabel("Fraction")
    ax.set_xticks(classes)
    # Legend moved to figure-level.

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(legend_labels),
        frameon=False,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.84))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_selection_trend(
    metric_name: str,
    trend_baseline: Dict[str, List[float]],
    trend_setaware: Dict[str, List[float]],
    out_path: pathlib.Path,
) -> None:
    metric_key = f"{metric_name}_mean"
    std_key = f"{metric_name}_std"
    label = "Selection Std" if metric_name == "std" else "KL to Uniform"
    colors = COLORS
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, trend in [("Baseline", trend_baseline), ("Set-Aware", trend_setaware)]:
        y = np.array(trend.get(metric_key, []), dtype=np.float32)
        yerr = np.array(trend.get(std_key, []), dtype=np.float32)
        gens = trend.get("gens", [])
        ax.plot(gens, y, label=name, color=colors[name.lower().replace("-", "_")], marker="o")
        ax.fill_between(gens, y - yerr, y + yerr, color=colors[name.lower().replace("-", "_")], alpha=0.18)
    ax.set_xlabel("Generation")
    ax.set_ylabel(label)
    ax.set_title(f"Selection Imbalance Trend ({label})")
    ax.legend(frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_combined_wide(
    agg_baseline: Dict[str, Dict[str, List[float]]],
    agg_setaware: Dict[str, Dict[str, List[float]]],
    per_gen_baseline: Dict[int, List[Dict]],
    per_gen_setaware: Dict[int, List[Dict]],
    out_path: pathlib.Path,
    hist_generation: int = 1,
) -> None:
    """
    1x3 compact plot; histogram uses log scale (no inset). Fonts and size tuned for ICML cross-column width.
    """
    # 【修改1】跨栏尺寸与小字体
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), constrained_layout=True)
    mpl.rcParams.update({"font.size": 8, "legend.fontsize": 7, "axes.labelsize": 8})

    colors = COLORS
    base_x = np.array(agg_baseline["gens"], dtype=np.float32) - 0.05
    set_x = np.array(agg_setaware["gens"], dtype=np.float32) + 0.05

    # (a) Overall Accuracy
    ax = axes[0]
    for name, agg, xs, style in [
        ("Baseline", agg_baseline, base_x, {"linestyle": "--", "marker": "o"}),
        ("Set-Aware", agg_setaware, set_x, {"linestyle": "-", "marker": "o"}),
    ]:
        color = colors[name.lower().replace("-", "_")]
        ax.plot(xs, agg["acc"]["mean"], label=name, color=color, **style)
        ax.fill_between(
            xs,
            np.array(agg["acc"]["mean"]) - np.array(agg["acc"]["std"]),
            np.array(agg["acc"]["mean"]) + np.array(agg["acc"]["std"]),
            color=color,
            alpha=0.12,
        )
    ax.set_xlabel("Generation", labelpad=2)
    ax.set_ylabel("Accuracy", labelpad=2)
    ax.set_title("(a) Overall Accuracy", fontsize=9, y=1.02)
    ax.set_ylim(0.25, 0.5)
    ax.set_xticks(agg_baseline["gens"])

    # (b) Worst-Class Accuracy
    ax = axes[1]
    for name, agg, xs, style in [
        ("Baseline", agg_baseline, base_x, {"linestyle": "--", "marker": "o"}),
        ("Set-Aware", agg_setaware, set_x, {"linestyle": "-", "marker": "o"}),
    ]:
        color = colors[name.lower().replace("-", "_")]
        ax.plot(xs, agg["worst"]["mean"], label=name, color=color, **style)
        ax.fill_between(
            xs,
            np.array(agg["worst"]["mean"]) - np.array(agg["worst"]["std"]),
            np.array(agg["worst"]["mean"]) + np.array(agg["worst"]["std"]),
            color=color,
            alpha=0.12,
        )
    ax.set_xlabel("Generation", labelpad=2)
    ax.set_ylabel("Accuracy", labelpad=2)
    ax.set_title("(b) Worst-Class Acc", fontsize=9, y=1.02)
    ax.set_ylim(0.0, 0.25)
    ax.set_xticks(agg_baseline["gens"])

    # 【修改2】共享图例放顶部
    lines_labels = [axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False, fontsize=8)

    # (c) Selection Histogram (log scale)
    ax = axes[2]
    hist_base = average_hist_counts(per_gen_baseline, hist_generation)
    hist_set = average_hist_counts(per_gen_setaware, hist_generation)
    classes = np.arange(10)
    width = 0.38
    ax.set_yscale("log")
    ax.bar(classes - width / 2, hist_base, width=width, label="Baseline", color=colors["baseline"])
    ax.bar(classes + width / 2, hist_set, width=width, label="Set-Aware", color=colors["set_aware"])

    # 理想平衡线
    ax.axhline(400, color="#444", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.text(10.2, 400, "Ideal", va="center", ha="left", fontsize=7, color="#444")

    ax.set_title(f"(c) Class Dist. (Gen {hist_generation})", fontsize=9, y=1.02)
    ax.set_xlabel("Class", labelpad=2)
    ax.set_ylabel("Count (Log Scale)", labelpad=0)
    ax.set_xticks(classes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pseudo_hist_generation(
    per_gen_baseline: Dict[int, List[Dict]],
    per_gen_setaware: Dict[int, List[Dict]],
    generation: int,
    out_path: pathlib.Path,
    extra_methods: List[Tuple[str, Dict[int, List[Dict]], str]] | None = None,
    show_legend: bool = True,
) -> None:
    """Bar chart of pseudo-label fraction at a specific generation."""
    colors = COLORS
    hist_base = average_hist_counts(per_gen_baseline, generation)
    hist_set = average_hist_counts(per_gen_setaware, generation)
    def _normalize_or_uniform(hist: np.ndarray) -> np.ndarray:
        total = hist.sum()
        if total > 0:
            return hist / total
        if generation == 0:
            return np.ones_like(hist, dtype=np.float32) / len(hist)
        return hist
    hist_base = _normalize_or_uniform(hist_base)
    hist_set = _normalize_or_uniform(hist_set)

    classes = np.arange(10)
    series: List[Tuple[str, np.ndarray, str]] = [
        ("Baseline", hist_base, "baseline"),
        ("Set-Aware", hist_set, "set_aware"),
    ]
    if extra_methods:
        for label, per_gen, color_key in extra_methods:
            hist_extra = average_hist_counts(per_gen, generation)
            if hist_extra.sum() > 0:
                hist_extra = hist_extra / hist_extra.sum()
            elif generation == 0:
                hist_extra = np.ones_like(hist_extra, dtype=np.float32) / len(hist_extra)
            series.append((label, hist_extra, color_key))

    width = 0.8 / max(2, len(series))
    offsets = np.linspace(-(len(series) - 1) / 2, (len(series) - 1) / 2, len(series)) * width
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    for (label, hist, color_key), offset in zip(series, offsets):
        ax.bar(classes + offset, hist, width=width, label=label, color=colors.get(color_key, "#333333"))
    ax.set_xlabel("Class", fontsize=10)
    ax.set_ylabel("Fraction", fontsize=10)
    ax.set_title(f"Pseudo-Label Distribution (Gen {generation})", fontsize=11, pad=2)
    ax.set_xticks(classes)
    max_val = max(float(np.max(h)) for _, h, _ in series) if series else 0.0
    if max_val > 0:
        ax.set_ylim(0.0, max_val * 1.35)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncol=len(series),
            frameon=False,
            fontsize=10,
        )
    ax.grid(alpha=0.25, axis="y")
    # Highlight minority recovery for classes 3 & 4
    ymin, ymax = ax.get_ylim()
    rect = mpl.patches.Rectangle(
        (2.5, ymin),
        2.0,
        ymax - ymin,
        fill=False,
        linestyle="--",
        linewidth=1.6,
        edgecolor="#c0392b",
        alpha=0.9,
    )
    ax.add_patch(rect)
    if generation > 0:
        peak = max(hist_base[3], hist_base[4], hist_set[3], hist_set[4])
        ax.annotate(
            "Minority Recovered",
            xy=(3.5, peak),
            xytext=(5.6, peak + 0.04),
            fontsize=10,
            color="#c0392b",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.6),
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if show_legend:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    else:
        plt.savefig(out_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot exp9 CIFAR-10 recursive self-training curves (Times font).")
    parser.add_argument("--results-dir", type=str, default=str(TABLES_DIR))
    parser.add_argument("--out", type=str, default=str(FIGURES_DIR / "exp9_plots_times.png"))
    parser.add_argument(
        "--trend-out",
        type=str,
        default=str(FIGURES_DIR / "exp9_selection_trend.png"),
        help="Output path for selection imbalance trend figure.",
    )
    parser.add_argument(
        "--combo-out",
        type=str,
        default=str(FIGURES_DIR / "exp9_cifar10_combo.png"),
        help="Output path for wide combined figure.",
    )
    parser.add_argument("--imbalance-metric", choices=["std", "kl"], default="std", help="Metric for selection trend.")
    parser.add_argument("--hist-generation", type=int, default=1, help="Generation to visualize histogram in combo plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = pathlib.Path(args.results_dir)
    per_gen_baseline = load_merged_results(results_dir, "baseline")
    per_gen_setaware = load_merged_results(results_dir, "set_aware")
    agg_baseline = aggregate_metrics(per_gen_baseline)
    agg_setaware = aggregate_metrics(per_gen_setaware)

    plot_curves(agg_baseline, agg_setaware, pathlib.Path(args.out))
    trend_base = compute_selection_metrics(per_gen_baseline)
    trend_set = compute_selection_metrics(per_gen_setaware)
    plot_selection_trend(args.imbalance_metric, trend_base, trend_set, pathlib.Path(args.trend_out))
    plot_combined_wide(
        agg_baseline,
        agg_setaware,
        per_gen_baseline,
        per_gen_setaware,
        pathlib.Path(args.combo_out),
        hist_generation=args.hist_generation,
    )
    print(f"Saved plots to {args.out}, {args.trend_out}, and {args.combo_out}")


if __name__ == "__main__":
    main()
