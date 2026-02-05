import argparse
import csv
import json
import pathlib
import statistics
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
BASE_TABLES_DIR = ROOT_DIR / "Total_results" / "Tables" / SCRIPT_DIR.name
RESULTS_DIR = BASE_TABLES_DIR / "Results"
BASE_FIGURES_DIR = ROOT_DIR / "Total_results" / "Figures" / SCRIPT_DIR.name
FIGURES_DIR = BASE_FIGURES_DIR / "results"
DEFAULT_SUMMARY_OUT = BASE_TABLES_DIR / "exp11_gpt2_g4_summary.csv"
DEFAULT_SUMMARY_ROOTS = "dphi1,dphi1_leash,dispersion,ppl_safety,rep_filter_thr0.6,pointwise_mix0.2,pointwise_mix0.5"

# --- 样式设置 (针对 ICML 2025 优化) ---
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",  # 公式字体更像 LaTeX
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,  # 线条稍微细一点，更精致
    "axes.labelsize": 16,  # 字号加大，适应双栏排版
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 13,
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42,  # 确保导出文本可编辑（非轮廓）
    "ps.fonttype": 42,
})

# --- 颜色与样式 ---
# 稍微调整颜色，使其更深沉一点，符合学术风格
STYLES = {
    "no_filter": {"label": "No Filter", "c": "#7f8c8d", "ls": "--", "mk": "o", "z": 1},  # 灰色
    "pointwise": {"label": "Pointwise", "c": "#d62728", "ls": ":", "mk": "^", "z": 2},  # 深红
    "set_aware": {"label": "Set-Aware (Ours)", "c": "#1f77b4", "ls": "-", "mk": "D", "z": 3},  # 经典蓝
}


def parse_summary_roots(raw: str) -> list[pathlib.Path]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    roots: list[pathlib.Path] = []
    for part in parts:
        p = pathlib.Path(part)
        if p.is_absolute():
            roots.append(p)
        elif "exp11_gpt2_model" in part:
            roots.append(ROOT_DIR / part)
        else:
            roots.append(RESULTS_DIR / part)
    return roots


def summarize(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def classify_method(results_root: pathlib.Path, method: str) -> tuple[str, str, str]:
    name = results_root.name
    variant = ""
    if name.startswith("pointwise_mix"):
        category = "reference"
        method_label = "mix_original"
        variant = name.replace("pointwise_mix", "")
    elif name == "rep_filter_thr0.6":
        category = "domain_specific"
        method_label = "rep_filter"
        variant = "thr0.6"
    elif name == "ppl_safety":
        category = "pointwise"
        method_label = "ppl_safety"
    elif name == "dispersion":
        category = "geometric"
        method_label = "dispersion"
    elif name == "dphi1":
        category = "pointwise" if method in {"no_filter", "pointwise"} else "geometric"
        method_label = method
        if method == "pointwise":
            method_label = "pointwise_raw"
    elif name == "dphi1_leash":
        category = "geometric"
        method_label = "set_aware_leash"
    else:
        category = "other"
        method_label = method
    return category, method_label, variant


def collect_summary_rows(results_roots: list[pathlib.Path], gen: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in results_roots:
        if not root.exists():
            print(f"Warning: missing results root {root}")
            continue
        seed_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
        per_method: dict[str, dict[int, tuple[float, float]]] = {}
        for sd in seed_dirs:
            metrics_path = sd / "metrics_diversity_ppl.csv"
            if not metrics_path.exists():
                continue
            df = pd.read_csv(metrics_path)
            df = df[df["generation"] == gen]
            for _, row in df.iterrows():
                method = str(row.get("method", "")).strip()
                if not method:
                    continue
                per_method.setdefault(method, {})[int(sd.name)] = (float(row["val_ppl"]), float(row["distinct4"]))
        for method, seed_map in per_method.items():
            seeds = sorted(seed_map.keys())
            val_ppl = [seed_map[s][0] for s in seeds]
            d4 = [seed_map[s][1] for s in seeds]
            ppl_mean, ppl_std = summarize(val_ppl)
            d4_mean, d4_std = summarize(d4)
            category, method_label, variant = classify_method(root, method)
            rows.append(
                {
                    "results_root": str(root),
                    "category": category,
                    "method": method,
                    "method_label": method_label,
                    "variant": variant,
                    "gen": gen,
                    "n_seeds": len(seeds),
                    "seeds": ",".join(str(s) for s in seeds),
                    "val_ppl_mean": ppl_mean,
                    "val_ppl_std": ppl_std,
                    "distinct4_mean": d4_mean,
                    "distinct4_std": d4_std,
                }
            )
    return rows


def write_summary_csv(rows: list[dict[str, object]], out_path: pathlib.Path) -> None:
    if not rows:
        print("Warning: no summary rows to write.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote summary CSV to {out_path}")


def load_results(results_root: pathlib.Path) -> pd.DataFrame:
    """Load metrics from results_root; fallback to dummy data if none found."""
    records = []
    # 尝试读取真实数据
    for path in results_root.glob("*/*metrics_diversity_ppl.json"):
        seed = path.parent.name
        try:
            with path.open("r") as f:
                data = json.load(f)
            history = data.get("history", {})
            for method, items in history.items():
                for rec in items:
                    records.append({
                        "seed": seed,
                        "method": method,
                        "generation": rec.get("generation", 0),
                        "distinct4": rec.get("distinct4", np.nan),
                        "val_ppl": rec.get("val_ppl", np.nan),
                    })
        except Exception:
            pass

    # 如果没有数据，生成 Dummy Data (方便调试绘图风格)
    if not records:
        print(f"Warning: No data found in {results_root}, utilizing dummy data for visualization.")
        gens = np.arange(1, 11)
        return pd.DataFrame({
            "method": ["no_filter"] * 10 + ["pointwise"] * 10 + ["set_aware"] * 10,
            "generation": np.tile(gens, 3),
            "distinct4": np.concatenate([
                np.linspace(0.32, 0.34, 10),
                np.linspace(0.40, 0.48, 10),
                np.linspace(0.38, 0.42, 10)
            ]),
            "val_ppl": np.concatenate([
                np.linspace(4200, 4500, 10),
                np.geomspace(4000, 18000, 10),
                np.linspace(5000, 6500, 10)
            ]),
        }).assign(gen_plot=lambda x: x["generation"])

    df = pd.DataFrame.from_records(records)
    df["gen_plot"] = df["generation"] + 1
    return df


def aggregate_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean across seeds."""
    return df.groupby(["method", "generation"]).agg(
        distinct4_mean=("distinct4", "mean"),
        val_ppl_mean=("val_ppl", "mean"),
    ).reset_index().assign(gen_plot=lambda x: x["generation"] + 1)


def plot_phase_portrait(df_mean: pd.DataFrame, out_path: pathlib.Path) -> None:
    """
    绘制相图：Diversity vs Quality。
    风格：ICML 极简风，带轨迹箭头和 'Better' 指示。
    """
    # 调整画布大小：更紧凑，适合单栏或半页展示
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    all_ppl = []

    # 绘制曲线
    for method in ["no_filter", "pointwise", "set_aware"]:
        if method not in STYLES:
            continue
        style = STYLES[method]
        subset = df_mean[df_mean["method"] == method].sort_values("gen_plot")
        if subset.empty:
            continue

        all_ppl.extend(subset["val_ppl_mean"].tolist())

        # 主曲线
        ax.plot(
            subset["distinct4_mean"],
            subset["val_ppl_mean"],
            label=style["label"],
            color=style["c"],
            linestyle=style["ls"],
            linewidth=2.0,  # 稍微细一点
            marker=style["mk"],
            markersize=6,
            markevery=(0, len(subset) - 1),  # 只标记起点和终点，减少杂乱
            alpha=0.9,
            zorder=style["z"],
        )

        # 绘制箭头 (显示演化方向)
        if len(subset) > 2:
            mid = len(subset) // 2
            p1, p2 = subset.iloc[mid], subset.iloc[mid + 1]
            ax.annotate(
                "",
                xy=(p2["distinct4_mean"], p2["val_ppl_mean"]),
                xytext=(p1["distinct4_mean"], p1["val_ppl_mean"]),
                arrowprops=dict(arrowstyle="->", color=style["c"], lw=1.5),
                zorder=style["z"],
            )

        # 标记 G10 (终点)
        end = subset.iloc[-1]
        # 智能调整文字位置避免遮挡
        offset = (5, 5)
        if method == "no_filter":
            offset = (-25, -15)
        if method == "pointwise":
            offset = (5, 5)

        ax.annotate(
            f"G{int(end['gen_plot'])}",
            (end["distinct4_mean"], end["val_ppl_mean"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=10,
            color=style["c"],
            fontweight="bold",
        )

    # --- 坐标轴设置 ---
    ax.set_yscale("log")

    # 自动计算更紧凑的 Y 轴范围
    if all_ppl:
        min_p, max_p = min(all_ppl), max(all_ppl)
        ax.set_ylim(min_p * 0.8, max_p * 1.5)

    # --- 极简风格的 "Better" 指示 ---
    # 在右下角画一个小箭头，指示 PPL 越低越好，Diversity 越高越好
    ax.annotate(
        "Better Trade-off",
        xy=(0.97, 0.22),
        xycoords="axes fraction",
        xytext=(0.78, 0.48),
        textcoords="axes fraction",
        arrowprops=dict(color="#1f4e79", arrowstyle="->", lw=2.0, connectionstyle="arc3,rad=0.2"),
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=12,
        color="#1f4e79",
        fontweight="bold",
    )

    ax.set_xlabel("Diversity (Dist-4)", fontweight="bold", fontsize=12)
    ax.set_ylabel("Val PPL (Log)", fontweight="bold", fontsize=12)

    # 网格线淡化
    ax.grid(True, which="major", ls="--", alpha=0.3, color="gray")

    # Legend 放在图内最佳位置
    legend = ax.legend(loc="lower left", frameon=True, fancybox=False, edgecolor="k", framealpha=0.9)
    # Match legend text color to line color for fast visual parsing.
    for text, handle in zip(legend.get_texts(), legend.legend_handles):
        try:
            text.set_color(handle.get_color())
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dual_axis(df_mean: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Dual-axis evolution plot (Distinct-4 and PPL vs generation)."""
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    for method in ["no_filter", "pointwise", "set_aware"]:
        if method not in STYLES:
            continue
        style = STYLES[method]
        subset = df_mean[df_mean["method"] == method].sort_values("gen_plot")
        if subset.empty:
            continue

        # PPL (虚线) on Right Axis
        ax2.plot(
            subset["gen_plot"],
            subset["val_ppl_mean"],
            color=style["c"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"{style['label']} (PPL)",
        )
        # Distinct (实线) on Left Axis
        ax1.plot(
            subset["gen_plot"],
            subset["distinct4_mean"],
            color=style["c"],
            marker=style["mk"],
            linestyle="-",
            linewidth=2,
            markersize=5,
            alpha=0.9,
            label=f"{style['label']} (Div)",
        )

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Diversity (Distinct-4)", fontweight="bold")
    ax2.set_ylabel("Validation PPL (log)", fontweight="bold", color="#555")
    ax2.set_yscale("log")
    ax1.grid(True, axis="x", alpha=0.3)

    # Unified Legend (Manual placement outside or top)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # 放在顶部外侧，避免遮挡数据
    fig.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_recursive_lines(df_mean: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Separate line plots for PPL and Distinct-4."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for method in ["no_filter", "pointwise", "set_aware"]:
        if method not in STYLES:
            continue
        style = STYLES[method]
        subset = df_mean[df_mean["method"] == method].sort_values("gen_plot")
        if subset.empty:
            continue

        # Plot PPL
        axes[0].plot(subset["gen_plot"], subset["val_ppl_mean"], label=style["label"], color=style["c"], ls=style["ls"], marker=style["mk"], markevery=2)
        # Plot Diversity
        axes[1].plot(subset["gen_plot"], subset["distinct4_mean"], label=style["label"], color=style["c"], ls=style["ls"], marker=style["mk"], markevery=2)

    axes[0].set_yscale("log", base=10)
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Validation PPL (log scale)")
    axes[0].set_title("Quality (PPL) vs Generation")
    axes[0].grid(alpha=0.3, which="both")

    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Distinct-4")
    axes[1].set_title("Diversity vs Generation")
    axes[1].grid(alpha=0.3)

    # Shared Legend
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=pathlib.Path, default=RESULTS_DIR / "dphi1")
    parser.add_argument(
        "--out-phase",
        type=pathlib.Path,
        default=FIGURES_DIR / "exp11_phase_portrait.png",
    )
    # Optional: Add paths for other plots if needed
    parser.add_argument("--out-dual", type=pathlib.Path, default=FIGURES_DIR / "exp11_dual_metrics.png")
    parser.add_argument("--summary-roots", type=str, default=DEFAULT_SUMMARY_ROOTS, help="Comma-separated results root names or paths.")
    parser.add_argument("--summary-gen", type=int, default=4, help="Generation to summarize for CSV output.")
    parser.add_argument("--summary-out", type=pathlib.Path, default=DEFAULT_SUMMARY_OUT, help="CSV output path for summary.")
    parser.add_argument("--write-summary", action="store_true", help="Write summary CSV from existing results.")
    parser.add_argument("--summary-only", action="store_true", help="Only write summary CSV and skip plots.")

    args = parser.parse_args()

    if args.summary_only:
        args.write_summary = True

    if args.write_summary:
        summary_roots = parse_summary_roots(args.summary_roots)
        rows = collect_summary_rows(summary_roots, gen=int(args.summary_gen))
        write_summary_csv(rows, args.summary_out)

    if args.summary_only:
        return

    df = load_results(args.results_root)
    df_mean = aggregate_mean(df)

    print(f"Generating Phase Portrait at {args.out_phase}...")
    plot_phase_portrait(df_mean, args.out_phase)

    # 顺便生成另外两张图（可选，用于检查）
    # plot_dual_axis(df_mean, args.out_dual)
    # plot_recursive_lines(df_mean, args.out_phase.parent / "exp11_lines.png")

    print("Done.")


if __name__ == "__main__":
    main()
