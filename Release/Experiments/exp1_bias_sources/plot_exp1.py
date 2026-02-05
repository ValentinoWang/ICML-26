import argparse
import csv
import pathlib
from typing import Dict, List, Tuple, Optional

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BASE_TABLES_DIR = ROOT / "Total_results" / "Tables" / SCRIPT_DIR.name
BASE_FIGURES_DIR = ROOT / "Total_results" / "Figures" / SCRIPT_DIR.name
TABLES_DIR = BASE_TABLES_DIR / "results"
FIGURES_DIR = BASE_FIGURES_DIR / "results"

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # 引入 seaborn 用于获取更好的调色板

# --- 1. A+ 级样式设置 (ICML Style) ---
# 注意：使用 text.usetex = True 需要你的系统安装了 LaTeX (如 TeXLive, MiKTeX)
# 如果运行报错，请将 text.usetex 改为 False
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{mathptmx}",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.6,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# --- 2. 优化后的配色与线型（按用户指定） ---
COLORS = {
    "no_filter": "#949494",        # gray
    "standard_filter": "#DE8F05",  # orange
    "hard_threshold": "#56B4E9",   # sky blue
    "k_center": "#029E73",         # green
    "dpp": "#CC78BC",              # purple
    "ema": "#CA9161",              # brown
    "dst": "#D55E00",              # vermillion
    "l2ac": "#ECE133",             # yellow
    "ours": "#0173B2",             # blue
}

# 自定义线型 Tuple：(offset, (on_len, off_len, ...))
LINE_STYLES = {
    "no_filter": (0, (4, 2)),          # medium dash
    "standard_filter": (0, (1, 2)),    # dotted
    "hard_threshold": (0, (6, 2, 1, 2)),
    "k_center": (0, (3, 2)),
    "dpp": (0, (2, 2, 1, 2)),
    "ema": (0, (5, 1)),
    "dst": (0, (4, 1, 1, 1)),
    "l2ac": (0, (2, 1)),
    "ours": "-",
}

LABEL_MAP = {
    "no_filter": "No Filter",
    "standard_filter": "Pointwise MLP",
    "hard_threshold": "Hard-Threshold",
    "k_center": "k-Center (Coreset)",
    "dpp": "DPP (MAP)",
    "ema": "EMA",
    "dst": "DST (Decoupled)",
    "l2ac": "L2AC (Meta-weight)",
    "ours": "Ours (Correction)",
}
METHOD_KEYS = sorted(LABEL_MAP.keys(), key=len, reverse=True)

# ... (load_series_from_csv 函数保持不变) ...
def load_series_from_csv(path: pathlib.Path) -> Tuple[np.ndarray, List[str], Dict[str, Dict]]:
    # ... (此处代码与你原版一致，省略以节省空间) ...
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]

    if not rows:
        raise ValueError("CSV file is empty.")

    gens = np.array([float(r[0]) for r in rows])

    cfg_order: List[str] = []
    data: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    
    for col in header[1:]:
        parts = col.split("_")
        if len(parts) < 2:
            continue
        stat = parts[-1]
        method = None
        for key in METHOD_KEYS:
            prefix = key + "_"
            if col.startswith(prefix):
                method = key
                break
        if method is None:
            continue
        cfg = col[len(method) + 1 : -len(stat) - 1] 

        if cfg not in data:
            data[cfg] = {}
            cfg_order.append(cfg)
        if method not in data[cfg]:
            data[cfg][method] = {"mean": [], "std": []}

    for row in rows:
        for col, val in zip(header[1:], row[1:]):
            parts = col.split("_")
            if len(parts) < 2:
                continue
            stat = parts[-1]
            method = None
            for key in METHOD_KEYS:
                prefix = key + "_"
                if col.startswith(prefix):
                    method = key
                    break
            if method is None:
                continue
            cfg = col[len(method) + 1 : -len(stat) - 1]
            if cfg in data and method in data[cfg]:
                data[cfg][method][stat].append(float(val))

    series: Dict[str, Dict] = {}
    for cfg in cfg_order:
        series[cfg] = {"series": {}}
        for method, stats in data[cfg].items():
            series[cfg]["series"][method] = {
                "mean": np.array(stats["mean"], dtype=float),
                "std": np.array(stats["std"], dtype=float),
            }
    return gens, cfg_order, series

def plot_from_series(gens: np.ndarray, 
                     cfg_order: List[str], 
                     series: Dict[str, Dict], 
                     out_path: pathlib.Path, 
                     n_seeds: int = 5) -> None:
    """
    主绘图函数：优化了布局和图例位置逻辑
    """
    def _format_title(cfg_name: str) -> str:
        if cfg_name.startswith("b"):
            return rf"Hard Bias $|b|={cfg_name[1:]}$" # 标题更详细一点
        if cfg_name.startswith("alpha"):
            return rf"Ridge $\alpha={cfg_name[len('alpha'):] }$"
        if cfg_name.startswith("sig"):
            sigma_part, *rest = cfg_name.split("_n")
            sigma_val = sigma_part.replace("sig", "")
            n_val = rest[0] if rest else ""
            return rf"$\sigma={sigma_val},\ n={n_val}$"
        return cfg_name

    cols = len(cfg_order)
    # 调整画布比例
    fig, axes = plt.subplots(1, cols, figsize=(5.5 * cols, 4.5))
    
    if cols == 1:
        axes = [axes]

    # 绘制顺序：确保 Ours 永远最后绘制 (Top Z-order)
    method_priority = [
        "no_filter",
        "hard_threshold",
        "ema",
        "standard_filter",
        "k_center",
        "dpp",
        "dst",
        "l2ac",
        "ours",
    ]

    for ax, cfg_name in zip(axes, cfg_order):
        cfg_series = series[cfg_name]["series"]
        
        # 将带后缀的方法名映射到基名
        ordered_pairs = []
        for base in method_priority:
            for full_name in cfg_series.keys():
                if full_name.startswith(base):
                    ordered_pairs.append((base, full_name))

        for base_name, full_name in ordered_pairs:
            vals = cfg_series[full_name]
            mean = np.array(vals["mean"], dtype=float)
            std = np.array(vals["std"], dtype=float)

            if n_seeds > 1:
                ci = 1.96 * std / np.sqrt(n_seeds)
            else:
                ci = std

            label = LABEL_MAP.get(base_name, base_name)
            color = COLORS.get(base_name, "#333333")
            ls = LINE_STYLES.get(base_name, "-")

            # Ours 层级最高，线宽更粗、基线更轻
            is_ours = (base_name == "ours")
            z_ord = 10 if is_ours else 2
            lw_adj = 2.4 if is_ours else 1.6
            line_alpha = 1.0 if is_ours else 0.8
            fill_alpha = 0.18 if is_ours else 0.08
            
            line, = ax.plot(
                gens,
                mean,
                label=label,
                color=color,
                linestyle=ls,
                linewidth=lw_adj,
                alpha=line_alpha,
                zorder=z_ord,
            )
            
            # 阴影透明度
            ax.fill_between(
                gens,
                mean - ci,
                mean + ci,
                color=color,
                alpha=fill_alpha,
                linewidth=0,
                zorder=z_ord - 1,
            )

        ax.set_xlabel("Generation")
        if ax == axes[0]:
            ax.set_ylabel(r"Error $\|\theta_t - \theta^*\|_2$")
        
        # 移除默认 Title，建议作为 Figure Caption 或者放在图内
        # 如果非要标题，可以用 text 注解的方式放在图内上方，显得更紧凑
        # ax.set_title(_format_title(cfg_name)) 
        
        # 设置 Y 轴略微留白
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=gens.max())

        # 内嵌图例与标题注解
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                loc="upper right",
                frameon=True,
                facecolor="white",
                framealpha=0.85,
                edgecolor="none",
            )
        ax.text(
            0.02,
            0.95,
            _format_title(cfg_name),
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=13,
            fontweight="bold",
        )
        # 标注 Bias Floor（标准滤波器的平稳段）
        if "standard_filter" in cfg_series:
            std_mean = cfg_series["standard_filter"]["mean"]
            if len(std_mean) > 0:
                idx = max(0, int(len(std_mean) * 0.8) - 1)
                x_target = gens[idx]
                y_target = std_mean[idx]
                ax.annotate(
                    "Bias Floor",
                    xy=(x_target, y_target),
                    xytext=(x_target * 0.6, y_target * 1.25),
                    arrowprops=dict(arrowstyle="->", color=COLORS["standard_filter"], lw=1.5),
                    color=COLORS["standard_filter"],
                    fontsize=11,
                    ha="right",
                    va="bottom",
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 同时保存 PDF (矢量, 用于投稿) 和 PNG (预览)
    plt.savefig(out_path.with_suffix('.pdf'), bbox_inches="tight")
    plt.savefig(out_path.with_suffix('.png'), dpi=300, bbox_inches="tight")
    
    print(f"Saved refined plot to {out_path.with_suffix('.pdf')} (and .png)")
    plt.close(fig)


def plot_scenarios(
    summaries: Dict[str, Dict[str, Dict]],
    fig_dir: pathlib.Path,
    subtitles: Dict[str, str],
    n_seeds: int = 1,
) -> None:
    """
    Render per-scenario plots directly from in-memory summaries produced by run_exp1.
    """
    name_map = {"const": "exp1_1.1_const", "ridge": "exp1_1.2_ridge", "bayes": "exp1_1.3_bayes"}
    fig_dir.mkdir(parents=True, exist_ok=True)
    for scenario, cfgs in summaries.items():
        if not cfgs:
            continue
        first_cfg = next(iter(cfgs.values()))
        first_method = next(iter(first_cfg["series"]))
        gens = np.arange(1, len(first_cfg["series"][first_method]["mean"]) + 1)
        cfg_order = list(cfgs.keys())
        series = {cfg_name: {"series": cfgs[cfg_name]["series"]} for cfg_name in cfg_order}
        out_path = fig_dir / f"{name_map.get(scenario, scenario)}.png"
        plot_from_series(gens, cfg_order, series, out_path, n_seeds=n_seeds)

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Exp1 results (Refined ICML style).")
    parser.add_argument("--csv", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--n-seeds", type=int, default=1)
    args = parser.parse_args()

    if args.out is not None:
        out_path = args.out
    else:
        try:
            rel = args.csv.resolve().relative_to(TABLES_DIR.resolve())
            out_path = (FIGURES_DIR / rel).with_suffix(".png")
        except Exception:
            out_path = args.csv.with_suffix(".png")
    
    try:
        gens, cfg_order, series = load_series_from_csv(args.csv)
        plot_from_series(gens, cfg_order, series, out_path, n_seeds=args.n_seeds)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
