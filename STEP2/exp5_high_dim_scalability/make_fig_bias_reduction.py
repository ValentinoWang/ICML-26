import csv
import pathlib
from typing import Dict, List, Sequence, Optional
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BASE_TABLES_DIR = ROOT / "Total_results" / "Tables" / SCRIPT_DIR.name
BASE_FIGURES_DIR = ROOT / "Total_results" / "Figures" / SCRIPT_DIR.name
TABLES_DIR = BASE_TABLES_DIR / "results"
FIGURES_DIR = BASE_FIGURES_DIR / "results"

# --- 1. 全局样式升级 (更像 LaTeX/Nature 风格) ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,       # 坐标轴线稍微加粗
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.direction": "in",     # 刻度朝内，更紧凑
    "ytick.direction": "in",
    "legend.fontsize": 13,
    "legend.frameon": False,     # 去掉图例边框，更简洁
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42,          # 确保导出 PDF 字体可编辑
    "hatch.linewidth": 0.6,      # 【关键】让填充纹理变细，不显得脏
    "hatch.color": "white",      # 【关键】纹理颜色设为白色，增加质感
})

# --- 2. 统一配色方案 (美化版) ---
METHODS = ["no_filter", "mlp_filter", "mlp_correction", "ours"]
# 方案说明：
# No Filter: 灰色 (作为背景基准)
# MLP Filter: 雾霾蓝 (冷静的对比色)
# MLP Correction: 黛紫色 (另一种对比色)
# Ours: 砖红色 (最醒目，暖色，突出重点)

STYLES = {
    "no_filter": {
        "label": "No Filter", 
        "c": "#B0B0B0",          # 中性灰
        "ls": "--", "mk": "o", "z": 1, 
        "hatch": "///", "edge": "#808080" # 单独定义边框色
    },
    "mlp_filter": {
        "label": "Pointwise Filter", 
        "c": "#6A89CC",          # 雾霾蓝 (Serenity)
        "ls": ":", "mk": "^", "z": 2, 
        "hatch": "...", "edge": "#4A69BD"
    },
    "mlp_correction": { 
        "label": "MLP+Corr", 
        "c": "#82CCDD",          # 清爽的青蓝色 (或选紫色 #6D214F)
        "ls": "-.", "mk": "s", "z": 2.5, 
        "hatch": "xxx", "edge": "#60A3BC"
    },
    "ours": {
        "label": "Set-Aware (Ours)", 
        "c": "#E55039",          # 砖红色 (Alizarin) - 视觉重心
        "ls": "-", "mk": "D", "z": 10, # zorder 设高一点，确保永远在最上层
        "hatch": "", "edge": "#B71540" # 边框比填充色深一点，增加立体感
    },
}

# --- 2.5. CSV helpers ---
def read_tail_csv(path: pathlib.Path) -> Dict[int, Dict[str, float]]:
    tails: Dict[int, Dict[str, float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            dim = int(float(row.get("dim", 0)))
            tails[dim] = {}
            for m in METHODS:
                key = f"{m}_mean"
                if key in row and row[key] != "":
                    tails[dim][m] = float(row[key])
    return tails
# ... (中间的文件读取代码保持不变) ...

# --- 3. 绘图函数的微调 (Plotting Refinements) ---

def plot_adaptive_bar(tails: Dict[int, Dict[str, float]], out_path: pathlib.Path) -> None:
    dims = sorted(tails.keys())
    methods = ["no_filter", "mlp_filter", "mlp_correction", "ours"]
    
    x = np.arange(len(dims))
    width = 0.18  # 稍微调窄一点，留出更多呼吸感
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 添加淡灰色的背景网格，提升数据可读性
    ax.grid(True, which="major", axis="y", ls="-", color="#EAEAEA", alpha=1.0, zorder=0)

    for i, method in enumerate(methods):
        style = STYLES.get(method, {})
        values = [tails[d][method] for d in dims]
        
        offset = (i - 1.5) * width
        
        # 这里的关键修改：
        # 1. 使用 style.get("edge") 让边框颜色比填充色深，增加立体感
        # 2. linewidth=0.8 让柱子边缘更清晰
        ax.bar(
            x + offset, 
            values, 
            width, 
            label=style["label"], 
            color=style["c"],
            edgecolor=style.get("edge", "white"), # 如果没有定义 edge 就用白边
            linewidth=1.2, # 稍微加粗边框
            hatch=style.get("hatch", ""), 
            alpha=0.95,    # 不透明度高一点，颜色更实
            zorder=style["z"]
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("Dimension ($d$)", fontweight='bold')
    ax.set_ylabel("Tail Error (Log Scale)", fontweight='bold')
    ax.set_yscale("log")
    
    # 图例设置：放在顶部，水平排列
    ax.legend(
        loc="lower center", 
        bbox_to_anchor=(0.5, 1.02), 
        ncol=4, 
        frameon=False, 
        fontsize=13,
        handletextpad=0.4, # 图标和文字靠紧一点
        columnspacing=1.5  # 列间距
    )
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_tail_vs_dim_line(tails: Dict[int, Dict[str, float]], out_path: pathlib.Path) -> None:
    """折线图美化"""
    dims = sorted(tails.keys())
    methods = ["no_filter", "mlp_filter", "mlp_correction", "ours"]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.grid(True, which="major", ls="-", color="#EAEAEA", zorder=0)

    for method in methods:
        style = STYLES.get(method, {})
        values = [tails[d][method] for d in dims]
        
        # 线条美化：加阴影或加粗
        ax.plot(
            dims, 
            values, 
            label=style["label"], 
            color=style["c"], 
            marker=style["mk"], 
            linestyle=style["ls"],
            linewidth=2.5,        # 线条加粗
            markersize=9,         # 标记点加大
            markeredgecolor='white', # 标记点加白边，显得精致
            markeredgewidth=1.5,
            zorder=style["z"]
        )

    ax.set_xlabel("Dimension ($d$)", fontweight='bold')
    ax.set_ylabel("Tail Error", fontweight='bold')
    ax.set_yscale("log")
    ax.set_xticks(dims)
    
    # 移除顶部和右侧脊柱
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor='#EAEAEA', framealpha=0.9)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_adaptive(
    tails: Dict[int, Dict[str, float]],
    out_path: pathlib.Path,
) -> None:
    plot_adaptive_bar(tails, out_path)


def main() -> None:
    csv_path = TABLES_DIR / "exp5_tail_summary.csv"
    tails = read_tail_csv(csv_path)
    plot_tail_vs_dim_line(tails, FIGURES_DIR / "exp5_tail_vs_dim_line.png")
    plot_adaptive(tails, FIGURES_DIR / "exp5_bias_reduction_rate.png")
    plot_adaptive_bar(tails, FIGURES_DIR / "exp5_high_dim_bar.png")


if __name__ == "__main__":
    main()


def plot_tail_vs_dim(
    dims: Sequence[int],
    tail_stats: Dict[int, Dict[str, Dict[str, float]]],
    out_path: pathlib.Path,
    n_seeds: int,
    methods: Sequence[str],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in methods:
        style = STYLES.get(m, {})
        means = np.array([tail_stats[d][m]["mean"] for d in dims])
        stds = np.array([tail_stats[d][m]["std"] for d in dims])
        ci = 1.96 * stds / np.sqrt(max(n_seeds, 1))
        ax.plot(
            dims,
            means,
            label=style.get("label", m),
            color=style.get("c", None),
            linestyle=style.get("ls", "-"),
            marker=style.get("mk", "o"),
            linewidth=2,
            markersize=7,
            zorder=style.get("z", 1),
        )
        ax.fill_between(dims, means - ci, means + ci, color=style.get("c", None), alpha=0.12, zorder=0)
    ax.set_xlabel("Dimension ($d$)", fontweight="bold")
    ax.set_ylabel("Tail Error", fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(dims)
    ax.grid(True, which="major", ls="--", alpha=0.3)
    ax.legend(loc="best", frameon=True, edgecolor="k", framealpha=0.85)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(
    all_series: Dict[int, Dict[str, Dict[str, List[float]]]],
    out_path: pathlib.Path,
    n_seeds: int,
    methods: Sequence[str],
) -> None:
    dims = sorted(all_series.keys())
    if not dims:
        return
    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4), sharey=False)
    if len(dims) == 1:
        axes = [axes]
    for ax, d in zip(axes, dims):
        series = all_series[d]
        gens = np.arange(1, len(next(iter(series.values()))["mean"]) + 1)
        for m in methods:
            style = STYLES.get(m, {})
            mean = np.array(series[m]["mean"])
            std = np.array(series[m]["std"])
            ci = 1.96 * std / np.sqrt(max(n_seeds, 1))
            ax.plot(
                gens,
                mean,
                label=style.get("label", m),
                color=style.get("c", None),
                linestyle=style.get("ls", "-"),
                marker=style.get("mk", "o"),
                linewidth=2,
                markersize=6,
            )
            ax.fill_between(gens, mean - ci, mean + ci, color=style.get("c", None), alpha=0.12)
        ax.set_title(f"d={d}")
        ax.set_xlabel("Generation")
        ax.set_ylabel(r"$\|\theta_t-\theta^*\|_2$")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
