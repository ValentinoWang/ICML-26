import pathlib
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# --- 1. 统一视觉配置 (Global Design System) ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.constrained_layout.use": True, # 自动防止重叠
    "pdf.fonttype": 42,
})

# --- 2. 统一配色字典 ---
STYLES = {
    "no_filter": {
        "label": "No Filter", 
        "color": "#7f8c8d",  # 灰色 (Baseline)
        "linestyle": "--",   # 虚线
        "zorder": 1          # 放在底层
    },
    "dst": {
        "label": "DST (Decoupled)", 
        "color": "#8c564b",
        "linestyle": "-.",
        "zorder": 2
    },
    "batch_stats": {
        "label": "Pointwise + Batch Stats",
        "color": "#ff7f0e",
        "linestyle": "--",
        "zorder": 2
    },
    "l2ac": {
        "label": "L2AC (Meta-weight)", 
        "color": "#17becf",
        "linestyle": ":",
        "zorder": 2
    },
    "ours": {
        "label": "Set-Aware (Ours)", 
        "color": "#2ca02c",  # 绿色 (Stable/Correct) - 与 Exp7 t-SNE 保持一致
        "linestyle": "-",    # 实线
        "zorder": 3          # 放在顶层
    }
}

def plot_series(stats: Dict[str, Dict[str, List[float]]], out_dir: pathlib.Path, n_seeds: int) -> None:
    # 准备数据
    # 注意：这里假设所有曲线长度一致
    first_metric = next(iter(stats)) # e.g. "mse"
    first_method = next(iter(stats[first_metric])) # e.g. "no_filter"
    n_gens = len(stats[first_metric][first_method]["mean"])
    gens = np.arange(1, n_gens + 1)

    # 创建画布：宽长比适中
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # 定义要画的指标配置
    metrics_cfg = [
        {
            "key": "mse", 
            "title": "(a) Test MSE (Real Data)", 
            "ylabel": "Mean Squared Error", 
            "methods": ["no_filter", "batch_stats", "dst", "l2ac", "ours"],
            "arrow": True # 是否画 "Lower is Better" 箭头
        },
        {
            "key": "norm", 
            "title": "(b) Parameter Norm Dynamics", 
            "ylabel": r"Parameter Norm $\|\theta_t\|_2$", 
            "methods": ["no_filter", "batch_stats", "dst", "l2ac", "ours"],
            "arrow": False
        }
    ]

    legend_handles = []
    legend_labels = []

    for ax, cfg in zip(axes, metrics_cfg):
        metric_key = cfg["key"]
        
        for method_key in cfg["methods"]:
            # 获取数据
            if method_key not in stats[metric_key]:
                continue
                
            mean = np.array(stats[metric_key][method_key]["mean"])
            std = np.array(stats[metric_key][method_key]["std"])
            ci = 1.96 * std / np.sqrt(n_seeds)
            
            # 获取样式
            style = STYLES.get(method_key, {"color": "black", "linestyle": "-"})
            
            # 绘图
            line, = ax.plot(
                gens, mean, 
                label=style["label"], 
                color=style["color"], 
                linestyle=style["linestyle"], 
                linewidth=2.0,
                zorder=style.get("zorder", 1)
            )
            
            # 误差带
            ax.fill_between(
                gens, mean - ci, mean + ci, 
                color=style["color"], 
                alpha=0.15, 
                linewidth=0,
                zorder=style.get("zorder", 1) - 0.5
            )

            # 收集图例 (只收集一次)
            if ax == axes[0]:
                legend_handles.append(line)
                legend_labels.append(style["label"])

        # 装饰子图
        ax.set_xlabel("Generation", fontweight="bold")
        ax.set_ylabel(cfg["ylabel"], fontweight="bold")
        ax.set_title(cfg["title"], loc="left", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlim(left=0, right=n_gens)

        # 特殊装饰：Better Arrow
        if cfg.get("arrow"):
            # 在图的右上角画一个向下的箭头，表示 MSE 越低越好
            ax.annotate("Lower is better", 
                        xy=(0.85, 0.15), xycoords='axes fraction',
                        xytext=(0.85, 0.35), textcoords='axes fraction',
                        arrowprops=dict(facecolor='black', arrowstyle="->", alpha=0.5),
                        ha='center', fontsize=9, color="#555", style='italic')

    # --- 全局图例 (Shared Legend) ---
    # 放在顶部，水平排列
    fig.legend(
        legend_handles, 
        legend_labels, 
        loc="upper center", 
        bbox_to_anchor=(0.5, 1.08), 
        ncol=len(legend_handles), 
        frameon=False, 
        fontsize=11
    )

    # 保存
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp7_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved styled plot to {out_path}")
    plt.close()


def _estimate_contraction_rate(series: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    prev = np.maximum(series[:-1], eps)
    return 1.0 - series[1:] / prev


def plot_contraction_diagnostics(
    csv_path: pathlib.Path,
    out_path: pathlib.Path,
    methods: List[str] | None = None,
) -> None:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    gens = data["generation"].astype(int)
    if methods is None:
        methods = ["no_filter", "ours"]

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.4))
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.5, zorder=0)

    for method_key in methods:
        mse_key = f"{method_key}_mse_mean"
        if mse_key not in data.dtype.names:
            continue
        mse = np.array(data[mse_key], dtype=float)
        c_hat = _estimate_contraction_rate(mse)
        style = STYLES.get(method_key, {"color": "black", "linestyle": "-"})
        ax.plot(
            gens[1:],
            c_hat,
            label=style["label"],
            color=style["color"],
            linestyle=style.get("linestyle", "-"),
            linewidth=2.0,
            zorder=style.get("zorder", 1),
        )
        ax.axhline(
            float(np.mean(c_hat)),
            color=style["color"],
            linestyle=":",
            linewidth=1.2,
            alpha=0.7,
        )

    ax.set_xlabel("Generation", fontweight="bold")
    ax.set_ylabel(r"Estimated contraction $\hat{c}_t$", fontweight="bold")
    ax.set_title("Contraction-Rate Diagnostic", loc="left", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(left=1, right=gens[-1])
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved contraction diagnostic to {out_path}")
    plt.close()
