#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
绘制 MT 工业检测场景下的 mAP50 vs epoch 曲线。

对比：
- YOLO baseline（Pretrain-Finetune 版本）；
- YOLO + L_bias（ICML.yolo_bias_finetune 实验）。

说明：
- 当前仓库尚未包含 YOLO + L_bias + g_phi 的 results.csv，
  因此图中暂不展示该方法，后续可在同一脚本中补充。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def get_paths(seed: int = 2195) -> Dict[str, Dict[str, Path]]:
    """构造每个场景、每个方法对应的 results.csv 路径。"""
    script_dir = Path(__file__).resolve().parent
    # project_root = .../Style_Filter
    project_root = script_dir.parents[2]
    baseline_root = project_root / "Baseline"

    pf_results_root = baseline_root / "Pretrain-Finetune" / "Results"
    # 仅 L_bias 版本（不含 g_phi）的结果目录现命名为 Bias_only，这里默认使用 lambda_bias=1e-4 的结果
    bias_results_root = baseline_root / "ICML" / "Results" / "Bias_only" / "lambda_0.0001"

    scenarios = ["few-shot", "small", "medium", "high"]

    paths: Dict[str, Dict[str, Path]] = {}
    for scenario in scenarios:
        paths[scenario] = {
            "YOLO baseline": pf_results_root
            / scenario
            / f"seed_{seed}"
            / "results"
            / "results.csv",
            "YOLO + L_bias": bias_results_root
            / scenario
            / f"seed_{seed}"
            / "results"
            / "results.csv",
        }
    return paths


def load_map_curve(csv_path: Path) -> Tuple[List[int], List[float]]:
    """从 Ultralytics YOLO 的 results.csv 中抽取 (epoch, mAP50) 序列。"""
    df = pd.read_csv(csv_path)
    if "metrics/mAP50(B)" not in df.columns:
        raise KeyError(f"metrics/mAP50(B) 不在列中: {csv_path}")

    epochs = df["epoch"].tolist()
    map50 = df["metrics/mAP50(B)"].tolist()
    return epochs, map50


def plot_all_scenarios(seed: int = 2195) -> None:
    """绘制 four-in-one 图：few-shot/small/medium/high 四个场景。"""
    paths = get_paths(seed=seed)
    figs_dir = Path(__file__).resolve().parent / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    scenario_order = ["few-shot", "small", "medium", "high"]
    scenario_titles = {
        "few-shot": "MT-tgt few-shot (1%)",
        "small": "MT-tgt small (5%)",
        "medium": "MT-tgt medium (20%)",
        "high": "MT-tgt high (100%)",
    }

    for ax, scenario in zip(axes.ravel(), scenario_order):
        method_paths = paths[scenario]
        for method_name, csv_path in method_paths.items():
            if not csv_path.exists():
                print(f"[WARN] 找不到文件: {csv_path}")
                continue
            epochs, map50 = load_map_curve(csv_path)
            # 使用更粗的线条和标记点，使曲线在图中更明显
            ax.plot(
                epochs,
                map50,
                label=method_name,
                linewidth=2.0,
                marker="o",
                markersize=2,
            )

        ax.set_title(scenario_titles.get(scenario, scenario), fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP50 (val)")
        ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels))

    fig.suptitle(
        f"MT 工业检测场景：YOLO baseline vs YOLO + L_bias (seed={seed})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    out_path = figs_dir / f"exp3_mt_yolo_map50_vs_epoch_seed{seed}_4scenarios.png"
    fig.savefig(out_path, dpi=200)
    print(f"[INFO] 已保存图像: {out_path}")


def plot_fewshot_small(seed: int = 2195) -> None:
    """单独绘制 few-shot / small 两个关键场景。"""
    paths = get_paths(seed=seed)
    figs_dir = Path(__file__).resolve().parent / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    scenarios = ["few-shot", "small"]
    scenario_titles = {
        "few-shot": "MT-tgt few-shot (1%)",
        "small": "MT-tgt small (5%)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, scenario in zip(axes, scenarios):
        method_paths = paths[scenario]
        for method_name, csv_path in method_paths.items():
            if not csv_path.exists():
                print(f"[WARN] 找不到文件: {csv_path}")
                continue
            epochs, map50 = load_map_curve(csv_path)
            ax.plot(
                epochs,
                map50,
                label=method_name,
                linewidth=2.0,
                marker="o",
                markersize=2,
            )

        ax.set_title(scenario_titles.get(scenario, scenario), fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP50 (val)")
        ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels))

    fig.suptitle(
        f"MT 工业检测 few-shot/small：YOLO baseline vs YOLO + L_bias (seed={seed})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.15, 1, 0.95])

    out_path = figs_dir / f"exp3_mt_yolo_map50_vs_epoch_seed{seed}_fewshot_small.png"
    fig.savefig(out_path, dpi=200)
    print(f"[INFO] 已保存图像: {out_path}")


def main() -> None:
    """为三个种子各绘制一张 4 场景图，并额外为 few-shot/small 绘制放大图。"""
    seeds = [1088, 2195, 4960]
    for seed in seeds:
        plot_all_scenarios(seed=seed)
        plot_fewshot_small(seed=seed)


if __name__ == "__main__":
    main()
