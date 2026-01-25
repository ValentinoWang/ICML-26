#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick helper to plot mAP50 curves from Results_stream/*/metrics.json.
Usage:
  python plot_map50.py --scenario few-shot --out Results_stream/plots/few-shot_map50.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def load_series(scenario: str, method_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    """
    Return {seed: [(epoch, map50), ...]} for a method/scenario.
    """
    series: Dict[str, List[Tuple[int, float]]] = {}
    for seed_dir in (method_dir / scenario).glob("seed_*"):
        metrics_path = seed_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open() as f:
            data = json.load(f)
        points: List[Tuple[int, float]] = []
        for row in data:
            epoch = row.get("epoch")
            m = row.get("eval_map50")
            if epoch is None or m is None:
                continue
            points.append((int(epoch), float(m)))
        if points:
            series[seed_dir.name] = sorted(points, key=lambda x: x[0])
    return series


def plot_map50(scenario: str, methods: List[str], out_path: Path, show_seeds: bool = False, results_root: Path | None = None) -> None:
    plt.figure(figsize=(8, 4.5))
    results_root = results_root or (ROOT / "Results_stream")

    # 收集所有方法的曲线
    all_epochs = set()
    per_method: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    for method in methods:
        method_dir = results_root / method
        if not method_dir.exists():
            continue
        series = load_series(scenario, method_dir)
        if not series:
            continue
        per_method[method] = series
        for pts in series.values():
            for ep, _ in pts:
                all_epochs.add(ep)

    if not all_epochs:
        print("No data found.")
        return

    epoch_axis = sorted(all_epochs)

    def interp(xs: List[int], ys: List[float], target_epochs: List[int]) -> List[float]:
        if not xs or not ys:
            return [np.nan] * len(target_epochs)
        xs_np = np.array(xs, dtype=float)
        ys_np = np.array(ys, dtype=float)
        return np.interp(target_epochs, xs_np, ys_np, left=np.nan, right=np.nan).tolist()

    # 先画平均曲线，再画各 seed 曲线（虚线）
    for method, series in per_method.items():
        curves = []
        for seed, pts in series.items():
            ep, val = zip(*pts)
            curves.append(interp(list(ep), list(val), epoch_axis))
            if show_seeds:
                plt.plot(ep, val, linestyle="--", alpha=0.35, label=f"{method} ({seed})")
        if curves:
            arr = np.array(curves, dtype=float)
            mean = np.nanmean(arr, axis=0)
            plt.plot(epoch_axis, mean, marker="o", linestyle="-", linewidth=2, label=f"{method} (mean)")
    plt.xlabel("Epoch")
    plt.ylabel("mAP50")
    plt.title(f"mAP50 vs Epoch - {scenario}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="few-shot")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["Baseline_SelfTraining", "TENT", "EATA_Lite", "Ours_Bias_only"],
    )
    parser.add_argument("--out", type=Path, default=Path("Results_stream/plots/map50.png"))
    parser.add_argument("--show-seeds", action="store_true", help="是否绘制单独的 seed 曲线（默认为仅均值）")
    parser.add_argument("--root", type=Path, default=ROOT / "Results_stream", help="结果根目录（可切换到 Results_stream_nofreeze 等）")
    args = parser.parse_args()

    plot_map50(args.scenario, args.methods, args.out, show_seeds=args.show_seeds, results_root=args.root)


if __name__ == "__main__":
    main()
