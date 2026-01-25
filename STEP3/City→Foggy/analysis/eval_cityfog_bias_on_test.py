#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在 Foggy 全局测试集 (split='test') 上评估 ICML City→Foggy 偏置控制模型。

目的：
- 与 Pretrain-Finetune/City→Foggy baseline 使用同一个 split 和 data.yaml；
- 读取 ICML/Results_CityFog/Bias_only 下的 best.pt，复用 baseline 的 common.py 评估逻辑。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import sys
import argparse

# Baseline 根目录：.../Baseline
BASELINE_ROOT = Path(__file__).resolve().parents[3]

# 将 City→Foggy 目录加入 sys.path，使用其中的 common.py
CITYFOG_ROOT = BASELINE_ROOT / "Pretrain-Finetune" / "City→Foggy"
if str(CITYFOG_ROOT) not in sys.path:
    sys.path.insert(0, str(CITYFOG_ROOT))

from common import FOGGY_SCENARIO_YAMLS, resolve_path, setup_ultralytics  # type: ignore

setup_ultralytics()
from ultralytics import YOLO  # noqa: E402


def eval_icml_cityfog_on_test(
    results_root: Path,
    scenarios: List[str],
    seeds: List[int],
) -> Dict[str, Dict[int, Dict[str, float]]]:
    out: Dict[str, Dict[int, Dict[str, float]]] = {}

    for scenario in scenarios:
        if scenario not in FOGGY_SCENARIO_YAMLS:
            raise ValueError(f"未知场景: {scenario}")
        data_yaml = resolve_path(FOGGY_SCENARIO_YAMLS[scenario])

        per_seed: Dict[int, Dict[str, float]] = {}
        for seed in seeds:
            ckpt = (
                results_root
                / scenario
                / f"seed_{seed}"
                / "results"
                / "weights"
                / "best.pt"
            )
            if not ckpt.exists():
                print(f"⚠️ 跳过 seed {seed}, 未找到 ICML best.pt: {ckpt}")
                continue

            print(f"\n=== [ICML Bias] City→Foggy 测试 (scenario={scenario}, seed={seed}) ===")
            print(f"模型: {ckpt}")
            print(f"数据配置: {data_yaml}")

            yolo = YOLO(str(ckpt))
            # 训练阶段使用的是 BiasWrappedModel，这里评估时只需要原始 DetectionModel，
            # 因此若存在 inner 属性则解包，避免 AutoBackend 访问不到 fuse() 等方法。
            if hasattr(yolo.model, "inner"):
                yolo.model = yolo.model.inner

            metrics = yolo.val(
                data=data_yaml,
                split="test",
                imgsz=640,
                batch=64,
                device="0",
                workers=8,
                save=False,
                verbose=True,
                project=str(results_root / "eval_foggy_test"),
                name=f"icml_{scenario}_seed_{seed}",
                exist_ok=True,
            )

            results = metrics.results_dict
            per_seed[seed] = {
                "precision": float(results.get("metrics/precision(B)", 0.0)),
                "recall": float(results.get("metrics/recall(B)", 0.0)),
                "mAP50": float(results.get("metrics/mAP50(B)", 0.0)),
                "mAP50-95": float(results.get("metrics/mAP50-95(B)", 0.0)),
            }

        out[scenario] = per_seed

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="在 Foggy 测试集上评估 ICML City→Foggy 偏置控制模型")
    parser.add_argument(
        "--lambda-bias",
        type=float,
        default=1e-4,
        help="对应训练时使用的 λ_bias（决定 Results_CityFog/<exp_subdir>/lambda_xxx 子目录）",
    )
    parser.add_argument(
        "--exp-subdir",
        type=str,
        default="Bias_only",
        help="实验结果子目录名：Bias_only 或 Bias+Filter 等（默认：Bias_only）",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["few-shot", "small"],
        help="要评估的场景列表（默认：few-shot small）",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1088, 2195, 4960],
        help="要评估的随机种子列表（默认：1088 2195 4960）",
    )
    args = parser.parse_args()

    if args.lambda_bias == 0:
        lambda_dir = "lambda_0"
    else:
        lambda_dir = f"lambda_{args.lambda_bias:g}"

    results_root = (
        BASELINE_ROOT
        / "ICML"
        / "City→Foggy"
        / "Results"
        / args.exp_subdir
        / lambda_dir
    )
    results_root.mkdir(parents=True, exist_ok=True)

    results = eval_icml_cityfog_on_test(results_root, args.scenarios, args.seeds)

    summary_path = results_root / "icml_cityfog_bias_eval_foggy_test_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📂 ICML City→Foggy 偏置控制模型的 Foggy 测试集评估结果已保存到: {summary_path}")


if __name__ == "__main__":
    main()
