#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MT/Toy 数据集上的 YOLO + L_bias 偏置控制实验入口
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import sys

# 将 Baseline 根目录加入 sys.path，确保可以导入 ICML 包
BASELINE_ROOT = Path(__file__).resolve().parents[2]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from ICML.mt.config import build_bias_finetune_config
from ICML.core.yolo_bias_finetune.train_bias_yolo import run_bias_finetune_for_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键运行 YOLO + L_bias 偏置控制实验（MT/Toy）")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="要运行的目标域场景列表（默认：配置文件中的所有场景）",
    )
    parser.add_argument(
        "--lambda-bias",
        type=float,
        default=1e-4,
        help="L_bias 的损失权重 λ_bias",
    )
    parser.add_argument(
        "--theta-good-seed",
        type=int,
        default=1088,
        help="选择哪个 shared_pretrain seed 作为 θ_good（对应 Pretrain-Finetune/Results/shared_pretrain/seed_xxx）",
    )
    parser.add_argument(
        "--use-filter",
        action="store_true",
        help="是否启用样本过滤器 g_phi（YOLO + L_bias + g_phi）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = build_bias_finetune_config(
        theta_good_seed=args.theta_good_seed,
        lambda_bias=args.lambda_bias,
    )

    if args.scenarios:
        scenarios: List[str] = args.scenarios
    else:
        scenarios = list(cfg.scenarios)

    print("🎯 YOLO + L_bias 偏置控制实验（MT/Toy，一键模式）")
    print(f"   场景: {scenarios}")
    print(f"   Seeds: {cfg.seeds}")
    print(f"   θ_good seed: {args.theta_good_seed}")
    print(f"   λ_bias: {args.lambda_bias}")
    print(f"   使用样本过滤器 g_phi: {args.use_filter}")

    for scenario in scenarios:
        if scenario not in cfg.scenario_data_cfg:
            print(f"⚠️ 跳过未知场景: {scenario}")
            continue
        run_bias_finetune_for_scenario(
            scenario=scenario,
            lambda_bias=args.lambda_bias,
            theta_good_seed=args.theta_good_seed,
            use_sample_filter=args.use_filter,
        )

    if args.use_filter:
        print("✅ 所有 YOLO + L_bias + g_phi 实验完成")
    else:
        print("✅ 所有 YOLO + L_bias 实验完成")


if __name__ == "__main__":
    main()
