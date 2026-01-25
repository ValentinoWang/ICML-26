#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cityscapes → FoggyCityscapes YOLO + L_bias 偏置控制实验入口（CityFog 子包版本）
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List
import sys
import json

# Baseline 根目录：.../Baseline
BASELINE_ROOT = Path(__file__).resolve().parents[2]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from ultralytics import YOLO

from ICML.cityfog.config import build_cityfog_bias_finetune_config
from ICML.core.yolo_bias_finetune.anchor import AnchorModel
from ICML.core.yolo_bias_finetune.bias_trainer import BiasDetectionTrainer


def build_overrides(
    scenario: str,
    training_cfg: Dict[str, Any],
    data_yaml: Path,
    project_dir: Path,
    seed: int,
    model_path: Path,
) -> Dict[str, Any]:
    """构造传给 BiasDetectionTrainer 的 overrides 字典。"""
    scenario_epochs = training_cfg.get("epochs", 5000)

    overrides: Dict[str, Any] = {
        "task": "detect",
        "model": str(model_path),
        "data": str(data_yaml),
        "epochs": scenario_epochs,
        "batch": training_cfg.get("batch_size", 16),
        "workers": training_cfg.get("workers", 8),
        "patience": training_cfg.get("patience", 20),
        "lr0": training_cfg.get("lr", 0.0001),
        "optimizer": training_cfg.get("optimizer", "AdamW"),
        "project": str(project_dir),
        "name": "results",
        "device": training_cfg.get("device", 0),
        "imgsz": training_cfg.get("imgsz", 640),
        "save": True,
        "save_period": training_cfg.get("save_period", -1),
        "verbose": training_cfg.get("verbose", True),
        "plots": training_cfg.get("plots", True),
        "amp": training_cfg.get("amp", True),
        "cache": training_cfg.get("cache", False),
        "freeze": training_cfg.get("freeze", 10),
        "resume": False,
        "seed": seed,
        "deterministic": training_cfg.get("deterministic", True),
    }
    return overrides


def run_cityfog_bias_finetune_for_scenario(
    scenario: str,
    lambda_bias: float = 1e-4,
    theta_good_seed: int = 1088,
) -> None:
    """在指定 Foggy 场景上运行 YOLO + L_bias 微调（遍历所有预设 seeds）。"""
    cfg = build_cityfog_bias_finetune_config(
        theta_good_seed=theta_good_seed,
        lambda_bias=lambda_bias,
    )

    if scenario not in cfg.scenario_data_cfg:
        raise ValueError(f"未知场景: {scenario}")

    data_yaml = cfg.scenario_data_cfg[scenario]
    if not data_yaml.exists():
        raise FileNotFoundError(f"数据配置文件不存在: {data_yaml}")

    if not cfg.theta_good_path.exists():
        raise FileNotFoundError(
            f"θ_good 模型文件不存在: {cfg.theta_good_path} "
            f"(请先在 Cityscapes-DET 上完成 shared_pretrain_city 训练)"
        )

    base_results_root = cfg.results_root / (f"lambda_{lambda_bias:g}" if lambda_bias != 0 else "lambda_0")
    scenario_root = base_results_root / scenario
    scenario_root.mkdir(parents=True, exist_ok=True)

    anchor = AnchorModel(cfg.theta_good_path)

    for seed in cfg.seeds:
        print(f"\n=== City→Foggy 场景 {scenario} | 种子 {seed} ===")
        seed_root = scenario_root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)

        overrides = build_overrides(
            scenario=scenario,
            training_cfg=cfg.training,
            data_yaml=data_yaml,
            project_dir=seed_root,
            seed=seed,
            model_path=cfg.theta_good_path,
        )

        print("开始 City→Foggy YOLO + L_bias 训练")
        trainer = BiasDetectionTrainer(
            anchor_model=anchor,
            lambda_bias=cfg.lambda_bias,
            use_sample_filter=False,
            filter_mode="mlp",
            mlpfilter_init_path=None,
            overrides=overrides,
        )
        trainer.train()

        best_ckpt = Path(trainer.best)
        if best_ckpt.exists():
            yolo_best = YOLO(str(best_ckpt))
            best_model = yolo_best.model
            inner_best = getattr(best_model, "inner", best_model)
            # AnchorModel 不暴露 model.device，使用内部维护的 _device
            inner_best.to(anchor._device)
            bias_loss = anchor.compute_bias_loss(inner_best)
            print(f"[City→Foggy] L_bias(θ_best, θ_good) = {bias_loss.item():.4f}")

            bias_info = {
                "scenario": scenario,
                "seed": seed,
                "lambda_bias": cfg.lambda_bias,
                "theta_good": str(cfg.theta_good_path),
                "bias_loss": float(bias_loss.item()),
                "best_checkpoint": str(best_ckpt),
            }
            out_file = seed_root / "bias_evaluation.json"
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(bias_info, f, indent=2, ensure_ascii=False)
            print(f"已保存 L_bias 评估结果到: {out_file}")
        else:
            print(f"未找到 best.pt，跳过 L_bias 评估: {best_ckpt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="City→Foggy YOLO + L_bias 偏置控制实验")
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
        help="选择哪个 shared_pretrain_city seed 作为 θ_good",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_cityfog_bias_finetune_config(
        theta_good_seed=args.theta_good_seed,
        lambda_bias=args.lambda_bias,
    )

    if args.scenarios:
        scenarios: List[str] = args.scenarios
    else:
        scenarios = list(cfg.scenarios)

    print("🎯 Cityscapes → FoggyCityscapes YOLO + L_bias 偏置控制实验")
    print(f"   场景: {scenarios}")
    print(f"   Seeds: {cfg.seeds}")
    print(f"   θ_good seed: {args.theta_good_seed}")
    print(f"   λ_bias: {args.lambda_bias}")

    for scenario in scenarios:
        if scenario not in cfg.scenario_data_cfg:
            print(f"⚠️ 跳过未知场景: {scenario}")
            continue
        run_cityfog_bias_finetune_for_scenario(
            scenario=scenario,
            lambda_bias=args.lambda_bias,
            theta_good_seed=args.theta_good_seed,
        )

    print("✅ 所有 City→Foggy YOLO + L_bias 实验完成")


if __name__ == "__main__":
    main()
