#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO + L_bias 微调实验脚本（MT/Toy 场景）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch

from ICML.mt.config import build_bias_finetune_config, build_bias_finetune_noise_config
from .anchor import AnchorModel
from .bias_trainer import BiasDetectionTrainer


def build_overrides(
    scenario: str,
    training_cfg: Dict[str, Any],
    data_yaml: Path,
    project_dir: Path,
    seed: int,
    model_path: Path,
) -> Dict[str, Any]:
    """构造传给 BiasDetectionTrainer 的 overrides 字典。"""
    scenario_epochs = training_cfg.get("epochs", 500)

    overrides: Dict[str, Any] = {
        "task": "detect",
        "model": str(model_path),
        "data": str(data_yaml),
        "epochs": scenario_epochs,
        "batch": training_cfg.get("batch_size", 90),
        "workers": training_cfg.get("workers", 10),
        "patience": training_cfg.get("patience", 30),
        "lr0": training_cfg.get("lr", 0.001),
        "optimizer": training_cfg.get("optimizer", "AdamW"),
        "project": str(project_dir),
        # 与 Pretrain-Finetune 目录风格统一：每个 seed 下固定一个 results 目录
        "name": "results",
        "device": training_cfg.get("device", 0),
        "imgsz": training_cfg.get("imgsz", 640),
        "save": True,
        "save_period": training_cfg.get("save_period", -1),
        "verbose": training_cfg.get("verbose", True),
        "plots": training_cfg.get("plots", True),
        "amp": training_cfg.get("amp", True),
        "cache": training_cfg.get("cache", False),
        "resume": False,
        "seed": seed,
        "deterministic": training_cfg.get("deterministic", True),
    }
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO + L_bias 微调实验 (Toy)")
    parser.add_argument(
        "--scenario",
        type=str,
        default="few-shot",
        choices=["few-shot", "small", "medium", "high"],
        help="目标域场景名称",
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
        help="选择哪个 shared_pretrain seed 作为 θ_good",
    )
    parser.add_argument(
        "--use-filter",
        action="store_true",
        help="是否启用样本过滤器 g_phi（YOLO + L_bias + g_phi）",
    )
    args = parser.parse_args()

    run_bias_finetune_for_scenario(
        scenario=args.scenario,
        lambda_bias=args.lambda_bias,
        theta_good_seed=args.theta_good_seed,
        use_sample_filter=args.use_filter,
        filter_mode="mlp",
    )


def run_bias_finetune_for_scenario(
    scenario: str,
    lambda_bias: float = 1e-4,
    theta_good_seed: int = 1088,
    use_sample_filter: bool = False,
    filter_mode: str = "mlp",
) -> None:
    """在指定场景上运行 YOLO + L_bias 微调（对所有预设 seeds 循环）。"""
    cfg = build_bias_finetune_config(
        theta_good_seed=theta_good_seed,
        lambda_bias=lambda_bias,
    )

    if scenario not in cfg.scenario_data_cfg:
        raise ValueError(f"未知场景: {scenario}")

    data_yaml = cfg.scenario_data_cfg[scenario]
    if not data_yaml.exists():
        raise FileNotFoundError(f"数据配置文件不存在: {data_yaml}")

    if not cfg.theta_good_path.exists():
        raise FileNotFoundError(f"θ_good 模型文件不存在: {cfg.theta_good_path}")

    # 根据是否启用样本过滤器 g_phi（MLP FiLTER）区分结果根目录，避免与纯 YOLO+L_bias 结果混在一起；
    # 同时在 Bias_only 路径下按 λ_bias 再细分一层子目录，显式区分不同 λ 的实验。
    if use_sample_filter:
        # 根据 results_root 的名称自动推导对应的 Bias+Filter 目录名：
        # - Bias_only        -> Bias+Filter
        # - Bias_only_noise  -> Bias+Filter_noise
        root_name = cfg.results_root.name
        if root_name.startswith("Bias_only"):
            filter_root_name = root_name.replace("Bias_only", "Bias+Filter", 1)
        else:
            filter_root_name = "Bias+Filter"
        base_results_root = cfg.results_root.parent / filter_root_name
    else:
        # 在 Bias_only 或 Bias_only_noise 路径下按 λ_bias 再细分一层子目录，显式区分不同 λ 的实验。
        if lambda_bias == 0:
            lambda_dir = "lambda_0"
        else:
            lambda_dir = f"lambda_{lambda_bias:g}"
        base_results_root = cfg.results_root / lambda_dir

    scenario_root = base_results_root / scenario
    scenario_root.mkdir(parents=True, exist_ok=True)

    anchor = AnchorModel(cfg.theta_good_path)

    for seed in cfg.seeds:
        print(f"\n=== 场景 {scenario} | 种子 {seed} ===")
        # 与 Pretrain-Finetune/Results 目录模式保持一致:
        #   Toy/Results/yolo_bias_finetune/<scenario>/seed_<seed>/results/...
        seed_root = scenario_root / f"seed_{seed}"
        run_root = seed_root / "results"
        seed_root.mkdir(parents=True, exist_ok=True)

        overrides = build_overrides(
            scenario=scenario,
            training_cfg=cfg.training,
            data_yaml=data_yaml,
            project_dir=seed_root,  # project=seed_root
            seed=seed,
            model_path=cfg.theta_good_path,
        )

        if use_sample_filter:
            print("开始 YOLO + L_bias + g_phi 训练（启用样本过滤器，MLP + meta-init 若可用）")
            mlpfilter_init_path = seed_root / "mlpfilter_meta.pt"
        else:
            print("开始 YOLO + L_bias 训练（未启用样本过滤器）")
            mlpfilter_init_path = None

        trainer = BiasDetectionTrainer(
            anchor_model=anchor,
            lambda_bias=cfg.lambda_bias,
            use_sample_filter=use_sample_filter,
            filter_mode=filter_mode,
            mlpfilter_init_path=mlpfilter_init_path,
            overrides=overrides,
        )
        trainer.train()

        best_ckpt = Path(trainer.best)
        if best_ckpt.exists():
            model = trainer.model
            if hasattr(model, "inner"):
                inner_model = model.inner
            else:
                inner_model = model

            inner_model.to(trainer.device)
            bias_loss = anchor.compute_bias_loss(inner_model)
            print(f"L_bias(θ_new, θ_good) = {bias_loss.item():.4f}")

            bias_info = {
                "scenario": scenario,
                "seed": seed,
                "lambda_bias": cfg.lambda_bias,
                "theta_good": str(cfg.theta_good_path),
                "bias_loss": float(bias_loss.item()),
                "best_checkpoint": str(best_ckpt),
            }
            # 将偏置评估结果保存在 seed 根目录，便于与 results 目录并列查看
            out_file = seed_root / "bias_evaluation.json"
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(bias_info, f, indent=2, ensure_ascii=False)
            print(f"已保存 L_bias 评估结果到: {out_file}")
        else:
            print(f"未找到 best.pt，跳过 L_bias 评估: {best_ckpt}")


def run_bias_finetune_for_scenario_noise(
    scenario: str,
    lambda_bias: float = 1e-4,
    theta_good_seed: int = 1088,
    use_sample_filter: bool = False,
    filter_mode: str = "mlp",
) -> None:
    """
    在带显式噪声 train（D_noise + OOD）的配置下运行 YOLO + L_bias 微调（few-shot/small）。

    与 run_bias_finetune_for_scenario 的主要区别：
    - 使用 build_bias_finetune_noise_config，data_yaml 指向 *_noise.yaml；
    - results_root 为 Toy/Results/Bias_only_noise，下游 Bias+Filter_noise 由此推导。
    """
    cfg = build_bias_finetune_noise_config(
        theta_good_seed=theta_good_seed,
        lambda_bias=lambda_bias,
    )

    if scenario not in cfg.scenario_data_cfg:
        raise ValueError(f"未知场景（或未配置噪声版）: {scenario}")

    data_yaml = cfg.scenario_data_cfg[scenario]
    if not data_yaml.exists():
        raise FileNotFoundError(f"数据配置文件不存在: {data_yaml}")

    if not cfg.theta_good_path.exists():
        raise FileNotFoundError(f"θ_good 模型文件不存在: {cfg.theta_good_path}")

    # 根据是否启用样本过滤器 g_phi 区分结果根目录；在 Bias_only_noise 路径下按 λ_bias 再细分一层。
    if use_sample_filter:
        base_results_root = cfg.results_root.parent / "Bias+Filter_noise"
    else:
        if lambda_bias == 0:
            lambda_dir = "lambda_0"
        else:
            lambda_dir = f"lambda_{lambda_bias:g}"
        base_results_root = cfg.results_root / lambda_dir

    scenario_root = base_results_root / scenario
    scenario_root.mkdir(parents=True, exist_ok=True)

    anchor = AnchorModel(cfg.theta_good_path)

    for seed in cfg.seeds:
        print(f"\n=== [NOISE] 场景 {scenario} | 种子 {seed} ===")
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

        if use_sample_filter:
            print("开始 YOLO + L_bias + g_phi 训练（噪声版，启用样本过滤器，MLP + meta-init 若可用）")
            mlpfilter_init_path = seed_root / "mlpfilter_meta.pt"
        else:
            print("开始 YOLO + L_bias 训练（噪声版，未启用样本过滤器）")
            mlpfilter_init_path = None

        trainer = BiasDetectionTrainer(
            anchor_model=anchor,
            lambda_bias=cfg.lambda_bias,
            use_sample_filter=use_sample_filter,
            filter_mode=filter_mode,
            mlpfilter_init_path=mlpfilter_init_path,
            overrides=overrides,
        )
        trainer.train()

        best_ckpt = Path(trainer.best)
        if best_ckpt.exists():
            model = trainer.model
            inner_model = model.inner if hasattr(model, "inner") else model
            inner_model.to(trainer.device)
            bias_loss = anchor.compute_bias_loss(inner_model)
            print(f"[NOISE] L_bias(θ_new, θ_good) = {bias_loss.item():.4f}")

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
            print(f"[NOISE] 已保存 L_bias 评估结果到: {out_file}")
        else:
            print(f"[NOISE] 未找到 best.pt，跳过 L_bias 评估: {best_ckpt}")


if __name__ == "__main__":
    main()
