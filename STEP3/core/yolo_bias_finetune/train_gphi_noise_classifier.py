#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
阶段 1：在显式噪声环境下对 g_phi (MLPFilter) 进行监督训练，将其作为“噪声检测器”预训练。

设计要点：
- 数据：使用 MT-tgt-split-noise/MT-tgt_*_train_mix（clean + D_noise + OOD）；
- 监督标签：
    - 若图像文件名属于原始 MT-tgt-split 的 train 图，则视为 clean，目标权重 w_i* = 1；
    - 否则视为 noise/OOD，目标权重 w_i* = 0；
- 模型：
    - θ：使用固定的 YOLO 检测模型（θ_good），不更新；
    - g_phi：MLPFilter，仅更新 φ；
- 训练目标：
    - 使用 PerSampleWeightedDetectionLoss 生成特征并得到当前 w_i；
    - 以 L_sup = MSE(w_i, w_i*) 训练 g_phi。

训练完成后，将 g_phi 参数保存到：
    Toy/Results/Bias+Filter_noise/<scenario>/seed_<seed>/mlpfilter_meta.pt
后续噪声版 Bias+Filter 训练会自动从该路径加载作为初始化。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List, Set

import torch

from ultralytics.models.yolo.detect import DetectionTrainer

BASELINE_ROOT = Path(__file__).resolve().parents[3]  # .../Baseline
ICML_ROOT = BASELINE_ROOT / "ICML"

if str(BASELINE_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(BASELINE_ROOT))

from ICML.mt.config import build_bias_finetune_noise_config
from ICML.core.yolo_bias_finetune.train_bias_yolo import build_overrides
from ICML.core.yolo_bias_finetune.anchor import AnchorModel
from ICML.core.yolo_bias_finetune.mlp_filter import MLPFilter, PerSampleWeightedDetectionLoss


def build_clean_name_set(scenario: str) -> Set[str]:
    """根据场景构造 clean 图像文件名集合，用于区分 clean vs noise/OOD。"""
    data_root = Path("/root/autodl-tmp/dataset/MT-tgt-split")
    scen_dir = data_root / f"MT-tgt_{scenario}_train" / "images"
    if not scen_dir.exists():
        return set()
    return {p.name for p in scen_dir.glob("*.jpg")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="阶段 1：在显式噪声环境下监督训练 g_phi (MLPFilter) 作为噪声检测器"
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=["few-shot", "small"],
        help="要运行的目标域场景列表（默认：few-shot 和 small）",
    )
    parser.add_argument(
        "--theta-good-seed",
        type=int,
        default=1088,
        help="选择哪个 shared_pretrain seed 作为 θ_good",
    )
    parser.add_argument(
        "--meta-batch-size",
        type=int,
        default=32,
        help="噪声监督训练的 batch size",
    )
    parser.add_argument(
        "--lr-phi",
        type=float,
        default=1e-3,
        help="MLP FiLTER (g_phi) 的学习率",
    )
    return parser.parse_args()


def train_gphi_noise_classifier_for_scenarios(
    scenarios: List[str],
    theta_good_seed: int = 1088,
    meta_batch_size: int = 32,
    lr_phi: float = 1e-3,
) -> None:
    """
    在给定场景列表上执行“训练 1”：在噪声混合 train 上监督训练 g_phi 作为噪声检测器。

    供一键脚本直接调用，避免遗漏预训练步骤。
    """
    cfg = build_bias_finetune_noise_config(
        theta_good_seed=theta_good_seed,
        lambda_bias=1e-4,  # 这里只需 θ_good，λ 对 g_phi 训练无影响
    )

    valid_scenarios: List[str] = [s for s in scenarios if s in cfg.scenario_data_cfg]
    if not valid_scenarios:
        print("⚠️ 未找到可用场景（few-shot/small），退出 g_phi 噪声预训练。")
        return

    for scenario in valid_scenarios:
        data_yaml = cfg.scenario_data_cfg[scenario]
        if not data_yaml.exists():
            print(f"⚠️ 数据配置文件不存在，跳过场景 {scenario}: {data_yaml}")
            continue

        print(
            f"\n🧪 [Train g_phi Noise-Classifier] scenario={scenario}, "
            f"θ_good={cfg.theta_good_path}"
        )

        # clean 文件名集合，用于生成监督标签 w_i*
        clean_names = build_clean_name_set(scenario)
        print(f"[INFO] scenario={scenario}, clean image names={len(clean_names)}")

        # 与 Bias+Filter_noise 训练保持路径约定一致
        mt_results_root = cfg.results_root.parent  # .../ICML/mt/Results
        bias_filter_noise_root = mt_results_root / "Bias+Filter_noise"

        for seed in cfg.seeds:
            print(f"\n=== 场景 {scenario} | 种子 {seed} | 训练 g_phi 噪声检测器 ===")

            # DetectionTrainer 仅用于复用 YOLO dataloader + 模型构建逻辑
            project_dir = (
                bias_filter_noise_root
                / scenario
                / f"seed_{seed}"
                / "gphi_noise_tmp"
            )
            overrides: Dict[str, Any] = build_overrides(
                scenario=scenario,
                training_cfg=cfg.training,
                data_yaml=data_yaml,
                project_dir=project_dir,
                seed=seed,
                model_path=cfg.theta_good_path,  # 使用 θ_good 权重
            )
            overrides["epochs"] = 1
            overrides["batch"] = meta_batch_size
            overrides["workers"] = min(overrides.get("workers", 10), 4)

            trainer = DetectionTrainer(overrides=overrides)
            trainer._setup_train(world_size=0)

            device = trainer.device
            det_model = trainer.model.to(device)
            # 冻结 YOLO 参数，仅训练 g_phi
            for p in det_model.parameters():
                p.requires_grad_(False)

            anchor = AnchorModel(cfg.theta_good_path, device=device)
            mlp_filter = MLPFilter(device=device).to(device)
            per_sample_loss = PerSampleWeightedDetectionLoss(
                det_model,
                mlp_filter,
                anchor_model=anchor,
            )

            # 优化器在第一次前向、MLPFilter 完成构建之后再延迟初始化
            opt_phi = None
            train_loader = trainer.train_loader
            if train_loader is None:
                print(f"⚠️ train_loader 为空，跳过场景 {scenario} seed {seed}")
                continue

            # 使用与 YOLO 相同的 epoch / patience 配置进行监督训练
            max_epochs = cfg.training.get("epochs", 500)
            patience = cfg.training.get("patience", 20)
            best_loss = float("inf")
            best_epoch = -1

            for epoch in range(max_epochs):
                det_model.train()
                running_loss = 0.0
                n_batches = 0
                print(f"[Epoch {epoch + 1}/{max_epochs}]")

                for batch in train_loader:
                    batch = trainer.preprocess_batch(batch)
                    imgs = batch["img"].to(device)

                    # 1) 前向构造 g_phi 的 w_i（通过 PerSampleWeightedDetectionLoss）
                    preds = det_model(imgs)
                    per_sample_loss(preds, batch)  # 忽略返回的 detection loss，只用 last_weights
                    w = per_sample_loss.last_weights  # (B,)
                    if w is None:
                        continue

                    # 2) 根据图像文件名构造监督标签 w_i*（clean=1, noise/OOD=0）
                    im_files = batch.get("im_file") or batch.get("im_files") or batch.get("paths")
                    if im_files is None:
                        continue
                    if isinstance(im_files, (tuple, list)):
                        names = [os.path.basename(str(p)) for p in im_files]
                    else:
                        names = [os.path.basename(str(im_files))]

                    target = torch.zeros_like(w)
                    for i, name in enumerate(names):
                        if name in clean_names:
                            target[i] = 1.0  # clean
                        else:
                            target[i] = 0.0  # noise/OOD

                    # 3) 监督损失：MSE(w_i, w_i*)
                    L_sup = torch.mean((w - target.to(w.device)) ** 2)

                    # 在第一次拿到有效的 w_i 时，再根据实际特征维度构建 MLP head 并创建优化器
                    if opt_phi is None:
                        if not any(p.requires_grad for p in mlp_filter.parameters()):
                            # 若此时仍无可训练参数，跳过本 batch
                            continue
                        opt_phi = torch.optim.Adam(mlp_filter.parameters(), lr=lr_phi)

                    opt_phi.zero_grad()
                    L_sup.backward()
                    opt_phi.step()

                    running_loss += float(L_sup.detach().cpu().item())
                    n_batches += 1

                if n_batches == 0:
                    print("⚠️ 本 epoch 无有效 batch，提前结束。")
                    break

                avg_epoch = running_loss / n_batches
                print(f"✅ Epoch {epoch + 1} 完成，L_sup(avg)={avg_epoch:.4f}")

                # 简单 early stopping 策略
                if avg_epoch < best_loss - 1e-4:
                    best_loss = avg_epoch
                    best_epoch = epoch
                elif epoch - best_epoch >= patience:
                    print(
                        f"⏹️ 早停：{epoch - best_epoch} epochs 无改进 "
                        f"(best={best_loss:.4f} @ epoch {best_epoch + 1})"
                    )
                    break

            # 保存 g_phi 的参数到 Bias+Filter_noise/<scenario>/seed_xxx/mlpfilter_meta.pt
            out_dir = bias_filter_noise_root / scenario / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "mlpfilter_meta.pt"
            torch.save(mlp_filter.state_dict(), out_path)
            print(f"💾 已保存 g_phi 噪声检测器参数到: {out_path}")


def main() -> None:
    args = parse_args()
    train_gphi_noise_classifier_for_scenarios(
        scenarios=args.scenarios,
        theta_good_seed=args.theta_good_seed,
        meta_batch_size=args.meta_batch_size,
        lr_phi=args.lr_phi,
    )


if __name__ == "__main__":
    main()
