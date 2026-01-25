#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
City→Foggy 场景下的 g_phi 规则蒸馏训练脚本（Rule Distillation for MLPFilter）。

目标：
- 使用固定的 YOLO + L_bias 检测模型 θ（已在 City→Foggy 上微调好）；
- 利用简单但“上帝视角”的规则，根据 z = [loss_box, loss_cls, loss_dfl, conf_diff, iou_diff]
  生成样本级伪标签（好样本=1，坏样本=0）；
- 在此基础上监督训练 MLPFilter (g_phi)，得到一个稳定的样本权重预测器。

注意：
- 这里只训练 g_phi，不更新 θ，也不做 meta-learning；
- 阈值采用 batch 内百分位 (percentile)，而不是固定魔法数字；
- 不使用 keep-rate 正则，避免与 BCE 的目标分布冲突。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Any, List

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

BASELINE_ROOT = Path(__file__).resolve().parents[3]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

# 兼容 PyTorch 2.6 默认 weights_only=True：提前允许自定义类反序列化
try:
    import torch.serialization as _ts
    from ICML.src.yolo_bias_finetune.bias_trainer import BiasWrappedModel

    _ts.add_safe_globals([BiasWrappedModel])
except Exception:
    pass

from ICML.cityfog.config import build_cityfog_bias_finetune_config
from ICML.core.yolo_bias_finetune.anchor import AnchorModel
from ICML.core.yolo_bias_finetune.mlp_filter import MLPFilter, PerSampleWeightedDetectionLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 City→Foggy few-shot/small train 上用规则蒸馏训练 g_phi (MLPFilter)"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["few-shot", "small"],
        help="要训练 g_phi 的场景列表（默认：few-shot small）",
    )
    parser.add_argument(
        "--theta-good-seed",
        type=int,
        default=1088,
        help="选择哪个 shared_pretrain_city seed 作为 θ_good",
    )
    parser.add_argument(
        "--theta-seed",
        type=int,
        default=None,
        help="选择哪个 CityFog Bias_only seed 的 θ 作为固定检测模型（不填则使用 --theta-seeds 或配置默认 seeds）",
    )
    parser.add_argument(
        "--theta-seeds",
        nargs="+",
        type=int,
        default=None,
        help="一次训练多个 θ seeds 的 g_phi（默认：使用配置文件中的所有 seeds）",
    )
    parser.add_argument(
        "--lambda-bias",
        type=float,
        default=1e-4,
        help="对应 Bias_only 训练时使用的 λ_bias（用于确定 θ checkpoint 路径）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help="规则蒸馏 g_phi 的训练轮数（遍历 train dataloader 的次数）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="规则蒸馏阶段的 batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="训练 g_phi 使用的设备字符串，例如 '0'、'1' 或 'cpu'",
    )
    parser.add_argument(
        "--keep-rate",
        type=float,
        default=0.95,
        help="规则蒸馏时保留的好样本比例（基于 score 的百分位）",
    )
    parser.add_argument(
        "--lr-phi",
        type=float,
        default=1e-3,
        help="MLPFilter (g_phi) 的学习率",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="rule distill 早停耐心值（按 epoch 计数）",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="早停时认为有提升的最小改变量",
    )
    parser.add_argument(
        "--bad-weight",
        type=float,
        default=0.0,
        help="规则蒸馏中坏样本的软标签权重（默认 0.0，可设为 0.1 以避免完全失明）",
    )
    parser.add_argument(
        "--floor-weight",
        type=float,
        default=0.3,
        help="对样本权重施加保底（Soft Reweighting），避免权重过低导致召回丢失",
    )
    return parser.parse_args()


class CityFogTrainLoader(DetectionTrainer):
    """
    轻量封装 DetectionTrainer，只复用其 build_dataset/get_dataloader 逻辑构建 train dataloader。
    """

    def __init__(self, data_yaml: str, batch_size: int, device: str) -> None:
        # 只传必要的 overrides
        overrides: Dict[str, Any] = {
            "task": "detect",
            "mode": "train",
            "model": None,  # 我们不会在这个 Trainer 中构建模型
            "data": data_yaml,
            "device": device,
            "batch": batch_size,
            "workers": 4,
            "imgsz": 640,
        }
        super().__init__(overrides=overrides)

    def get_model(self, cfg=None, weights=None, verbose=True):
        raise NotImplementedError("CityFogTrainLoader 不负责构建模型")


def build_cityfog_train_loader(data_yaml: str, batch_size: int, device_str: str) -> DataLoader:
    """使用 Ultralytics DetectionTrainer 构建 City→Foggy 的 train dataloader。"""
    trainer = CityFogTrainLoader(data_yaml=data_yaml, batch_size=batch_size, device=device_str)
    train_path, _ = trainer.get_dataset()  # 返回 train_img_path, val_img_path
    # 直接使用 trainer.get_dataloader，让内部正确传递 batch_size；rank=-1 表示非分布式
    train_loader = trainer.get_dataloader(train_path, batch_size=batch_size, rank=-1, mode="train")
    return train_loader


def train_gphi_rule_for_scenario(
    scenario: str,
    theta_good_seed: int,
    theta_seed: int,
    lambda_bias: float,
    epochs: int,
    batch_size: int,
    device_str: str,
    torch_device: torch.device,
    keep_rate: float,
    lr_phi: float,
    patience: int,
    min_delta: float,
    bad_weight: float,
    floor_weight: float,
) -> None:
    """在指定 CityFog 场景上执行规则蒸馏 g_phi 训练。"""
    cfg = build_cityfog_bias_finetune_config(theta_good_seed=theta_good_seed, lambda_bias=lambda_bias)
    if scenario not in cfg.scenario_data_cfg:
        print(f"⚠️ 未知场景: {scenario}，跳过规则蒸馏")
        return

    data_yaml = cfg.scenario_data_cfg[scenario]
    if not data_yaml.exists():
        print(f"⚠️ 数据配置文件不存在，跳过场景 {scenario}: {data_yaml}")
        return

    # 固定检测模型 θ：使用 CityFog Bias_only 的 best.pt
    lambda_dir = f"lambda_{lambda_bias:g}" if lambda_bias != 0 else "lambda_0"
    theta_ckpt = (
        BASELINE_ROOT
        / "ICML"
        / "City→Foggy"
        / "Results"
        / "Bias_only"
        / lambda_dir
        / scenario
        / f"seed_{theta_seed}"
        / "results"
        / "weights"
        / "best.pt"
    )
    if not theta_ckpt.exists():
        print(f"⚠️ 未找到 CityFog Bias_only θ checkpoint: {theta_ckpt}，跳过场景 {scenario}")
        return

    print(f"\n🧪 [Rule Distill g_phi] scenario={scenario}, θ_good_seed={theta_good_seed}, θ_seed={theta_seed}")
    print(f"    data_yaml = {data_yaml}")
    print(f"    θ (Bias_only) = {theta_ckpt}")

    # 构建固定检测模型和锚点 AnchorModel
    yolo = YOLO(str(theta_ckpt))
    det_model = yolo.model
    # 若 checkpoint 中保存的是 BiasWrappedModel，则解包出内部的 DetectionModel
    if hasattr(det_model, "inner"):
        det_model = det_model.inner
    det_model = det_model.to(torch_device).eval()

    anchor = AnchorModel(cfg.theta_good_path, device=torch_device)

    # 独立的 MLPFilter (g_phi) 用于规则蒸馏训练
    # 此处特征维度固定为 5: [loss_box, loss_cls, loss_dfl, conf_diff, iou_diff]
    g_phi = MLPFilter(input_dim=5, device=torch_device)
    g_phi.to(torch_device)

    # PerSampleWeightedDetectionLoss 只用于产出 z 特征（不在其中使用 MLP 权重）
    per_sample_loss = PerSampleWeightedDetectionLoss(
        model=det_model,
        sample_filter=None,
        anchor_model=anchor,
        filter_mode="mlp",  # 仍按 mlp 分支构造 z = [loss_box, loss_cls, loss_dfl, conf_diff, iou_diff]
    )

    optimizer_phi = torch.optim.Adam(g_phi.parameters(), lr=lr_phi)
    bce = nn.BCELoss()

    # 构建 CityFog train dataloader
    train_loader = build_cityfog_train_loader(str(data_yaml), batch_size=batch_size, device_str=device_str)

    best_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        running_loss = 0.0
        total_samples = 0
        print(f"\n=== [Rule Distill g_phi] Epoch {epoch + 1}/{epochs} ({scenario}) ===")

        for step, batch in enumerate(train_loader, start=1):
            imgs = batch["img"].to(torch_device, non_blocking=True).float()
            batch = {k: (v.to(torch_device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            batch["img"] = imgs

            # 1) 固定 θ 和 Anchor，提取 per-sample 特征 z（不需要对 θ 求梯度）
            with torch.no_grad():
                preds = det_model(batch["img"])
                _, _, features = per_sample_loss(preds, batch, return_features=True)  # (B, 5)

            # 特征标准化（批内）：z_norm
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
            z_norm = (features - mean) / std  # (B, 5)

            loss_box = z_norm[:, 0]
            loss_cls = z_norm[:, 1]
            loss_dfl = z_norm[:, 2]
            conf_diff = z_norm[:, 3]
            iou_diff = z_norm[:, 4]

            # 2) 规则定义坏样本：基于 score 的百分位
            # 示例：score = 0.5 * conf_diff + 0.5 * loss_cls
            score = 0.5 * conf_diff + 0.5 * loss_cls
            B = score.shape[0]
            keep = float(keep_rate)
            keep = min(max(keep, 0.0), 1.0)
            k = max(int(keep * B) - 1, 0)
            sorted_score, _ = torch.sort(score)
            thr = sorted_score[k]
            is_bad = score > thr

            # Soft reweighting：为坏样本设置保底权重，避免完全“失明”
            floor_w = float(floor_weight)
            floor_w = min(max(floor_w, 0.0), 1.0)
            bad_w = float(bad_weight)
            bad_w = max(min(bad_w, 1.0), 0.0)
            bad_w = max(bad_w, floor_w)
            target_weights = torch.where(is_bad, bad_w, 1.0).float().view(-1, 1)  # (B,1)
            target_weights = torch.clamp(target_weights, min=floor_w, max=1.0)

            # 3) 训练 g_phi：监督学习 w_i ≈ target_weights
            optimizer_phi.zero_grad()
            pred_weights = g_phi(z_norm)  # (B,1)
            pred_weights = torch.clamp(pred_weights, min=floor_w, max=1.0)

            loss_cheat = bce(pred_weights, target_weights)
            loss_cheat.backward()
            optimizer_phi.step()

            running_loss += float(loss_cheat.detach().cpu().item()) * B
            total_samples += B

            if step % 20 == 0:
                avg = running_loss / max(total_samples, 1)
                print(f"  [step {step}] rule_distill_loss={avg:.4f}")

        epoch_avg = running_loss / max(total_samples, 1)
        print(f"✅ Epoch {epoch + 1} 完成，平均 rule_distill_loss={epoch_avg:.4f}")

        if epoch_avg + min_delta < best_loss:
            best_loss = epoch_avg
            best_state = {k: v.detach().cpu().clone() for k, v in g_phi.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"⏹️ 早停：连续 {patience} 个 epoch 无明显改进（best_loss={best_loss:.4f}）")
                break

    # 保存 g_phi 参数
    out_dir = (
        BASELINE_ROOT
        / "ICML"
        / "City→Foggy"
        / "Results"
        / "Bias+Filter_rule"
        / f"lambda_{lambda_bias:g}"
        / scenario
        / f"seed_{theta_seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mlpfilter_rule.pt"
    state_to_save = best_state if best_state is not None else g_phi.state_dict()
    torch.save(state_to_save, out_path)
    # 记录本次规则蒸馏的关键参数，便于后续溯源
    meta = {
        "scenario": scenario,
        "theta_good_seed": theta_good_seed,
        "theta_seed": theta_seed,
        "lambda_bias": lambda_bias,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(device_str),
        "keep_rate": keep_rate,
        "lr_phi": lr_phi,
        "patience": patience,
        "min_delta": min_delta,
        "bad_weight": bad_weight,
        "floor_weight": floor_weight,
        "best_rule_distill_loss": best_loss if best_loss != float('inf') else None,
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = out_dir / "rule_train_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n💾 已保存规则蒸馏后的 g_phi 参数到: {out_path}")


def main() -> None:
    args = parse_args()

    base_cfg = build_cityfog_bias_finetune_config(theta_good_seed=args.theta_good_seed, lambda_bias=args.lambda_bias)

    # 默认跑配置里的三个 seeds；若显式指定则按用户输入覆盖
    if args.theta_seeds is not None:
        theta_seeds: List[int] = args.theta_seeds
    elif args.theta_seed is not None:
        theta_seeds = [args.theta_seed]
    else:
        theta_seeds = base_cfg.seeds

    device_str = args.device
    if device_str.lower() == "cpu":
        torch_device = torch.device("cpu")
    elif device_str.isdigit():
        torch_device = torch.device(f"cuda:{device_str}")
    else:
        torch_device = torch.device(device_str)

    scenarios: List[str] = args.scenarios

    for scenario in scenarios:
        for theta_seed in theta_seeds:
            train_gphi_rule_for_scenario(
                scenario=scenario,
                theta_good_seed=args.theta_good_seed,
                theta_seed=theta_seed,
                lambda_bias=args.lambda_bias,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device_str=device_str,
                torch_device=torch_device,
                keep_rate=args.keep_rate,
                lr_phi=args.lr_phi,
                patience=args.patience,
                min_delta=args.min_delta,
                bad_weight=args.bad_weight,
                floor_weight=args.floor_weight,
            )


if __name__ == "__main__":
    main()
