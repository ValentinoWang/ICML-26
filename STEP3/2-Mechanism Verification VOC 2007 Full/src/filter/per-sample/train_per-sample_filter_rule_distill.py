#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rule distillation for per-sample filter (g_phi) on noisy VOC2007:
- 数据：voc07_noisy.yaml（含“背景毒药”噪声）
- 特征：PerSampleWeightedDetectionLoss 提供的 z = [loss_box, loss_cls, loss_dfl, conf_diff, iou_diff]
- 规则标签：使用 loss_sum = loss_box + loss_cls + loss_dfl，大于 loss_thres(默认 0.3) 判为噪声样本 target=0，其余 target=1
- 训练：仅更新 g_phi（MLPFilter，内部带 BN），YOLO θ 固定（使用 anchor_voc.pt），早停 patience=20，max_epochs=10000
- 产物：mlp_filter_voc.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any

import torch
from torch import nn

from ultralytics.models.yolo.detect import DetectionTrainer

# 将 Baseline 根目录加入 sys.path，复用核心模块
THIS_FILE = Path(__file__).resolve()
ICML_ROOT = THIS_FILE.parents[3]  # /root/autodl-tmp/ICML
BASELINE_ROOT = ICML_ROOT.parent
import sys

if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from ICML.core.yolo_bias_finetune.anchor import AnchorModel
from ICML.core.yolo_bias_finetune.mlp_filter import MLPFilter, PerSampleWeightedDetectionLoss


DEFAULT_ANCHOR_DIR = (
    ICML_ROOT / "2-Mechanism Verification VOC 2007 Full" / "Results" / "Anchor" / "voc07_clean"
)
DEFAULT_DISTILL_DIR = (
    ICML_ROOT / "2-Mechanism Verification VOC 2007 Full" / "Results" / "distill_rule_per-sample" / "voc07_noisy"
)


def build_overrides(
    data_yaml: Path,
    model_path: Path,
    project_dir: Path,
    batch_size: int,
    seed: int,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "data": str(data_yaml),
        "model": str(model_path),
        "project": str(project_dir),
        "name": "sample_filter_rule_distill",
        "epochs": 1,  # dataloader 构建占位
        "batch": batch_size,
        "imgsz": 640,
        "workers": 4,
        "seed": seed,
    }
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule distillation for g_phi on noisy VOC2007")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/root/autodl-tmp/dataset/voc07_noisy_per-sample/voc07_noisy.yaml"),
        help="voc07_noisy.yaml 路径（per-sample 整图毒版本，默认指向 dataset 归档位置）",
    )
    parser.add_argument(
        "--anchor",
        type=Path,
        default=None,
        help="干净 VOC 训练得到的 anchor_voc_<model>.pt（默认自动在 Results/Anchor/voc07_clean 下取最新）",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="训练 batch size")
    parser.add_argument("--noise-ratio", type=float, default=0.3, help="保留参数，不再用于规则（兼容旧接口）")
    parser.add_argument("--lr-phi", type=float, default=1e-3, help="g_phi 学习率")
    parser.add_argument("--max-epochs", type=int, default=10000, help="最大轮数（早停为主，默认 10000）")
    parser.add_argument("--patience", type=int, default=20, help="Loss 连续多少轮不下降则早停（默认 20）")
    parser.add_argument("--seed", type=int, default=1088, help="随机种子")
    parser.add_argument(
        "--loss-thres",
        type=float,
        default=0.3,
        help="loss_sum (box+cls+dfl) 阈值，大于则判为噪声样本（权重 0）",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="保存 g_phi 权重的路径；若不指定，自动保存到 Results/distill_rule_per-sample/voc07_noisy/mlp_filter_<anchor>.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"数据 YAML 不存在: {args.data}")

    anchor_path = args.anchor
    if anchor_path is None:
        candidates = sorted(
            DEFAULT_ANCHOR_DIR.glob("anchor_voc_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"未找到 anchor 模型，请指定 --anchor 或在 {DEFAULT_ANCHOR_DIR} 下放置 anchor_voc_*.pt"
            )
        anchor_path = candidates[0]
    if not anchor_path.exists():
        raise FileNotFoundError(f"anchor_voc.pt 不存在: {anchor_path}")

    save_path = args.save_path
    if save_path is None:
        tag = anchor_path.stem.replace("anchor_", "")
        save_path = DEFAULT_DISTILL_DIR / f"mlp_filter_{tag}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print("🧪 Train g_phi (sample filter) via rule distillation on noisy VOC2007")
    print(f"   data: {args.data}")
    print(f"   anchor: {anchor_path}")
    print(f"   save_path: {save_path}")
    print(f"   batch_size: {args.batch_size}, noise_ratio: {args.noise_ratio}")
    print(f"   lr_phi: {args.lr_phi}, max_epochs: {args.max_epochs}, patience: {args.patience}")

    project_dir = save_path.parent  # 不再创建额外子目录，直接在 distill 根目录下记录
    overrides = build_overrides(
        data_yaml=args.data,
        model_path=anchor_path,
        project_dir=project_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    trainer = DetectionTrainer(overrides=overrides)
    trainer._setup_train(world_size=0)
    run_dir = save_path.parent  # 与权重同级，避免额外 runs_* 目录
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"
    args_path = run_dir / "args.json"

    meta = {
        "data": str(args.data),
        "anchor": str(anchor_path),
        "save_path": str(save_path),
        "batch_size": args.batch_size,
        "noise_ratio": args.noise_ratio,
        "lr_phi": args.lr_phi,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "seed": args.seed,
        "loss_thres": args.loss_thres,
    }
    with args_path.open("w") as f:
        json.dump(meta, f, indent=2)

    device = trainer.device
    det_model = trainer.model.to(device)
    for p in det_model.parameters():
        p.requires_grad_(False)
    det_model.eval()

    anchor = AnchorModel(anchor_path, device=device)
    mlp_filter = MLPFilter(device=device).to(device)
    per_sample_loss = PerSampleWeightedDetectionLoss(
        det_model,
        sample_filter=None,
        anchor_model=anchor,
    )

    optimizer = None  # 延迟到第一次前向后再初始化，避免空参数列表

    train_loader = trainer.train_loader
    if train_loader is None:
        raise RuntimeError("train_loader 为空，检查数据配置")

    best_loss = float("inf")
    patience_left = args.patience
    torch.manual_seed(args.seed)
    history = []
    bce = torch.nn.BCELoss()

    for epoch in range(args.max_epochs):
        mlp_filter.train()
        running = 0.0
        batches = 0
        noisy_samples = 0
        total_samples = 0

        for batch in train_loader:
            batch = trainer.preprocess_batch(batch)
            imgs = batch["img"].to(device)
            preds = det_model(imgs)

            _, _, feats = per_sample_loss(preds, batch, return_features=True)
            if feats is None:
                continue

            # 规则打分：使用 batch 内手工归一化，避免用尚未收敛的 BN 统计量
            feats = feats.to(device)
            if feats.shape[0] < 2:
                # BN 在 batch=1 时会报错，跳过过小批次
                continue
            loss_sum = feats[:, 0] + feats[:, 1] + feats[:, 2]
            target = torch.ones_like(loss_sum)
            noisy_mask = loss_sum > args.loss_thres
            target[noisy_mask] = 0.0  # 斩立决，完全切断噪声梯度
            noisy_samples += int(noisy_mask.sum().item())
            total_samples += noisy_mask.numel()

            # 学生前向使用原始特征，由内部 BN 处理
            pred_w = mlp_filter(feats).view(-1)

            # 首次拿到有效参数后再创建优化器
            if optimizer is None:
                params = [p for p in mlp_filter.parameters() if p.requires_grad]
                if not params:
                    # 若仍未构建网络，则跳过本 batch
                    continue
                optimizer = torch.optim.Adam(params, lr=args.lr_phi)
            target = target.to(dtype=pred_w.dtype)
            loss = bce(pred_w, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            batches += 1

        epoch_loss = running / max(batches, 1)
        improved = epoch_loss + 1e-6 < best_loss
        if improved:
            best_loss = epoch_loss
            patience_left = args.patience
            torch.save(mlp_filter.state_dict(), save_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[EarlyStop] no improvement for {args.patience} epochs, stop at {epoch+1}")
                break

        history.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "best_loss": best_loss,
                "improved": improved,
                "patience_left": patience_left,
                "noisy_samples": noisy_samples,
                "total_samples": total_samples,
            }
        )
        print(
            f"[Epoch {epoch+1}] loss={epoch_loss:.6f} "
            f"(best={best_loss:.6f}, improved={improved}, patience_left={patience_left}, "
            f"noisy={noisy_samples}/{total_samples})"
        )

    print(f"✅ g_phi saved to {save_path}")
    if history:
        with log_path.open("w", newline="") as f:
            fieldnames = [
                "epoch",
                "loss",
                "best_loss",
                "improved",
                "patience_left",
                "noisy_samples",
                "total_samples",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        print(f"📄 Training log saved to {log_path}")
    else:
        print("⚠️ No training steps were recorded; log file was not written.")


if __name__ == "__main__":
    main()
