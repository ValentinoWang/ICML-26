#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-anchor rule distillation for g_phi on VOC2007 (针对混合噪声/实例级噪声):
- 数据：建议使用 box 噪声版本（单图中只有部分框被篡改）
- 特征：每个 Anchor 的 z = [loss_box, loss_cls, loss_dfl(占位0), conf_diff, iou_diff]
- 规则标签：loss_sum = loss_box + loss_cls；loss_sum > loss_thres(默认 0.3) 的 Anchor 判为噪声，
            target=bad_weight(默认 0.0)；其余 target=1.0
- 训练：仅更新 g_phi（PerAnchorFilter，内部 BN+MLP），YOLO θ 固定（使用 anchor_voc.pt），早停 patience=20
- 产物：mlp_filter_per_anchor_<anchor>.pt，日志写入 runs_per_anchor_rule_distill
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import make_anchors

# 将 Baseline 根目录加入 sys.path，复用核心模块
THIS_FILE = Path(__file__).resolve()
ICML_ROOT = THIS_FILE.parents[3]  # /root/autodl-tmp/ICML
BASELINE_ROOT = ICML_ROOT.parent
import sys

if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from ICML.core.yolo_bias_finetune.anchor import AnchorModel
from ICML.core.yolo_bias_finetune.mlp_filter import (
    MLPFilter,
    PerSampleWeightedDetectionLoss,
)


class PerAnchorFilter(nn.Module):
    """对每个 Anchor 独立打分的 MLP（无 BN，避免小批次/稀疏特征导致塌缩）。输入形状 (B*N, D)，输出 (B*N, 1)。"""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 32, device: str | torch.device | None = None):
        super().__init__()
        self.device = torch.device(device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        logits = self.net(x)
        w = torch.sigmoid(logits)
        return w


class PerAnchorFeatureExtractor(PerSampleWeightedDetectionLoss):
    """
    继承 PerSampleWeightedDetectionLoss，只用于提取 per-anchor 特征，不做样本加权。
    返回 features: (B, A, 5) = [loss_box, loss_cls, 0, conf_diff, iou_diff]
    """

    def __call__(self, preds, batch, return_features: bool = False):
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (B, A, C)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # (B, A, 4*reg_max)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # (B, A, 4) stride units

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # 分类损失 per-anchor
        bce_all = self.bce(pred_scores, target_scores.to(dtype))  # (B, A, C)
        loss_cls = bce_all.sum(dim=2)  # (B, A)

        # 盒子 IoU 损失 per-anchor（仅正样本）
        pred_bboxes_pix = (pred_bboxes * stride_tensor).type(gt_bboxes.dtype)
        loss_box = pred_scores.new_zeros(loss_cls.shape)
        for i in range(batch_size):
            mask = fg_mask[i]
            if mask.any():
                iou = bbox_iou(pred_bboxes_pix[i][mask], target_bboxes[i][mask], xywh=False, CIoU=True, eps=1e-7)
                loss_box[i, mask] = 1.0 - iou  # CIoU 损失近似

        # Anchor 差异特征（逐 Anchor）
        if self.anchor_model is not None and hasattr(self.anchor_model, "model"):
            with torch.no_grad():
                anchor_preds = self.anchor_model.model(batch["img"].to(self.device))
                a_feats = anchor_preds[1] if isinstance(anchor_preds, tuple) else anchor_preds
                a_pred_distri, a_pred_scores = torch.cat(
                    [xi.view(a_feats[0].shape[0], self.no, -1) for xi in a_feats], 2
                ).split((self.reg_max * 4, self.nc), 1)
                a_pred_scores = a_pred_scores.permute(0, 2, 1).contiguous()  # (B, A, C)
                anc_conf = a_pred_scores.sigmoid().amax(dim=2)  # (B, A)

                a_pred_distri = a_pred_distri.permute(0, 2, 1).contiguous()
                anchor_bboxes = self.bbox_decode(anchor_points, a_pred_distri)
                anchor_bboxes_pix = (anchor_bboxes * stride_tensor).type(gt_bboxes.dtype)
        else:
            anc_conf = pred_scores.new_zeros(loss_cls.shape)
            anchor_bboxes_pix = pred_bboxes.detach()

        cur_conf = pred_scores.detach().sigmoid().amax(dim=2)  # (B, A)
        conf_diff = torch.abs(cur_conf - anc_conf)

        iou_diff = pred_scores.new_zeros(loss_cls.shape)
        for i in range(batch_size):
            iou_anchor = bbox_iou(pred_bboxes_pix[i], anchor_bboxes_pix[i], xywh=False, CIoU=True, eps=1e-7)
            if iou_anchor.ndim == 2:
                iou_anchor = torch.diag(iou_anchor)  # 取同一 Anchor 的 CIoU
            iou_diff[i] = 1.0 - iou_anchor

        # 占位 dfl_loss=0，保持 5 维
        loss_dfl = torch.zeros_like(loss_cls)
        features = torch.stack([loss_box, loss_cls, loss_dfl, conf_diff, iou_diff], dim=2)  # (B, A, 5)
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)

        if return_features:
            return features

        raise RuntimeError("PerAnchorFeatureExtractor 仅用于 return_features=True 的场景")


DEFAULT_ANCHOR_DIR = (
    ICML_ROOT / "2-Mechanism Verification VOC 2007 Full" / "Results" / "Anchor" / "voc07_clean"
)
DEFAULT_DISTILL_DIR = (
    ICML_ROOT / "2-Mechanism Verification VOC 2007 Full" / "Results" / "distill_rule_per-anchor" / "voc07_noisy"
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
        "name": "gphi_per_anchor_rule_distill",
        "epochs": 1,  # dataloader 构建占位
        "batch": batch_size,
        "imgsz": 640,
        "workers": 4,
        "seed": seed,
    }
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-anchor rule distillation for g_phi on VOC2007")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/root/autodl-tmp/dataset/voc07_noisy_per-anchor/voc07_noisy.yaml"),
        help="voc07_noisy.yaml 路径（建议 per-anchor/box 噪声版本）",
    )
    parser.add_argument(
        "--anchor",
        type=Path,
        default=None,
        help="干净 VOC 训练得到的 anchor_voc_<model>.pt（默认自动在 Results/Anchor/voc07_clean 下取最新）",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="训练 batch size")
    parser.add_argument("--lr-phi", type=float, default=1e-3, help="g_phi 学习率")
    parser.add_argument("--max-epochs", type=int, default=10000, help="最大轮数（早停为主，默认 10000）")
    parser.add_argument("--patience", type=int, default=20, help="Loss 连续多少轮不下降则早停（默认 20）")
    parser.add_argument("--seed", type=int, default=1088, help="随机种子")
    parser.add_argument(
        "--loss-thres",
        type=float,
        default=0.3,
        help="per-anchor loss_sum 阈值，大于则认为是坏 Anchor",
    )
    parser.add_argument(
        "--bad-weight",
        type=float,
        default=0.0,
        help="坏 Anchor 的软标签权重（0 表示直接切断梯度）",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=500,
        help="每侧使用多少个 Top-K（高损坏样本 + 低损好样本）参与蒸馏，默认各取 500",
    )
    parser.add_argument(
        "--pos-scale",
        type=float,
        default=1.0,
        help="好样本损失缩放（平衡正负样本），默认 1",
    )
    parser.add_argument(
        "--neg-scale",
        type=float,
        default=5.0,
        help="坏样本损失缩放（放大坏样本权重），默认 5",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="保存 g_phi 权重的路径；若不指定，自动保存到 Results/distill_rule_per-anchor/voc07_noisy/mlp_filter_per_anchor_<anchor>.pt",
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
        save_path = DEFAULT_DISTILL_DIR / f"mlp_filter_per_anchor_{tag}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print("🧪 Train g_phi via per-anchor rule distillation on noisy VOC2007")
    print(f"   data: {args.data}")
    print(f"   anchor: {anchor_path}")
    print(f"   save_path: {save_path}")
    print(f"   batch_size: {args.batch_size}")
    print(f"   lr_phi: {args.lr_phi}, max_epochs: {args.max_epochs}, patience: {args.patience}")
    print(
        f"   loss_thres: {args.loss_thres}, bad_weight: {args.bad_weight}, "
        f"topk: {args.topk}, pos_scale: {args.pos_scale}, neg_scale: {args.neg_scale}"
    )

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
        "lr_phi": args.lr_phi,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "seed": args.seed,
        "loss_thres": args.loss_thres,
        "bad_weight": args.bad_weight,
        "topk": args.topk,
        "pos_scale": args.pos_scale,
        "neg_scale": args.neg_scale,
    }
    with args_path.open("w") as f:
        json.dump(meta, f, indent=2)

    device = trainer.device
    det_model = trainer.model.to(device)
    for p in det_model.parameters():
        p.requires_grad_(False)
    det_model.eval()

    anchor = AnchorModel(anchor_path, device=device)
    per_anchor_feat = PerAnchorFeatureExtractor(
        det_model,
        sample_filter=None,
        anchor_model=anchor,
    )
    per_anchor_filter = PerAnchorFilter(device=device).to(device)

    optimizer = None
    train_loader = trainer.train_loader
    if train_loader is None:
        raise RuntimeError("train_loader 为空，检查数据配置")

    best_loss = float("inf")
    patience_left = args.patience
    torch.manual_seed(args.seed)
    history = []
    bce = torch.nn.BCELoss()

    for epoch in range(args.max_epochs):
        per_anchor_filter.train()
        running = 0.0
        batches = 0
        noisy_anchors = 0
        total_anchors = 0
        used_anchors = 0
        noisy_used_anchors = 0

        for batch in train_loader:
            batch = trainer.preprocess_batch(batch)
            imgs = batch["img"].to(device)
            preds = det_model(imgs)

            features = per_anchor_feat(preds, batch, return_features=True)  # (B, A, 5)
            if features is None:
                continue

            loss_sum = features[..., 0] + features[..., 1]  # box + cls
            flat_loss = loss_sum.reshape(-1)
            total_anchors += flat_loss.numel()

            # 统计 noisy_anchor 比例（基于阈值，仅用于日志）
            noisy_mask_all = loss_sum > args.loss_thres
            noisy_anchors += int(noisy_mask_all.sum().item())

            # 双向采样：Top-K 大损（噪声，target=bad_weight）+ Top-K 小损（好样本，target=1）
            k_each = min(args.topk, flat_loss.numel())
            if k_each > 0:
                idx_hi = torch.topk(flat_loss, k=k_each, largest=True).indices
                idx_lo = torch.topk(flat_loss, k=k_each, largest=False).indices
                idx = torch.cat([idx_hi, idx_lo], dim=0)
                # 去重并打乱
                idx = torch.unique(idx)
                idx = idx[torch.randperm(idx.numel(), device=idx.device)]
            else:
                idx = torch.arange(flat_loss.numel(), device=flat_loss.device)

            flat_feats = features.reshape(-1, features.shape[-1])[idx]
            flat_target = torch.ones_like(flat_loss, device=flat_loss.device)[idx]
            # 高损样本设为 bad_weight
            high_mask = flat_loss[idx] > args.loss_thres
            flat_target[high_mask] = args.bad_weight

            used_anchors += flat_target.numel()
            noisy_used_anchors += int((flat_target < 0.999).sum().item())

            pred_w = per_anchor_filter(flat_feats).view(-1)

            if optimizer is None:
                params = [p for p in per_anchor_filter.parameters() if p.requires_grad]
                if not params:
                    continue
                optimizer = torch.optim.Adam(params, lr=args.lr_phi)

            flat_target = flat_target.to(dtype=pred_w.dtype)
            weights = torch.ones_like(flat_target)
            weights[flat_target < 0.999] = args.neg_scale
            weights[flat_target >= 0.999] = args.pos_scale
            loss = (bce(pred_w, flat_target) * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            batches += 1

        epoch_loss = running / max(batches, 1)
        epoch_loss_scaled = epoch_loss * max(used_anchors, 1)
        improved = epoch_loss + 1e-6 < best_loss
        if improved:
            best_loss = epoch_loss
            patience_left = args.patience
            torch.save(per_anchor_filter.state_dict(), save_path)
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
                "epoch_loss_scaled": epoch_loss_scaled,
                "improved": improved,
                "patience_left": patience_left,
                "noisy_anchors": noisy_anchors,
                "total_anchors": total_anchors,
                "used_anchors": used_anchors,
                "noisy_used_anchors": noisy_used_anchors,
            }
        )
        print(
            f"[Epoch {epoch+1}] loss={epoch_loss:.6f} "
            f"(best={best_loss:.6f}, improved={improved}, patience_left={patience_left}, "
            f"noisy={noisy_anchors}/{total_anchors}, "
            f"used_topk={noisy_used_anchors}/{used_anchors}, "
            f"loss_scaled~{epoch_loss_scaled:.2f})"
        )

    print(f"✅ per-anchor g_phi saved to {save_path}")
    if history:
        with log_path.open("w", newline="") as f:
            fieldnames = [
                "epoch",
                "loss",
                "epoch_loss_scaled",
                "best_loss",
                "improved",
                "patience_left",
                "noisy_anchors",
                "total_anchors",
                "used_anchors",
                "noisy_used_anchors",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        print(f"📄 Training log saved to {log_path}")
    else:
        print("⚠️ No training steps were recorded; log file was not written.")


if __name__ == "__main__":
    main()
