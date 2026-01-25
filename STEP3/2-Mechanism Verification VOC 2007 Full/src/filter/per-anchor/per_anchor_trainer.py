#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-anchor 加权版 Trainer（继承 v8DetectionLoss，单次 assigner，支持 Warmup）
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss, DFLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist, make_anchors

from ICML.core.yolo_bias_finetune.anchor import AnchorModel


class PerAnchorFilter(nn.Module):
    """蒸馏同款 g_phi：BN + 两层 ReLU + Sigmoid，逐 Anchor 打分。"""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 32, device: str | torch.device | None = None):
        super().__init__()
        self.device = torch.device(device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.bn = nn.BatchNorm1d(input_dim, affine=False).to(self.device)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.bn(x)
        logits = self.net(x)
        return torch.sigmoid(logits)


class PerAnchorWeightedLoss(v8DetectionLoss):
    """
    继承 v8DetectionLoss：完全复用原版预处理/解码/assigner/bbox_loss/DFL 逻辑，
    在分子层面按 Anchor 权重加权，再用 target_scores_sum 归一化。
    支持 Warmup（current_epoch < warmup_epochs 时，权重恒 1，不做过滤）。
    """

    def __init__(
        self,
        model,
        anchor_model: AnchorModel,
        filter_net: PerAnchorFilter,
        lambda_bias: float = 0.0,
        loss_thres: float = 0.3,
        bad_weight: float = 0.0,
        topk: int = 500,
        pos_scale: float = 1.0,
        neg_scale: float = 1.0,
        warmup_epochs: int = 0,
    ) -> None:
        super().__init__(model)
        self.anchor_model = anchor_model
        self.filter_net = filter_net
        self.lambda_bias = float(lambda_bias)
        self.loss_thres = loss_thres
        self.bad_weight = bad_weight
        self.topk = topk
        self.pos_scale = pos_scale
        self.neg_scale = neg_scale
        self.warmup_epochs = int(max(0, warmup_epochs))
        self.current_epoch: int = 0
        self.dfl_loss = DFLoss(self.reg_max) if self.reg_max > 1 else None
        self._debug_count = 0

    def _decode_anchor_preds(self, batch_img: torch.Tensor):
        """解析 anchor 模型预测，返回 anc_conf (B,A) 与 anchor_bboxes (B,A,4) stride units。"""
        device_type = "cuda" if self.device.type != "cpu" else "cpu"
        with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
            anc_preds = self.anchor_model.model(batch_img.to(self.device).float())
            a_feats = anc_preds[1] if isinstance(anc_preds, tuple) else anc_preds
            a_pred_distri, a_pred_scores = torch.cat(
                [xi.view(a_feats[0].shape[0], self.no, -1) for xi in a_feats], 2
            ).split((self.reg_max * 4, self.nc), 1)
            a_pred_scores = a_pred_scores.permute(0, 2, 1).contiguous()  # (B, A, C)
            anc_conf = a_pred_scores.sigmoid().amax(dim=2)  # (B, A)

            a_pred_distri = a_pred_distri.permute(0, 2, 1).contiguous()
            anchor_points, stride_tensor = make_anchors(a_feats, self.stride, 0.5)
            anchor_bboxes = self.bbox_decode(anchor_points, a_pred_distri)  # stride units
            return anc_conf, anchor_bboxes

    def __call__(self, preds, batch):
        """完全复用父类流程，分子处注入 per-anchor 权重，分母仍用 target_scores_sum。"""
        if self._debug_count < 3:
            msg_enter = f"[g_phi debug] enter __call__: epoch={self.current_epoch}"
            LOGGER.info(msg_enter)
            print(msg_enter, flush=True)

        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets（与 v8DetectionLoss 一致）
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (B, A, 4)

        # Assigner 一次
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # ---------------- 原版分子计算（未归一化） ----------------
        # Cls
        loss_cls_raw = self.bce(pred_scores, target_scores.to(dtype)).sum(dim=-1)  # (B, A)

        loss_box_raw = torch.zeros_like(loss_cls_raw)
        loss_dfl_raw = torch.zeros_like(loss_cls_raw)
        if fg_mask.any():
            # 按标准 v8DetectionLoss：(1 - iou) * weight，再由 target_scores_sum 归一化
            target_bboxes /= stride_tensor
            weight = target_scores.sum(-1)  # (B, A)

            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True, eps=1e-7)
            if iou.ndim > 1:
                iou = iou.diag()  # 逐 Anchor 对应的 IoU，避免 NxN 矩阵形状不匹配
            loss_box_raw[fg_mask] = ((1.0 - iou) * weight[fg_mask]).to(loss_box_raw.dtype)

            if self.dfl_loss is not None:
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
                pred_dist_fg = pred_distri[fg_mask].view(-1, self.reg_max)
                target_fg = target_ltrb[fg_mask]
                loss_dfl_fg = self.dfl_loss(pred_dist_fg, target_fg).view(-1) * weight[fg_mask]
                loss_dfl_raw[fg_mask] = loss_dfl_fg.to(loss_dfl_raw.dtype)

        # ---------------- 构造特征 / 权重（仅分子加权，分母不变） ----------------
        use_rule = self.current_epoch >= self.warmup_epochs
        if use_rule:
            # Teacher diff
            conf_diff = torch.zeros_like(loss_cls_raw)
            iou_diff = torch.zeros_like(loss_cls_raw)
            if self.anchor_model is not None and hasattr(self.anchor_model, "model"):
                anc_conf, anchor_bboxes = self._decode_anchor_preds(batch["img"])
                cur_conf = pred_scores.detach().sigmoid().amax(dim=2)
                conf_diff = torch.abs(cur_conf - anc_conf)
                for i in range(batch_size):
                    iou_anchor = bbox_iou(pred_bboxes[i], anchor_bboxes[i], xywh=False, CIoU=True, eps=1e-7)
                    iou_anchor_diag = torch.diag(iou_anchor) if iou_anchor.ndim == 2 else torch.zeros_like(conf_diff[i])
                    iou_diff[i] = 1.0 - iou_anchor_diag

            features = torch.stack([loss_box_raw, loss_cls_raw, loss_dfl_raw, conf_diff, iou_diff], dim=2)
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)

            # 仅在前景 Anchor 上进行规则判定与加权，背景保持权重 1
            fg_feats = features[fg_mask]  # (N_pos, 5)
            fg_loss_sum = (loss_box_raw + loss_cls_raw + loss_dfl_raw)[fg_mask]

            # Top-K 只在前景上取，避免把所有前景都过滤掉
            k = min(self.topk, fg_loss_sum.numel()) if fg_loss_sum.numel() > 0 else 0
            if k > 0 and k < fg_loss_sum.numel():
                thresh = torch.topk(fg_loss_sum, k=k, largest=True).values[-1]
                mask_topk_pos = fg_loss_sum >= thresh
            else:
                mask_topk_pos = torch.ones_like(fg_loss_sum, dtype=torch.bool)

            target_weight_label = torch.ones_like(fg_loss_sum)
            noisy_mask = fg_loss_sum > self.loss_thres
            target_weight_label[noisy_mask] = self.bad_weight

            pred_weights_pos = torch.ones_like(fg_loss_sum)
            if fg_feats.numel() > 0:
                pred_weights_pos = self.filter_net(fg_feats).view(-1)
                pred_weights_pos = torch.nan_to_num(pred_weights_pos, nan=1.0, posinf=1.0, neginf=0.0)

            scale_pos = torch.ones_like(pred_weights_pos)
            pos_mask_rule = target_weight_label >= 0.999
            scale_pos[pos_mask_rule] = self.pos_scale
            scale_pos[~pos_mask_rule] = self.neg_scale

            final_weights = torch.ones_like(loss_cls_raw)
            if fg_feats.numel() > 0:
                pos_weights = pred_weights_pos * scale_pos * mask_topk_pos.float()
                if pos_weights.mean().item() < 1e-6:
                    LOGGER.warning("g_phi weights on FG ~0; clamp to min_eps")
                pos_weights = torch.clamp(pos_weights, min=1e-3)
                final_weights[fg_mask] = pos_weights.to(final_weights.dtype)
        else:
            # Warmup：权重恒 1，不做过滤，保持与 baseline 数值一致
            final_weights = torch.ones_like(loss_cls_raw)

        if self._debug_count < 3:  # 打印前 3 个 forward 的权重统计，避免被 tqdm 吞掉
            with torch.no_grad():
                mean_w = final_weights.mean().item()
                min_w = final_weights.min().item()
                max_w = final_weights.max().item()
                frac_low = (final_weights < 0.5).float().mean().item()
                frac_zero = (final_weights < 1e-3).float().mean().item()
                msg = (
                    f"[g_phi debug] epoch={self.current_epoch} mean_w={mean_w:.4f} "
                    f"min={min_w:.4f} max={max_w:.4f} frac_w<0.5={frac_low:.4f} frac_w~0={frac_zero:.4f}"
                )
                LOGGER.info(msg)
                print(msg, flush=True)
            self._debug_count += 1

        # ---------------- 归一化与超参缩放（对齐原版） ----------------
        loss[0] = (loss_box_raw * final_weights).sum() / target_scores_sum * self.hyp.box
        loss[1] = (loss_cls_raw * final_weights).sum() / target_scores_sum * self.hyp.cls
        loss[2] = (loss_dfl_raw * final_weights).sum() / target_scores_sum * self.hyp.dfl

        total_loss = loss.sum() * batch_size

        if self.lambda_bias > 0 and hasattr(self.anchor_model, "compute_bias_loss"):
            bias_loss = self.anchor_model.compute_bias_loss(self.model)
            total_loss = total_loss + self.lambda_bias * bias_loss

        return total_loss, loss.detach()


class PerAnchorBiasDetectionTrainer(DetectionTrainer):
    """使用 PerAnchorWeightedLoss 的 Trainer。"""

    def __init__(
        self,
        anchor_model: AnchorModel,
        lambda_bias: float,
        mlpfilter_path: Path,
        loss_thres: float = 0.3,
        bad_weight: float = 0.0,
        topk: int = 500,
        pos_scale: float = 1.0,
        neg_scale: float = 5.0,
        warmup_epochs: int = 0,
        overrides: Dict[str, Any] | None = None,
    ) -> None:
        self.anchor_model = anchor_model
        self.lambda_bias = float(lambda_bias)
        self.mlpfilter_path = mlpfilter_path
        self.loss_thres = loss_thres
        self.bad_weight = bad_weight
        self.topk = topk
        self.pos_scale = pos_scale
        self.neg_scale = neg_scale
        self.warmup_epochs = warmup_epochs
        super().__init__(overrides=overrides)

    def get_model(self, cfg=None, weights=None, verbose=True):
        return super().get_model(cfg, weights, verbose)

    def init_criterion(self):
        # 加载冻结的 g_phi，并生成自定义 Loss
        loss_fn = self._make_loss_fn()
        self.criterion = loss_fn
        self.model.criterion = loss_fn
        LOGGER.info("Using PerAnchorWeightedLoss (per-anchor filtering enabled).")
        return loss_fn

    def _make_loss_fn(self) -> PerAnchorWeightedLoss:
        """构建自定义 per-anchor Loss（加载 g_phi + anchor 模型，上到正确设备）。"""
        filter_net = PerAnchorFilter(device=self.device)
        state = torch.load(self.mlpfilter_path, map_location=self.device)
        filter_net.load_state_dict(state, strict=False)
        # 有些旧权重只包含 BN 统计且 running_var 极小，导致输出全 0；兜底重置/钳制 BN 统计
        if hasattr(filter_net, "bn"):
            with torch.no_grad():
                if hasattr(filter_net.bn, "running_mean"):
                    filter_net.bn.running_mean.zero_()
                if hasattr(filter_net.bn, "running_var"):
                    filter_net.bn.running_var.clamp_(min=1e-3)
        filter_net.eval()
        filter_net.requires_grad_(False)

        if self.anchor_model is not None and hasattr(self.anchor_model, "model"):
            self.anchor_model.model.to(self.device)
            self.anchor_model.model.eval()

        loss_fn = PerAnchorWeightedLoss(
            model=self.model,
            anchor_model=self.anchor_model,
            filter_net=filter_net,
            lambda_bias=self.lambda_bias,
            loss_thres=self.loss_thres,
            bad_weight=self.bad_weight,
            topk=self.topk,
            pos_scale=self.pos_scale,
            neg_scale=self.neg_scale,
            warmup_epochs=self.warmup_epochs,
        )
        return loss_fn

    def train_step(self, batch):
        if hasattr(self, "criterion") and hasattr(self.criterion, "current_epoch"):
            self.criterion.current_epoch = getattr(self, "epoch", 0)
        return super().train_step(batch)

    def set_model_attributes(self):
        """在基础属性设置完成后，再挂载自定义 Loss（此时 model.args 已就绪）。"""
        super().set_model_attributes()
        loss_fn = self._make_loss_fn()
        self.model.criterion = loss_fn
        self.criterion = loss_fn
        LOGGER.info("Attached PerAnchorWeightedLoss to model.criterion (post set_model_attributes)")

    def setup_model(self):
        """加载模型，Loss 延后到 set_model_attributes 中挂载。"""
        return super().setup_model()


__all__ = [
    "PerAnchorFilter",
    "PerAnchorWeightedLoss",
    "PerAnchorBiasDetectionTrainer",
]
