#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
样本过滤器 g_phi 模块

职责：
- 接收当前 batch 的 **样本级** 统计特征（如检测损失分量、GT 统计，以及 backbone 特征）；
- 输出位于 [0, 1] 的样本权重，用于对 detection loss 做 per-sample 重加权。

当前实现（Toy 版）：
- backbone：使用 YOLO 模型的中尺度特征图 P4（stride=16），做 global average pooling 得到 (B, C)；
- 统计特征：针对每个样本再拼接 8 维统计量：
    - loss 三个分量 + GT 统计两个分量；
    - 目标类别统计 1 个分量；
    - 与锚点 θ_good 预测差异相关的 2 个分量；
- head（g_phi 头）：两层 MLP + Sigmoid，将拼接后的 (B, C+8) 特征映射到权重向量 w ∈ [0, 1]^B；
- 在损失层面实现「样本级」重加权（per-sample re-weighting）。
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.ops import xyxy2xywh

from .anchor import AnchorModel


class MLPFilter(nn.Module):
    """
    g_phi 过滤器（带特征注意力 + 软截断）：

    - 输入：形如 (B, D) 的特征向量（默认 5 维 loss/conf/iou 差异等）
    - 结构：
        1) BatchNorm1d 对特征归一化
        2) Feature Attention：学习 5 维特征的重要性并做逐维加权
        3) MLP Head：两层隐藏 + 安全感知激活 (0.1 + 0.9 * sigmoid)
    """

    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dim: int = 32,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._configured_input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net: nn.Module | None = None  # 延迟构建
        if input_dim is not None:
            self._build_net(input_dim)

    def _build_net(self, input_dim: int) -> None:
        self._configured_input_dim = input_dim
        att_hidden = max(1, input_dim // 2)
        # 注意：最后一层不做 Sigmoid，在前向中统一做 0.1 + 0.9*sigmoid
        self.bn = nn.BatchNorm1d(input_dim, affine=False).to(self.device)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, att_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(att_hidden, input_dim),
            nn.Sigmoid(),
        ).to(self.device)
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            self._build_net(x.shape[1])
        x = x.to(self.device)
        # dtype 对齐
        target_dtype = next(self.net.parameters()).dtype
        x = x.to(dtype=target_dtype)

        x_norm = self.bn(x)
        att = self.attention(x_norm)
        x_weighted = x_norm * att
        logits = self.net(x_weighted)
        weights = 0.1 + 0.9 * torch.sigmoid(logits)
        return weights

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        # 若尚未构建网络，根据 state_dict 推断输入维度
        if self.net is None:
            in_features: int | None = None
            hidden_dim: int | None = None
            for k, v in state_dict.items():
                if k.endswith("net.0.weight") and v.ndim == 2:
                    hidden_dim = v.shape[0]
                    in_features = v.shape[1]
                    break
                if k.endswith("attention.0.weight") and v.ndim == 2 and in_features is None:
                    in_features = v.shape[1]
            if in_features is not None:
                if hidden_dim is not None:
                    self.hidden_dim = int(hidden_dim)
                self._build_net(int(in_features))
        return super().load_state_dict(state_dict, strict=strict)


class PerSampleWeightedDetectionLoss(v8DetectionLoss):
    """
    在 v8DetectionLoss 基础上加入样本级过滤器 g_phi，实现 per-sample re-weighting：

    - 先按原公式拆分出每个样本 i 的「检测损失分子」：
        N_box_i, N_cls_i, N_dfl_i；
    - 使用全局分母 T = target_scores_sum（与 v8DetectionLoss 保持一致），构造特征：
        f_i = [N_box_i / T, N_cls_i / T, N_dfl_i / T]；
    - 将 f_i 输入 MLPFilter（MLP FiLTER）得到 w_i ∈ [0, 1]；
    - 最终检测损失分量为：
        L_box = (Σ_i w_i · N_box_i) / T，
        L_cls = (Σ_i w_i · N_cls_i) / T，
        L_dfl = (Σ_i w_i · N_dfl_i) / T。

    当 w_i ≡ 1 时，整体与原 v8DetectionLoss 共享同一个分母 T，仅改变了「按样本拆分后的加权求和」，从而在不破坏尺度的前提下实现样本级重加权。
    """

    def __init__(
        self,
        model,
        sample_filter: MLPFilter | None,
        anchor_model: AnchorModel | None = None,
        tal_topk: int = 10,
        filter_mode: str = "mlp",
        heuristic_keep_rate: float = 0.8,
        heuristic_down_weight: float = 0.5,
    ) -> None:
        super().__init__(model, tal_topk=tal_topk)
        self.sample_filter = sample_filter
        # 锚点模型（θ_good）用于构造“teacher-student” 风格的预测差异特征；
        # 仅在前向中以 no_grad 模式调用，不参与梯度更新。
        self.anchor_model = anchor_model
        self.filter_mode = filter_mode
        self.heuristic_keep_rate = float(heuristic_keep_rate)
        self.heuristic_down_weight = float(heuristic_down_weight)
        # 最近一个 batch 的样本权重（便于分析），形状为 (B,) 或 None
        self.last_weights: Optional[torch.Tensor] = None

    def __call__(self, preds, batch, return_features: bool = False):
        """
        计算带样本级权重的检测损失。

        返回：
            loss_vec * batch_size, loss_vec.detach()
        其中 loss_vec 为长度为 3 的张量，对应 (box, cls, dfl)。
        """
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # (h, w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # (B, H*W, 4) in stride units

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # 全局分母（与原 v8DetectionLoss 一致），用于保持 loss 尺度稳定
        # 使用 clamp(min=1.0) 避免在极端小目标场景下出现除以 0。
        target_scores_sum_global = target_scores.sum().clamp(min=1.0)

        # 预分配 per-sample 分子 N_box_i / N_cls_i / N_dfl_i
        num_box = pred_scores.new_zeros(batch_size)
        num_cls = pred_scores.new_zeros(batch_size)
        num_dfl = pred_scores.new_zeros(batch_size)

        # 1) 分类部分：BCE 分子按样本拆分
        bce_all = self.bce(pred_scores, target_scores.to(dtype))  # (B, A, C)
        num_cls = bce_all.sum(dim=(1, 2))  # (B,)

        # 2) 边框 / DFL 部分：借助原有 BboxLoss 按样本拆分分子
        for i in range(batch_size):
            fg_mask_i = fg_mask[i : i + 1]  # (1, A)
            target_scores_i = target_scores[i : i + 1]  # (1, A, C)
            target_bboxes_i = target_bboxes[i : i + 1]  # (1, A, 4)
            pred_distri_i = pred_distri[i : i + 1]  # (1, A, 4 * reg_max)
            pred_bboxes_i = pred_bboxes[i : i + 1]  # (1, A, 4)

            # 与全局分母类似，对单样本分母也做 clamp，避免数值不稳定
            target_scores_sum_i = target_scores_i.sum().clamp(min=1.0)

            if fg_mask_i.any():
                loss_box_i, loss_dfl_i = self.bbox_loss(
                    pred_distri_i,
                    pred_bboxes_i,
                    anchor_points,
                    target_bboxes_i,
                    target_scores_i,
                    target_scores_sum_i,
                    fg_mask_i,
                )
                num_box[i] = loss_box_i * target_scores_sum_i
                num_dfl[i] = loss_dfl_i * target_scores_sum_i
            else:
                num_box[i] = 0.0
                num_dfl[i] = 0.0

        # 3) 构造 per-sample 特征 z = [loss_box, loss_cls, loss_dfl, conf_diff, iou_diff]
        # 3.1 基础难度特征：loss_box / loss_cls / loss_dfl（按全局分母归一化）
        denom = target_scores_sum_global + 1e-6
        loss_box_i = num_box / denom  # (B,)
        loss_cls_i = num_cls / denom  # (B,)
        loss_dfl_i = num_dfl / denom  # (B,)
        feats_loss = torch.stack([loss_box_i, loss_cls_i, loss_dfl_i], dim=1)  # (B, 3)

        # 3.2 Anchor 差异特征：conf_diff、iou_diff
        # 当前模型的 max conf
        cur_max_conf = pred_scores.detach().sigmoid().amax(dim=(1, 2), keepdim=False)  # (B,)

        # θ_good 的 max conf 和 bbox 预测
        if self.anchor_model is not None and hasattr(self.anchor_model, "model"):
            with torch.no_grad():
                anchor_preds = self.anchor_model.model(batch["img"].to(self.device))
                a_feats = anchor_preds[1] if isinstance(anchor_preds, tuple) else anchor_preds
                a_pred_distri, a_pred_scores = torch.cat(
                    [xi.view(a_feats[0].shape[0], self.no, -1) for xi in a_feats], 2
                ).split((self.reg_max * 4, self.nc), 1)
                a_pred_scores = a_pred_scores.permute(0, 2, 1).contiguous()  # (B, A, C)
                anc_max_conf = a_pred_scores.sigmoid().amax(dim=(1, 2), keepdim=False)  # (B,)

                # Anchor 预测的 bbox（与当前模型共享 anchor_points / stride_tensor）
                a_pred_distri = a_pred_distri.permute(0, 2, 1).contiguous()
                anchor_bboxes = self.bbox_decode(anchor_points, a_pred_distri)  # (B, H*W, 4)
        else:
            anc_max_conf = pred_scores.new_zeros(batch_size)
            anchor_bboxes = pred_bboxes.detach()

        conf_diff = cur_max_conf - anc_max_conf  # (B,)

        # IoU 差异：在当前 fg_mask 上，比较 θ_new 与 θ_good 的平均 IoU
        def _mean_iou_xyxy(boxes1, boxes2):
            # boxes1, boxes2: (N, 4) in xyxy
            if boxes1.numel() == 0 or boxes2.numel() == 0:
                return boxes1.new_tensor(0.0)
            x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
            y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
            x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
            y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
            inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
            area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
            area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
            union = (area1 + area2).clamp(min=1e-6) - inter
            iou = inter / union
            return iou.mean()

        # 将预测 bbox 和 anchor bbox 统一乘以 stride_tensor，得到像素坐标
        pred_bboxes_pix = (pred_bboxes * stride_tensor).type(gt_bboxes.dtype)  # (B, A, 4)
        anchor_bboxes_pix = (anchor_bboxes * stride_tensor).type(gt_bboxes.dtype)  # (B, A, 4)

        iou_cur_list = []
        iou_good_list = []
        for i in range(batch_size):
            mask = fg_mask[i]  # (A,)
            if mask.any():
                pb = pred_bboxes_pix[i][mask]  # (M,4)
                ab = anchor_bboxes_pix[i][mask]  # (M,4)
                tb = target_bboxes[i][mask]  # (M,4)
                iou_cur = _mean_iou_xyxy(pb, tb)
                iou_good = _mean_iou_xyxy(ab, tb)
            else:
                iou_cur = pred_scores.new_tensor(0.0)
                iou_good = pred_scores.new_tensor(0.0)
            iou_cur_list.append(iou_cur)
            iou_good_list.append(iou_good)

        iou_cur = torch.stack(iou_cur_list)   # (B,)
        iou_good = torch.stack(iou_good_list) # (B,)
        iou_diff = iou_cur - iou_good         # (B,)

        feats_anchor = torch.stack([conf_diff, iou_diff], dim=1)  # (B, 2)

        # 最终 g_phi 输入特征 z: [loss_box, loss_cls, loss_dfl, conf_diff, iou_diff]
        features = torch.cat([feats_loss, feats_anchor], dim=1).detach()  # (B, 5)
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        # 3') 根据配置选择样本权重策略
        if self.sample_filter is not None and self.filter_mode == "mlp":
            # 使用可训练 MLPFilter(g_phi) 生成样本权重
            weights = self.sample_filter(features).view(-1)  # (B,)
            weights = torch.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=0.0)
            self.last_weights = weights
        else:
            # 简单的鲁棒重加权 heuristic（不使用可训练 MLP）：
            #  - 计算每个样本的标量 loss_i = (N_box_i + N_cls_i + N_dfl_i) / T；
            #  - 保留 loss 较小的样本，重权为 1，其余样本降权到 heuristic_down_weight。
            loss_i = (num_box + num_cls + num_dfl) / (target_scores_sum_global + 1e-6)  # (B,)
            keep_rate = self.heuristic_keep_rate
            keep_rate = min(max(keep_rate, 0.0), 1.0)
            k = max(int(keep_rate * batch_size), 1)
            # 选出第 k 小的 loss 作为阈值
            sorted_loss, _ = torch.sort(loss_i)
            thresh = sorted_loss[k - 1]
            keep_mask = loss_i <= thresh

            weights = pred_scores.new_ones(batch_size)
            if keep_rate < 1.0:
                weights[~keep_mask] = self.heuristic_down_weight
            self.last_weights = weights

        # 4) 使用样本权重对分子加权，再用全局分母归一化
        weighted_num_box = (weights * num_box).sum()
        weighted_num_cls = (weights * num_cls).sum()
        weighted_num_dfl = (weights * num_dfl).sum()

        denom = target_scores_sum_global + 1e-6
        loss_box = weighted_num_box / denom
        loss_cls = weighted_num_cls / denom
        loss_dfl = weighted_num_dfl / denom

        loss_vec = torch.stack([loss_box, loss_cls, loss_dfl], dim=0)

        # 与原 v8DetectionLoss 一致：乘以 batch_size，Trainer 再对三个分量求和
        out_loss = loss_vec * batch_size
        out_items = loss_vec.detach()

        if return_features:
            # 返回标准化前的 features（z）的副本，由外部根据需要再做归一化
            return out_loss, out_items, features.detach()

        return out_loss, out_items


__all__ = ["MLPFilter", "PerSampleWeightedDetectionLoss"]
