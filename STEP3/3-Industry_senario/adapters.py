#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
长时 TTA 适配器（基于 YOLO 检测输出的分类分支）。

四种策略：
- SelfTrainingAdapter: 伪标签自训练（仅用分类分支）
- TentAdapter: 熵最小化 (TENT)
- EataLiteAdapter: 熵阈值筛选 + L2 正则（简化版 EATA/EWC）
- BiasOnlyAdapter: 伪标签自训练 + L_bias（到初始权重的 L2）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StepResult:
    loss: float
    metrics: Dict[str, Any]


class BaseAdapter:
    """适配器基类，负责优化器、锚点参数等通用逻辑。"""

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        device: torch.device,
        name: str,
        weight_decay: float = 0.0,
        optimizer: str = "adamw",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()  # 使用推理输出格式（便于解析），仍保留梯度
        self.device = device
        self.name = name
        params = [p for p in self.model.parameters() if p.requires_grad]
        if optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.initial_params = self._snapshot_params()

    def _snapshot_params(self) -> Tuple[torch.Tensor, ...]:
        with torch.no_grad():
            return tuple(p.detach().clone().to(self.device) for p in self.model.parameters() if p.requires_grad)

    def _l2_to_initial(self) -> torch.Tensor:
        if not self.initial_params:
            return torch.tensor(0.0, device=self.device)
        sq_sum = torch.tensor(0.0, device=self.device)
        for p, init in zip(self.model.parameters(), self.initial_params):
            if not p.requires_grad:
                continue
            diff = p - init
            sq_sum += torch.sum(diff * diff)
        return sq_sum

    def train_batch(self, batch: torch.Tensor) -> StepResult:  # pragma: no cover - 接口定义
        raise NotImplementedError


def _split_yolo_outputs(preds: torch.Tensor | Tuple[torch.Tensor, ...] | list) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 YOLO 检测输出拆分为 (cls_logits, obj_logit)。
    preds: [B, N, 5+nc] 或相似形状。
    """
    if isinstance(preds, (tuple, list)):
        first = preds[0]
        # 训练模式下可能返回 list[Tensor]，每个 [B, C, H, W]
        if isinstance(first, torch.Tensor) and first.dim() == 4:
            stacked = []
            for p in preds:  # type: ignore[assignment]
                stacked.append(p.view(p.shape[0], p.shape[1], -1))  # [B, C, HW]
            preds = torch.cat(stacked, dim=2).permute(0, 2, 1)  # [B, N, C]
        else:
            preds = first
    if preds.dim() == 3:
        # 若末维较小（通常为 no），认为形状已是 [B, N, no]；若末维较大则视为 [B, no, N] 需转置
        if preds.shape[-1] > preds.shape[-2]:
            preds = preds.permute(0, 2, 1)  # [B, no, N] -> [B, N, no]
    cls_logits = preds[..., 5:]
    obj_logit = preds[..., 4]
    return cls_logits, obj_logit


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return (-probs * torch.log(probs.clamp(min=1e-9))).sum(dim=-1)


class SelfTrainingAdapter(BaseAdapter):
    """仅使用伪标签进行自我更新的 Baseline。"""

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        device: torch.device,
        pseudo_threshold: float = 0.6,
        optimizer: str = "adamw",
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(model, lr, device, name="self-training", weight_decay=weight_decay, optimizer=optimizer)
        self.pseudo_threshold = pseudo_threshold

    def train_batch(self, batch) -> StepResult:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x_weak, x_strong = batch[0].to(self.device), batch[1].to(self.device)
        else:
            x_weak = x_strong = batch.to(self.device)

        # 伪标签来自 weak
        with torch.no_grad():
            self.model.eval()
            preds_w = self.model(x_weak)
            cls_w, obj_w = _split_yolo_outputs(preds_w)
            obj_prob = torch.sigmoid(obj_w)
            conf, pseudo = torch.softmax(cls_w, dim=-1).max(dim=-1)
            mask = obj_prob >= self.pseudo_threshold

        self.model.train()
        preds_s = self.model(x_strong)
        cls_s, _ = _split_yolo_outputs(preds_s)

        if mask.any():
            loss = F.cross_entropy(cls_s[mask], pseudo[mask])
            accept = int(mask.sum().item())
        else:
            loss = cls_s.sum() * 0.0
            accept = 0

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        param_distance = self._l2_to_initial()
        return StepResult(
            loss=float(loss.item()),
            metrics={
                "pseudo_accept": accept,
                "anchors_total": int(obj_prob.numel()),
                "mean_confidence": float(conf.mean().item()),
                "param_distance": float(param_distance.item()),
                "train_cls_loss": float(loss.item()),
            },
        )


class TentAdapter(BaseAdapter):
    """TENT: 仅最小化预测熵。"""

    def __init__(self, model: nn.Module, lr: float, device: torch.device, optimizer: str = "adamw", weight_decay: float = 0.0) -> None:
        super().__init__(model, lr, device, name="tent", weight_decay=weight_decay, optimizer=optimizer)

    def train_batch(self, batch: torch.Tensor) -> StepResult:
        self.model.train()
        preds = self.model(batch.to(self.device))
        cls_logits, _ = _split_yolo_outputs(preds)
        loss = _entropy_from_logits(cls_logits).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ent = _entropy_from_logits(cls_logits)
        param_distance = self._l2_to_initial()

        return StepResult(
            loss=float(loss.item()),
            metrics={
                "mean_entropy": float(ent.mean().item()),
                "max_entropy": float(ent.max().item()),
                "param_distance": float(param_distance.item()),
                "train_entropy_loss": float(loss.item()),
            },
        )


class EataLiteAdapter(BaseAdapter):
    """
    EATA 简化版：熵低样本参与更新 + L2 正则（模拟 EWC）。
    - 只用低熵样本 (entropy <= threshold) 计算熵损失；
    - 额外加入对初始参数的 L2 约束，避免漂移。
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        device: torch.device,
        entropy_threshold: float = 1.5,
        lambda_l2: float = 1e-3,
        optimizer: str = "adamw",
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(model, lr, device, name="eata-lite", weight_decay=weight_decay, optimizer=optimizer)
        self.entropy_threshold = entropy_threshold
        self.lambda_l2 = lambda_l2

    def train_batch(self, batch: torch.Tensor) -> StepResult:
        self.model.train()
        preds = self.model(batch.to(self.device))
        cls_logits, _ = _split_yolo_outputs(preds)
        ent = _entropy_from_logits(cls_logits)
        mask = ent <= self.entropy_threshold

        if mask.any():
            ent_loss = ent[mask].mean()
            used = int(mask.sum().item())
        else:
            ent_loss = ent.mean()
            used = 0

        l2_penalty = self._l2_to_initial()
        loss = ent_loss + self.lambda_l2 * l2_penalty

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        param_distance = self._l2_to_initial()
        return StepResult(
            loss=float(loss.item()),
            metrics={
                "used_low_entropy_anchors": used,
                "mean_entropy": float(ent.mean().item()),
                "l2_penalty": float(l2_penalty.item()),
                "param_distance": float(param_distance.item()),
                "train_entropy_loss": float(ent_loss.item()),
                "train_l2_penalty": float(l2_penalty.item()),
            },
        )


class BiasOnlyAdapter(BaseAdapter):
    """Ours: 伪标签自训练 + L_bias 锚定（L2 到初始权重）。"""

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        device: torch.device,
        pseudo_threshold: float = 0.6,
        lambda_bias: float = 1e-4,
        lambda_sensitivity: float = 5.0,
        optimizer: str = "adamw",
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(model, lr, device, name="bias-only", weight_decay=weight_decay, optimizer=optimizer)
        self.pseudo_threshold = pseudo_threshold
        self.lambda_bias = lambda_bias
        self.lambda_sensitivity = lambda_sensitivity

    def train_batch(self, batch) -> StepResult:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x_weak, x_strong = batch[0].to(self.device), batch[1].to(self.device)
        else:
            x_weak = x_strong = batch.to(self.device)

        # 伪标签来自 weak
        with torch.no_grad():
            self.model.eval()
            preds_w = self.model(x_weak)
            cls_w, obj_w = _split_yolo_outputs(preds_w)
            obj_prob = torch.sigmoid(obj_w)
            conf, pseudo = torch.softmax(cls_w, dim=-1).max(dim=-1)
            mask = obj_prob >= self.pseudo_threshold

        self.model.train()
        preds_s = self.model(x_strong)
        cls_s, _ = _split_yolo_outputs(preds_s)

        if mask.any():
            ce_loss = F.cross_entropy(cls_s[mask], pseudo[mask])
            accept = int(mask.sum().item())
        else:
            ce_loss = cls_s.sum() * 0.0
            accept = 0

        bias_loss = self._l2_to_initial()
        mean_conf = float(conf.mean().item())
        lambda_dyn = self.lambda_bias * (1.0 + self.lambda_sensitivity * (1.0 - mean_conf))
        loss = ce_loss + lambda_dyn * bias_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        param_distance = self._l2_to_initial()
        return StepResult(
            loss=float(loss.item()),
            metrics={
                "pseudo_accept": accept,
                "mean_confidence": float(conf.mean().item()),
                "bias_loss": float(bias_loss.item()),
                "lambda_dyn": float(lambda_dyn),
                "param_distance": float(param_distance.item()),
                "train_cls_loss": float(ce_loss.item()),
                "train_bias_loss": float(bias_loss.item()),
            },
        )


class CoTTAAdapter(BaseAdapter):
    """
    简化版 CoTTA：
    - 维护 EMA 教师模型提供伪标签
    - 学生以教师伪标签做 CE，并与教师做 KL 一致性
    - 每步后用 EMA 更新教师
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        device: torch.device,
        pseudo_threshold: float = 0.6,
        ema_alpha: float = 0.999,
        kl_weight: float = 1.0,
        optimizer: str = "adamw",
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(model, lr, device, name="cotta", weight_decay=weight_decay, optimizer=optimizer)
        self.pseudo_threshold = pseudo_threshold
        self.ema_alpha = ema_alpha
        self.kl_weight = kl_weight
        # 构造教师 EMA
        self.teacher = copy.deepcopy(self.model).to(device)
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _update_teacher(self) -> None:
        for ts, ss in zip(self.teacher.parameters(), self.model.parameters()):
            ts.mul_(self.ema_alpha).add_(ss, alpha=1.0 - self.ema_alpha)

    def train_batch(self, batch) -> StepResult:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x_weak, x_strong = batch[0].to(self.device), batch[1].to(self.device)
        else:
            x_weak = x_strong = batch.to(self.device)

        # 教师前向（无梯度）
        self.teacher.train()
        with torch.no_grad():
            t_preds = self.teacher(x_weak)
            t_cls, t_obj = _split_yolo_outputs(t_preds)
            t_obj_prob = torch.sigmoid(t_obj)
            t_conf, t_pseudo = torch.softmax(t_cls, dim=-1).max(dim=-1)
            t_mask = t_obj_prob >= self.pseudo_threshold

        # 学生前向
        self.model.train()
        s_preds = self.model(x_strong)
        s_cls, _ = _split_yolo_outputs(s_preds)

        if t_mask.any():
            ce_loss = F.cross_entropy(s_cls[t_mask], t_pseudo[t_mask])
            accept = int(t_mask.sum().item())
        else:
            ce_loss = s_cls.sum() * 0.0
            accept = 0

        # KL 一致性（全体）
        logp_s = torch.log_softmax(s_cls, dim=-1)
        p_t = torch.softmax(t_cls, dim=-1)
        kl_loss = F.kl_div(logp_s, p_t, reduction="batchmean")

        loss = ce_loss + self.kl_weight * kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # EMA 更新教师
        self._update_teacher()

        param_distance = self._l2_to_initial()
        return StepResult(
            loss=float(loss.item()),
            metrics={
                "pseudo_accept": accept,
                "mean_confidence_teacher": float(t_conf.mean().item()),
                "kl_loss": float(kl_loss.item()),
                "param_distance": float(param_distance.item()),
                "train_cls_loss": float(ce_loss.item()),
            },
        )


class StabilizedCoTTAAdapter(CoTTAAdapter):
    """
    Stabilized-CoTTA: 在 CoTTA 的基础上加入 L_bias 锚定（到初始参数）。
    损失 = CE(teacher pseudo) + kl_weight*KL(student||teacher) + lambda_bias*||theta - theta0||^2
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        device: torch.device,
        pseudo_threshold: float = 0.6,
        ema_alpha: float = 0.999,
        kl_weight: float = 1.0,
        lambda_bias: float = 1e-4,
        lambda_sensitivity: float = 5.0,
        optimizer: str = "adamw",
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(
            model,
            lr=lr,
            device=device,
            pseudo_threshold=pseudo_threshold,
            ema_alpha=ema_alpha,
            kl_weight=kl_weight,
            optimizer=optimizer,
            weight_decay=weight_decay,
        )
        self.lambda_bias = lambda_bias
        self.lambda_sensitivity = lambda_sensitivity

    def train_batch(self, batch) -> StepResult:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x_weak, x_strong = batch[0].to(self.device), batch[1].to(self.device)
        else:
            x_weak = x_strong = batch.to(self.device)

        # 教师前向（无梯度）
        self.teacher.train()
        with torch.no_grad():
            t_preds = self.teacher(x_weak)
            t_cls, t_obj = _split_yolo_outputs(t_preds)
            t_obj_prob = torch.sigmoid(t_obj)
            t_conf, t_pseudo = torch.softmax(t_cls, dim=-1).max(dim=-1)
            t_mask = t_obj_prob >= self.pseudo_threshold

        # 学生前向
        self.model.train()
        s_preds = self.model(x_strong)
        s_cls, _ = _split_yolo_outputs(s_preds)

        if t_mask.any():
            ce_loss = F.cross_entropy(s_cls[t_mask], t_pseudo[t_mask])
            accept = int(t_mask.sum().item())
        else:
            ce_loss = s_cls.sum() * 0.0
            accept = 0

        # KL 一致性
        logp_s = torch.log_softmax(s_cls, dim=-1)
        p_t = torch.softmax(t_cls, dim=-1)
        kl_loss = F.kl_div(logp_s, p_t, reduction="batchmean")

        # L_bias 锚定
        bias_loss = self._l2_to_initial()

        # 自适应 λ：根据 weak 伪标签的平均置信度调整锚定力度
        lambda_dyn = self.lambda_bias * (1.0 + getattr(self, "lambda_sensitivity", 0.0) * (1.0 - float(t_conf.mean().item())))

        loss = ce_loss + self.kl_weight * kl_loss + lambda_dyn * bias_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # EMA 更新教师
        self._update_teacher()

        return StepResult(
            loss=float(loss.item()),
            metrics={
                "pseudo_accept": accept,
                "mean_confidence_teacher": float(t_conf.mean().item()),
                "kl_loss": float(kl_loss.item()),
                "bias_loss": float(bias_loss.item()),
                "lambda_dyn": float(lambda_dyn),
            },
        )

__all__ = [
    "SelfTrainingAdapter",
    "TentAdapter",
    "EataLiteAdapter",
    "BiasOnlyAdapter",
    "CoTTAAdapter",
    "StabilizedCoTTAAdapter",
    "StepResult",
]
