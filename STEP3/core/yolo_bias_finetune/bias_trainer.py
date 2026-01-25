"""
基于 Ultralytics DetectionTrainer 的偏置控制训练器
"""

from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer

from .anchor import AnchorModel
from .mlp_filter import MLPFilter, PerSampleWeightedDetectionLoss


class BiasWrappedModel(nn.Module):
    """
    轻量包装 DetectionModel：
    - 保留 DetectionModel 的关键属性（stride、names、nc 等），避免破坏 Ultralytics Trainer 假设；
    - 在训练前向中使用 YOLO 检测损失 + L_bias；
    - 可选：在训练损失内部通过 MLP FiLTER (g_phi) 做 **样本级重加权**；
    - 验证 / 推理路径保持与原 YOLO 一致，不改动。
    """

    def __init__(
        self,
        inner: nn.Module,
        anchor: AnchorModel,
        lambda_bias: float,
        use_sample_filter: bool = False,
        filter_mode: str = "mlp",
    ) -> None:
        super().__init__()
        self.inner = inner
        self.anchor = anchor
        self.lambda_bias = lambda_bias
        self.use_sample_filter = bool(use_sample_filter)
        self.filter_mode = filter_mode

        # 样本过滤器 g_phi（MLP FiLTER）：将每个样本的检测损失统计映射到权重 w_i ∈ [0, 1]
        # 注意：per-sample loss 需要在 inner.args 就绪后才能构造，因此这里只创建过滤器本身，
        # PerSampleWeightedDetectionLoss 的实例会在 BiasDetectionTrainer.set_model_attributes 中延迟初始化。
        if self.use_sample_filter and filter_mode == "mlp":
            self.sample_filter: MLPFilter | None = MLPFilter()
        else:
            self.sample_filter = None
        self.per_sample_loss: PerSampleWeightedDetectionLoss | None = None

        # 代理 DetectionModel 的关键属性
        for attr in ("stride", "names", "nc", "yaml"):
            if hasattr(inner, attr):
                setattr(self, attr, getattr(inner, attr))
        # 代理 loss 方法，供 Validator 调用 model.loss(...) 计算 val loss（使用原始 YOLO 检测损失即可）
        if hasattr(inner, "loss"):
            self.loss = inner.loss

    def forward(self, *args, **kwargs):
        """
        训练模式:
            输入 batch(dict)，显式使用 YOLO 检测损失（可选样本级加权） + L_bias。

        验证 / 推理模式:
            输入 image tensor 等，直接透传到内部 DetectionModel，保持原 YOLO 行为。
        """
        # 训练阶段：DetectionTrainer 会调用 self.model(batch_dict)
        if self.training and args and isinstance(args[0], dict):
            batch = args[0]
            img = batch["img"]

            # 使用内部 YOLO DetectionModel 做前向预测（与原始训练流程一致）
            preds = self.inner(img)

            # 检测损失：有 g_phi 时优先使用 PerSampleWeightedDetectionLoss，
            # 如遇到数值异常（NaN/Inf）则回退到原始 v8DetectionLoss，避免训练崩溃。
            if self.per_sample_loss is not None:
                loss_vec, loss_items = self.per_sample_loss(preds, batch)
                if not torch.isfinite(loss_vec).all():
                    # 回退到原始损失，并禁用本 batch 的样本过滤器
                    if getattr(self.inner, "criterion", None) is None:
                        self.inner.criterion = self.inner.init_criterion()
                    loss_vec, loss_items = self.inner.criterion(preds, batch)
            else:
                if getattr(self.inner, "criterion", None) is None:
                    self.inner.criterion = self.inner.init_criterion()
                loss_vec, loss_items = self.inner.criterion(preds, batch)

            detection_loss = loss_vec.sum()

            # L_bias：约束当前模型与锚点模型 θ_good 之间的距离
            bias_loss = self.anchor.compute_bias_loss(self.inner)
            total_loss = detection_loss + self.lambda_bias * bias_loss
            return total_loss, loss_items

        # 验证 / 推理：保持原 YOLO 模型行为
        return self.inner(*args, **kwargs)


class BiasDetectionTrainer(DetectionTrainer):
    """在 Ultralytics DetectionTrainer 基础上叠加 L_bias（可选 g_phi）的版本。"""

    def __init__(
        self,
        anchor_model: AnchorModel,
        lambda_bias: float,
        use_sample_filter: bool = False,
        filter_mode: str = "mlp",
        mlpfilter_init_path: Path | None = None,
        overrides: Dict[str, Any] | None = None,
    ) -> None:
        self.anchor_model = anchor_model
        self.lambda_bias = float(lambda_bias)
        self.use_sample_filter = bool(use_sample_filter)
        self.filter_mode = filter_mode
        # 若提供了 meta 阶段训练得到的 g_phi 参数路径，则在 set_model_attributes 中加载
        self.mlpfilter_init_path = mlpfilter_init_path
        super().__init__(overrides=overrides)

    def set_model_attributes(self):
        """
        在父类基础上设置模型属性，并同步到内部 DetectionModel。

        DetectionTrainer 会在 self.model 上设置 nc/names/args 等属性；
        这里将这些关键信息同步给 inner DetectionModel，避免后续损失构建时
        访问 self.args 等属性出错。
        """
        super().set_model_attributes()

        if hasattr(self.model, "inner"):
            inner = self.model.inner
            # 同步类别数和名称
            for attr in ("nc", "names"):
                if hasattr(self.model, attr):
                    setattr(inner, attr, getattr(self.model, attr))
            # 同步超参数 args（loss 初始化依赖）
            if hasattr(self, "args"):
                inner.args = self.args

            # 若启用了样本过滤器 g_phi，在此处初始化 PerSampleWeightedDetectionLoss，
            # 此时 inner.args 等属性已经就绪，避免在 BiasWrappedModel.__init__ 早期访问不到 args。
            if getattr(self.model, "per_sample_loss", None) is None and self.use_sample_filter:
                # 若存在 meta 阶段训练得到的 g_phi 参数，则优先加载，作为 MLP FiLTER 的初始化
                if (
                    getattr(self.model, "sample_filter", None) is not None
                    and self.mlpfilter_init_path is not None
                    and self.mlpfilter_init_path.exists()
                ):
                    # DetectionModel 本身没有 device 属性，这里用 Trainer 维护的 device 信息
                    device = getattr(self, "device", "cpu")
                    state = torch.load(self.mlpfilter_init_path, map_location=device)
                    try:
                        self.model.sample_filter.load_state_dict(state, strict=False)
                    except Exception:
                        # 若 state_dict 结构不完全匹配，忽略加载错误，退回随机初始化
                        pass

                self.model.per_sample_loss = PerSampleWeightedDetectionLoss(
                    inner,
                    self.model.sample_filter,
                    anchor_model=self.anchor_model,
                    filter_mode=getattr(self.model, "filter_mode", "mlp"),
                )

            # 再次确保整个模型（包括新挂载的 sample_filter / per_sample_loss）统一在 Trainer.device 上
            # 避免 EMA 在更新时遇到 CPU/CUDA 混合参数。
            self.model.to(self.device)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """构建模型时，将原始 DetectionModel 包装为一个带 L_bias 的模型。"""
        base_model = super().get_model(cfg, weights, verbose)
        return BiasWrappedModel(
            inner=base_model,
            anchor=self.anchor_model,
            lambda_bias=self.lambda_bias,
            use_sample_filter=self.use_sample_filter,
            filter_mode=self.filter_mode,
        )


__all__ = ["BiasDetectionTrainer", "BiasWrappedModel"]
