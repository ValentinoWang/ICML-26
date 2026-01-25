"""
锚点模型（θ_good）管理模块

职责：
- 加载并冻结源域黄金模型 θ_good
- 提供计算 L_bias 的纯函数接口
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from ultralytics import YOLO


class AnchorModel:
    """
    封装 θ_good（锚点模型）的轻量类

    - 内部使用 Ultralytics YOLO 加载 best.pt
    - 只关心参数向量，不参与梯度更新
    """

    def __init__(self, checkpoint: Path, device: str | int | None = None) -> None:
        self.checkpoint = Path(checkpoint)
        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        # Ultralytics YOLO 模型
        yolo_model = YOLO(str(self.checkpoint))
        model = yolo_model.model

        # 冻结参数并移动到设备
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model.to(self._device)

        # 预先展平成一个长向量，便于计算 L2 距离
        with torch.no_grad():
            self._flat_params = self._flatten_parameters(self.model.parameters())

    @staticmethod
    def _flatten_parameters(params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
        flat = [p.detach().view(-1) for p in params if p is not None]
        if not flat:
            return torch.zeros(0)
        return torch.cat(flat)

    def compute_bias_loss(self, model: torch.nn.Module) -> torch.Tensor:
        """
        计算当前模型与锚点模型之间的 L_bias = ||θ_new - θ_good||^2

        注意：
        - self._flat_params 在初始化时已使用 no_grad 构建，不参与梯度；
        - 此处不能再使用 no_grad，否则会截断对 θ_new 的反向传播。
        """
        current_params = [p.view(-1) for p in model.parameters() if p is not None]
        if not current_params:
            return torch.tensor(0.0, device=self._device)
        flat_current = torch.cat(current_params)

        # 将锚点向量裁剪到相同长度（以防模型结构有细微差异）
        ref = self._flat_params
        min_len = min(ref.numel(), flat_current.numel())
        ref_slice = ref[:min_len]
        cur_slice = flat_current[:min_len]

        diff = cur_slice - ref_slice
        return torch.sum(diff * diff)


__all__ = ["AnchorModel"]
