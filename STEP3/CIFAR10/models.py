#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CIFAR-10 模型构建

当前实现：
- 使用 torchvision.models.resnet18，并将最后一层替换为 num_classes 维全连接层。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from torchvision.models import resnet18
except Exception as e:  # pragma: no cover
    raise ImportError(
        "导入 torchvision.models 失败，请先安装：pip install torchvision"
    ) from e


def build_resnet18_cifar10(num_classes: int = 10, device: Optional[torch.device] = None) -> nn.Module:
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


__all__ = ["build_resnet18_cifar10"]

