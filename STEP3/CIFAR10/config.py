#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CIFAR-10 实验配置

职责：
- 统一管理 CIFAR-10 数据根目录、结果目录等路径；
- 提供 seeds / 训练轮数等基础超参，便于在脚本间共享。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


BASELINE_ROOT = Path(__file__).resolve().parents[2]  # .../Baseline


@dataclass
class CIFAR10Config:
    # CIFAR-10 数据下载 / 缓存根目录
    data_root: Path = BASELINE_ROOT / "dataset" / "CIFAR10"
    # 结果根目录
    results_root: Path = BASELINE_ROOT / "ICML" / "CIFAR10" / "Results"
    # 默认随机种子列表
    seeds: List[int] = field(default_factory=lambda: [1088])
    # 类别数
    num_classes: int = 10

    # 训练超参（给出一个合理默认，脚本中仍可通过 CLI 覆盖）
    epochs_theta_good: int = 50
    epochs_gphi: int = 5
    batch_size: int = 128
    num_workers: int = 4


def get_default_cifar10_config() -> CIFAR10Config:
    cfg = CIFAR10Config()
    cfg.results_root.mkdir(parents=True, exist_ok=True)
    cfg.data_root.mkdir(parents=True, exist_ok=True)
    return cfg


__all__ = ["CIFAR10Config", "get_default_cifar10_config", "BASELINE_ROOT"]

