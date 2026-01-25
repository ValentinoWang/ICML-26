#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CIFAR-10 数据集封装（带可控标签噪声）

设计：
- 统一使用 torchvision.datasets.CIFAR10；
- 通过 noise_rate 在 train 集上注入对称标签噪声；
- 每个样本返回 (img, label, is_noisy) 三元组，方便后续监督训练 g_phi。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from torchvision.datasets import CIFAR10
except Exception as e:  # pragma: no cover - 环境缺少 torchvision 时给出清晰提示
    raise ImportError(
        "导入 torchvision 失败，请先安装：pip install torchvision"
    ) from e


@dataclass
class CIFAR10DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


class CIFAR10WithNoise(CIFAR10):
    """
    在标准 CIFAR-10 基础上注入对称标签噪声，并返回 is_noisy 标记。
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = True,
        transform=None,
        noise_rate: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__(root=root, train=train, download=download, transform=transform)

        import numpy as np

        self.noise_rate = float(noise_rate)
        self.rng = np.random.RandomState(seed)

        n = len(self.targets)
        self.is_noisy = np.zeros(n, dtype=bool)

        if train and self.noise_rate > 0.0:
            num_noisy = int(self.noise_rate * n)
            indices = self.rng.choice(n, size=num_noisy, replace=False)
            for idx in indices:
                orig = int(self.targets[idx])
                # 从其余类别中随机采样一个错误标签
                candidates = [c for c in range(10) if c != orig]
                new_label = int(self.rng.choice(candidates))
                self.targets[idx] = new_label
                self.is_noisy[idx] = True

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        is_noisy = bool(self.is_noisy[index]) if hasattr(self, "is_noisy") else False
        return img, target, is_noisy


def build_cifar10_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 4,
    noise_rate: float = 0.0,
    seed: int = 0,
) -> CIFAR10DataLoaders:
    """
    构建带噪声标记的 CIFAR-10 dataloader 三元组。
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_dataset = CIFAR10WithNoise(
        root=data_root,
        train=True,
        download=True,
        transform=transform_train,
        noise_rate=noise_rate,
        seed=seed,
    )
    # 简单地将官方 test split 视作验证 + 测试，这里先整体作为 val/test 共用
    # 若后续需要，可再细分。
    val_dataset = CIFAR10WithNoise(
        root=data_root,
        train=False,
        download=True,
        transform=transform_test,
        noise_rate=0.0,
        seed=seed,
    )
    test_dataset = val_dataset

    def make_loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset, shuffle=False)
    test_loader = make_loader(test_dataset, shuffle=False)

    return CIFAR10DataLoaders(train=train_loader, val=val_loader, test=test_loader)


__all__ = ["CIFAR10DataLoaders", "CIFAR10WithNoise", "build_cifar10_dataloaders"]

