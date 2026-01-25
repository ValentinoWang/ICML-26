#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在干净 CIFAR-10 上训练 θ_good（ResNet-18）。

用途：
- 作为后续 g_phi 训练的「干净教师」模型；
- 结果保存在 ICML/CIFAR10/Results/theta_good_seed_<seed>/best.pt。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from .config import get_default_cifar10_config
from .datasets import build_cifar10_dataloaders
from .models import build_resnet18_cifar10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在干净 CIFAR-10 上训练 θ_good（ResNet-18）")
    parser.add_argument("--seed", type=int, default=1088, help="随机种子")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--device", type=str, default=None, help="设备，例如 'cuda:0' 或 'cpu'")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    n_samples = 0

    for imgs, labels, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        running_loss += float(loss.detach().cpu().item()) * bs
        n_samples += bs

    return running_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0

    for imgs, labels, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())

    return correct / max(total, 1)


def main() -> None:
    args = parse_args()
    cfg = get_default_cifar10_config()

    set_seed(args.seed)

    device = torch.device(args.device) if args.device is not None else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    dls = build_cifar10_dataloaders(
        data_root=str(cfg.data_root),
        batch_size=args.batch_size,
        num_workers=cfg.num_workers,
        noise_rate=0.0,  # θ_good 使用干净标签
        seed=args.seed,
    )

    model = build_resnet18_cifar10(num_classes=cfg.num_classes, device=device)

    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

    results_root = cfg.results_root / f"theta_good_seed_{args.seed}"
    results_root.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, dls.train, optimizer, device)
        val_acc = evaluate(model, dls.val, device)
        scheduler.step()

        print(
            f"[θ_good][Epoch {epoch + 1}/{args.epochs}] "
            f"train_loss={train_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # 保存最好模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = results_root / "best.pt"
            torch.save(model.state_dict(), best_path)

    # 保存最后一轮
    last_path = results_root / "last.pt"
    torch.save(model.state_dict(), last_path)
    print(f"✅ θ_good 训练完成，best_acc={best_acc:.4f}，权重已保存到: {results_root}")


if __name__ == "__main__":
    main()

