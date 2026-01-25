#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在带人工标签噪声的 CIFAR-10 上训练 g_phi（MLPFilter），作为“噪声检测器”。

设计思路：
- θ_good：在干净 CIFAR-10 上训练好的 ResNet-18（参考 train_theta_good_cifar10.py）；
- 数据：在 train 集上注入对称标签噪声（noise_rate），并返回 is_noisy 标记；
- 特征 z：
    - loss_cls: per-sample CrossEntropy loss；
    - 其它分量填 0，使得 z 维度为 5，方便与检测版保持一致；
    - conf_hard: 1 - max softmax prob，作为“样本难度”的 proxy；
- 监督信号：
    - target = 1（好样本）若 is_noisy=False；
    - target = 0（坏样本）若 is_noisy=True；
- 模型：
    - 使用 ICML.core.yolo_bias_finetune.MLPFilter(input_dim=5) 作为 g_phi；
- 结果：
    - 将 g_phi 参数保存到 ICML/CIFAR10/Results/gphi_noise/noise_<rate>/seed_<seed>/mlpfilter_cifar10_noise.pt。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import get_default_cifar10_config
from .datasets import build_cifar10_dataloaders
from .models import build_resnet18_cifar10
from ICML.core.yolo_bias_finetune.mlp_filter import MLPFilter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在带人工标签噪声的 CIFAR-10 上训练 g_phi（MLPFilter 噪声检测器）"
    )
    parser.add_argument("--seed", type=int, default=1088, help="随机种子")
    parser.add_argument(
        "--noise-rate",
        type=float,
        default=0.4,
        help="train 集标签噪声比例（0~1）",
    )
    parser.add_argument("--epochs", type=int, default=5, help="g_phi 训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--device", type=str, default=None, help="设备，例如 'cuda:0' 或 'cpu'")
    parser.add_argument(
        "--theta-good-path",
        type=str,
        default=None,
        help="可选：指定 θ_good 权重路径（默认从 CIFAR10/Results/theta_good_seed_xxx/best.pt 推导）",
    )
    return parser.parse_args()


def build_theta_good(cfg, device: torch.device, seed: int, theta_good_path: Path | None) -> nn.Module:
    """
    构建并加载 θ_good（如存在），否则仅返回随机初始化的 ResNet-18。
    """
    model = build_resnet18_cifar10(num_classes=cfg.num_classes, device=device)

    if theta_good_path is None:
        theta_good_path = cfg.results_root / f"theta_good_seed_{seed}" / "best.pt"

    if theta_good_path.exists():
        state = torch.load(theta_good_path, map_location=device)
        model.load_state_dict(state)
        print(f"✅ 已从 {theta_good_path} 加载 θ_good 权重")
    else:
        print(f"⚠️ 未找到 θ_good 权重 {theta_good_path}，将使用随机初始化 ResNet-18")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main() -> None:
    args = parse_args()
    cfg = get_default_cifar10_config()

    set_seed(args.seed)

    device = torch.device(args.device) if args.device is not None else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # 构建带噪声标记的 CIFAR-10 dataloader
    dls = build_cifar10_dataloaders(
        data_root=str(cfg.data_root),
        batch_size=args.batch_size,
        num_workers=cfg.num_workers,
        noise_rate=args.noise_rate,
        seed=args.seed,
    )

    theta_good = build_theta_good(
        cfg,
        device=device,
        seed=args.seed,
        theta_good_path=Path(args.theta_good_path) if args.theta_good_path else None,
    )

    # g_phi：输入 5 维特征 z=[loss_cls,0,0,conf_hard,0]
    g_phi = MLPFilter(input_dim=5, device=device)
    g_phi.to(device)

    optimizer = torch.optim.Adam(g_phi.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    out_root = cfg.results_root / "gphi_noise" / f"noise_{args.noise_rate:g}" / f"seed_{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        g_phi.train()
        running_loss = 0.0
        n_samples = 0

        print(f"\n[Epoch {epoch + 1}/{args.epochs}] 训练 g_phi (noise_rate={args.noise_rate})")

        for imgs, labels, is_noisy in dls.train:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            is_noisy = torch.as_tensor(is_noisy, device=device, dtype=torch.bool)

            with torch.no_grad():
                logits = theta_good(imgs)
                probs = F.softmax(logits, dim=1)
                conf, _ = probs.max(dim=1)
                loss_ce = F.cross_entropy(logits, labels, reduction="none")  # (B,)

            # 构造 5 维特征：
            # loss_cls = CE, 其余两个损失分量置 0；conf_hard = 1 - conf；最后一维占位 0
            zeros = torch.zeros_like(loss_ce)
            conf_hard = 1.0 - conf
            z = torch.stack([loss_ce, zeros, zeros, conf_hard, zeros], dim=1)  # (B,5)
            z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)

            # 监督标签：clean=1, noisy=0
            target = (~is_noisy).float().view(-1, 1)

            optimizer.zero_grad()
            pred_weights = g_phi(z)  # (B,1)
            loss = bce(pred_weights, target)
            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            running_loss += float(loss.detach().cpu().item()) * bs
            n_samples += bs

        avg_loss = running_loss / max(n_samples, 1)
        print(f"✅ Epoch {epoch + 1} 完成，L_BCE(avg)={avg_loss:.4f}")

    out_path = out_root / "mlpfilter_cifar10_noise.pt"
    torch.save(g_phi.state_dict(), out_path)
    print(f"\n💾 已保存 CIFAR-10 g_phi 噪声检测器到: {out_path}")


if __name__ == "__main__":
    main()

