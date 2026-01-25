"""
CIFAR-10 简单任务子包（ICML.CIFAR10）

目的：
- 提供一个比 MT / City→Foggy 简单得多的分类任务环境；
- 使用 ResNet-18 + 人工标签噪声，在 CIFAR-10 上训练 g_phi 作为“噪声检测/样本过滤器”；
- 方便与检测版 g_phi 做对比实验，验证稳定性与可解释性。

推荐入口脚本：
- ICML.CIFAR10.train_theta_good_cifar10      —— 在干净 CIFAR-10 上训练 θ_good（ResNet-18）；
- ICML.CIFAR10.train_gphi_noise_cifar10      —— 在有标签噪声的 CIFAR-10 上训练 g_phi（MLPFilter）。
"""

from . import train_theta_good_cifar10, train_gphi_noise_cifar10  # noqa: F401

__all__ = ["train_theta_good_cifar10", "train_gphi_noise_cifar10"]

