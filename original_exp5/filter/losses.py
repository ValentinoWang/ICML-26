import torch
from torch import nn


def classification_loss(weights: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy on per-sample weights.
    """
    return nn.functional.binary_cross_entropy(weights.clamp(1e-4, 1 - 1e-4), labels)


def contraction_loss(theta_new: torch.Tensor, theta_good: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum((theta_new - theta_good) ** 2, dim=-1))


def ess_loss(weights: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Effective sample size penalty to avoid over-concentration.
    """
    w = weights
    ess = (w.sum(dim=1) ** 2) / (w.pow(2).sum(dim=1) + 1e-8)
    penalty = torch.clamp(tau - ess, min=0.0)
    return penalty.mean()


def correction_reg(delta_phi: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum(delta_phi ** 2, dim=-1))
