import torch
from torch import nn


class StandardFilter(nn.Module):
    """
    Point-wise filter: encodes each candidate independently and outputs weights.
    No global context, no explicit bias correction head.
    """

    def __init__(self, dim: int, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.weight_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
        Returns:
            weights: [B, N] in [0,1]
        """
        h = self.encoder(x)
        weights = torch.sigmoid(self.weight_head(h)).squeeze(-1)
        return weights

    @staticmethod
    def weighted_estimate(x: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        w = weights.unsqueeze(-1)
        num = (w * x).sum(dim=1)
        den = w.sum(dim=1, keepdim=True).clamp_min(eps)
        return num / den
