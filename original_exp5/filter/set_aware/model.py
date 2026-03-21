import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, inp: int, hidden: int, out: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SetAwareBiasRobustFilter(nn.Module):
    """
    Set-based filter with dual heads:
    - sample weights w_i
    - global bias correction delta_phi
    - optional gate head (for gated variants)
    """

    def __init__(
        self,
        dim: int,
        hidden: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = MLP(dim, hidden, hidden, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.weight_head = MLP(hidden, hidden, 1, dropout=dropout)
        self.bias_head = MLP(hidden, hidden, dim, dropout=dropout)
        self.gate_head = MLP(hidden, hidden, 1, dropout=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, D] candidate set.
        Returns:
            weights: [B, N] in [0,1]
            delta_phi: [B, D]
        """
        h0 = self.encoder(x)
        h = self.transformer(h0)
        weights = torch.sigmoid(self.weight_head(h)).squeeze(-1)
        h_global = h.mean(dim=1)
        delta_phi = self.bias_head(h_global)
        return weights, delta_phi

    def forward_with_gate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Same as forward, additionally returns a gating scalar in [0,1] predicted from pooled set features.
        """
        h0 = self.encoder(x)
        h = self.transformer(h0)
        weights = torch.sigmoid(self.weight_head(h)).squeeze(-1)
        h_global = h.mean(dim=1)  # [B, hidden]
        delta_phi = self.bias_head(h_global)
        gate = torch.sigmoid(self.gate_head(h_global)).squeeze(-1)  # [B]
        return weights, delta_phi, gate

    @staticmethod
    def weighted_estimate(x: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Compute weighted mean of candidates.
        Args:
            x: [B, N, D]
            weights: [B, N]
        """
        w = weights.unsqueeze(-1)
        num = (w * x).sum(dim=1)
        den = w.sum(dim=1, keepdim=True).clamp_min(eps)
        return num / den

    def forward_correction_only(self, x: torch.Tensor) -> torch.Tensor:
        """
        Correction-only path: ignore weighting head, predict global delta_phi via set interactions.
        Returns:
            delta_phi: [B, D]
        """
        h0 = self.encoder(x)
        h = self.transformer(h0)
        h_global = h.mean(dim=1)
        return self.bias_head(h_global)
