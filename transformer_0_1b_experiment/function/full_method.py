from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import torch
from torch import nn


@dataclass
class FullMethodInputs:
    chart: Any
    scene: Any
    prev_e: torch.Tensor
    current_e: torch.Tensor
    ref_candidates: torch.Tensor
    sel_candidates: torch.Tensor
    weights: torch.Tensor
    channel_extras: dict[str, Any] = field(default_factory=dict)
    compose_extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class FullMethodStepResult:
    weighted_channels: dict[str, torch.Tensor]
    weighted_stack: torch.Tensor
    transported_stack: torch.Tensor
    prefixes: list[torch.Tensor]
    hall_basis: torch.Tensor
    pair_index: list[tuple[int, int]]
    weighted_w: torch.Tensor
    weighted_d2: torch.Tensor
    weighted_h: torch.Tensor
    weighted_d_hat: torch.Tensor
    weighted_d_tilde: torch.Tensor
    full_next: torch.Tensor
    residual: torch.Tensor
    residual_gen: float
    volume_change: float
    diversity: float
    direction_norm: float
    order_reg: float


@runtime_checkable
class FullMethodAdapter(Protocol):
    def build_channels(self, inputs: FullMethodInputs) -> dict[str, torch.Tensor]: ...

    def stack_channel_generators(self, channels: dict[str, torch.Tensor]) -> torch.Tensor: ...

    def build_d2(self, channel_stack: torch.Tensor, order_weights: torch.Tensor) -> torch.Tensor: ...

    def matrix_exp(self, matrix: torch.Tensor) -> torch.Tensor: ...

    def adjoint(self, group: torch.Tensor, generator: torch.Tensor) -> torch.Tensor: ...

    def compose_update(
        self,
        chart: Any,
        current_e: torch.Tensor,
        d2: torch.Tensor,
        d_tilde: torch.Tensor,
        *,
        compose_extras: Mapping[str, Any] | None = None,
    ) -> torch.Tensor: ...

    def effective_rank(self, covariance: torch.Tensor) -> float: ...


def generator_feature_vector_torch(generator: torch.Tensor) -> torch.Tensor:
    dim = generator.shape[0] - 1
    linear = generator[:dim, :dim]
    translation = generator[:dim, dim]
    symmetric = 0.5 * (linear + linear.T)
    skew = 0.5 * (linear - linear.T)
    return torch.stack(
        [
            torch.linalg.matrix_norm(linear, ord="fro"),
            torch.linalg.vector_norm(translation),
            torch.trace(linear),
            torch.linalg.matrix_norm(symmetric, ord="fro"),
            torch.linalg.matrix_norm(skew, ord="fro"),
            translation.abs().mean(),
            linear.diag().mean(),
            linear.abs().mean(),
        ]
    )


def transport_channels_torch(
    channel_stack: torch.Tensor,
    *,
    matrix_exp: callable,
    adjoint: callable,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    dim_plus = channel_stack.shape[-1]
    prefix = torch.eye(dim_plus, device=channel_stack.device, dtype=channel_stack.dtype)
    transported = []
    prefixes = [prefix]
    for idx in range(channel_stack.shape[0]):
        transported.append(adjoint(prefix, channel_stack[idx]))
        prefix = prefix @ matrix_exp(channel_stack[idx])
        prefixes.append(prefix)
    return torch.stack(transported, dim=0), prefixes


def build_hall2_basis_torch(channel_stack: torch.Tensor) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    basis = [channel_stack[idx] for idx in range(channel_stack.shape[0])]
    pair_index: list[tuple[int, int]] = []
    for i in range(channel_stack.shape[0]):
        for j in range(i + 1, channel_stack.shape[0]):
            basis.append(channel_stack[i] @ channel_stack[j] - channel_stack[j] @ channel_stack[i])
            pair_index.append((i, j))
    return torch.stack(basis, dim=0), pair_index


def combine_lie_basis_torch(coeffs: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.einsum("k,kij->ij", coeffs, basis)


class TinyMLP(nn.Module):
    def __init__(self, inp: int, hidden: int, out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FullMethodController(nn.Module):
    def __init__(self, channel_count: int = 5, hidden: int = 64, feature_dim: int = 8):
        super().__init__()
        self.channel_count = channel_count
        self.feature_dim = feature_dim
        self.basis_size = channel_count + channel_count * (channel_count - 1) // 2
        self.encoder = TinyMLP(feature_dim, hidden, hidden)
        self.order_head = TinyMLP(hidden * 3, hidden, 1)
        self.corrector_head = TinyMLP(hidden + feature_dim, hidden, self.basis_size)
        self.direction_head = TinyMLP(hidden + feature_dim, hidden, self.basis_size)

    def encode(self, channel_stack: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = torch.stack([generator_feature_vector_torch(channel) for channel in channel_stack], dim=0)
        embeddings = self.encoder(features)
        global_summary = embeddings.mean(dim=0)
        feature_summary = features.mean(dim=0)
        return features, embeddings, torch.cat([global_summary, feature_summary], dim=0)

    def build_order_weights(self, embeddings: torch.Tensor, global_summary: torch.Tensor) -> torch.Tensor:
        weights = []
        for i in range(self.channel_count):
            for j in range(i + 1, self.channel_count):
                score_ij = self.order_head(torch.cat([embeddings[i], embeddings[j], global_summary], dim=0)).squeeze(-1)
                score_ji = self.order_head(torch.cat([embeddings[j], embeddings[i], global_summary], dim=0)).squeeze(-1)
                weights.append(torch.tanh(score_ij - score_ji))
        return torch.stack(weights, dim=0)

    def forward(
        self,
        channel_stack: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, embeddings, context = self.encode(channel_stack)
        weights = self.build_order_weights(embeddings, embeddings.mean(dim=0))
        corrector_coeffs = self.corrector_head(context)
        direction_coeffs = self.direction_head(context)
        return features, embeddings, weights, corrector_coeffs, direction_coeffs


def run_full_method_step_aligned(
    *,
    inputs: FullMethodInputs,
    controller: FullMethodController,
    adapter: FullMethodAdapter,
) -> FullMethodStepResult:
    weighted_channels = adapter.build_channels(inputs)
    weighted_stack = adapter.stack_channel_generators(weighted_channels)
    transported_stack, prefixes = transport_channels_torch(
        weighted_stack,
        matrix_exp=adapter.matrix_exp,
        adjoint=adapter.adjoint,
    )
    _, _, learned_w, corrector_coeffs, direction_coeffs = controller(transported_stack)
    weighted_d2 = adapter.build_d2(transported_stack, learned_w)
    hall_basis, pair_index = build_hall2_basis_torch(transported_stack)
    weighted_d_hat = combine_lie_basis_torch(corrector_coeffs, hall_basis)
    weighted_h = combine_lie_basis_torch(direction_coeffs, hall_basis)
    weighted_d_tilde = adapter.adjoint(adapter.matrix_exp(weighted_h), weighted_d_hat)
    full_next = adapter.compose_update(
        inputs.chart,
        inputs.current_e,
        weighted_d2,
        weighted_d_tilde,
        compose_extras=inputs.compose_extras,
    )

    residual = weighted_d2 - weighted_d_tilde
    volume_change = float(torch.abs(torch.trace(residual[: inputs.chart.dim, : inputs.chart.dim])).item())
    diversity = adapter.effective_rank(weighted_channels["sel_cov"])

    return FullMethodStepResult(
        weighted_channels=weighted_channels,
        weighted_stack=weighted_stack,
        transported_stack=transported_stack,
        prefixes=prefixes,
        hall_basis=hall_basis,
        pair_index=pair_index,
        weighted_w=learned_w,
        weighted_d2=weighted_d2,
        weighted_h=weighted_h,
        weighted_d_hat=weighted_d_hat,
        weighted_d_tilde=weighted_d_tilde,
        full_next=full_next,
        residual=residual,
        residual_gen=float(torch.linalg.matrix_norm(residual, ord="fro").item()),
        volume_change=volume_change,
        diversity=diversity,
        direction_norm=float(torch.linalg.matrix_norm(weighted_h, ord="fro").item()),
        order_reg=float(torch.mean(learned_w.pow(2)).item()),
    )


__all__ = [
    "FullMethodAdapter",
    "FullMethodController",
    "FullMethodInputs",
    "FullMethodStepResult",
    "build_hall2_basis_torch",
    "combine_lie_basis_torch",
    "generator_feature_vector_torch",
    "run_full_method_step_aligned",
    "transport_channels_torch",
]
