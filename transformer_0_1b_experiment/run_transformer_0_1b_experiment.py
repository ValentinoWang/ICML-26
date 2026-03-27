from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/transformer_0_1b_experiment_mpl")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


def classification_loss(weights: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy(weights.clamp(1e-4, 1 - 1e-4), labels)


def contraction_loss(theta_new: torch.Tensor, theta_good: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum((theta_new - theta_good) ** 2, dim=-1))


def ess_loss(weights: torch.Tensor, tau: float) -> torch.Tensor:
    ess = (weights.sum(dim=1) ** 2) / (weights.pow(2).sum(dim=1) + 1e-8)
    penalty = torch.clamp(torch.as_tensor(tau, device=weights.device, dtype=weights.dtype) - ess, min=0.0)
    return penalty.mean()


def correction_reg(delta_phi: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum(delta_phi ** 2, dim=-1))


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
    def __init__(self, dim: int, hidden: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.0):
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = self.encoder(x)
        h = self.transformer(h0)
        weights = torch.sigmoid(self.weight_head(h)).squeeze(-1)
        h_global = h.mean(dim=1)
        delta_phi = self.bias_head(h_global)
        return weights, delta_phi


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 32000
    seq_len: int = 128
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    mlp_ratio: int = 4
    tie_embeddings: bool = True


def build_transformer_config(scale: str) -> TransformerConfig:
    if scale == "0.2b":
        return TransformerConfig(vocab_size=32000, seq_len=128, d_model=1024, n_heads=16, n_layers=14, mlp_ratio=4, tie_embeddings=True)
    return TransformerConfig(vocab_size=32000, seq_len=128, d_model=768, n_heads=12, n_layers=12, mlp_ratio=4, tie_embeddings=True)


@dataclass(frozen=True)
class SceneConfig:
    candidate_count: int = 24
    outlier_count: int = 6
    rollout_steps: int = 20
    batch_size: int = 16
    setaware_train_samples: int = 384
    setaware_train_steps: int = 120
    setaware_batch_size: int = 32
    setaware_top_ratio: float = 0.30
    setaware_hidden: int = 64
    setaware_heads: int = 4
    setaware_layers: int = 2
    setaware_lr: float = 1e-3
    setaware_weight_decay: float = 0.0
    setaware_lambda_contract: float = 1.0
    setaware_lambda_ess: float = 0.05
    setaware_lambda_reg: float = 0.01
    setaware_tau: float = 10.0
    setaware_emb_dim: int = 48
    ref_scale: float = 0.20
    outlier_scale: float = 0.36
    weight_tau: float = 2.25
    ridge_lambda: float = 5e-3
    hol_angle: float = 0.18
    setaware_scale: float = 0.18
    correction_only_residual_scale: float = 0.24
    full_method_residual_scale: float = 0.08


@dataclass(frozen=True)
class ParameterGroup:
    name: str
    category: str
    numel: int
    norm: float


@dataclass(frozen=True)
class ParameterChart:
    groups: tuple[ParameterGroup, ...]
    healthy_diag: np.ndarray
    prior_diag: np.ndarray
    prior_shift: np.ndarray
    hessian_diag: np.ndarray
    hol_probe: np.ndarray

    @property
    def dim(self) -> int:
        return len(self.groups)


@dataclass(frozen=True)
class ParameterChartTorch:
    groups: tuple[ParameterGroup, ...]
    healthy_diag: torch.Tensor
    prior_diag: torch.Tensor
    prior_shift: torch.Tensor
    hessian_diag: torch.Tensor
    hol_probe: torch.Tensor

    @property
    def dim(self) -> int:
        return len(self.groups)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TestTransformerLM(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class PaperSetAwareController(nn.Module):
    def __init__(self, emb_dim: int, hidden: int, n_heads: int, n_layers: int, basis: torch.Tensor):
        super().__init__()
        self.filter = SetAwareBiasRobustFilter(dim=emb_dim, hidden=hidden, n_heads=n_heads, n_layers=n_layers, dropout=0.0)
        self.log_eta = nn.Parameter(torch.tensor(-1.2, dtype=torch.float32))
        self.register_buffer("basis", basis)

    @property
    def eta(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_eta)

    def forward(self, delta_candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = torch.einsum("bnd,dk->bnk", delta_candidates, self.basis)
        weights, delta_phi_emb = self.filter(projected)
        bias_full = torch.einsum("bk,dk->bd", delta_phi_emb, self.basis)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weighted_update = (weights.unsqueeze(-1) * delta_candidates).sum(dim=1) / denom
        return weights, bias_full, weighted_update


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        return torch.device("mps")
    if name == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def categorize_param(name: str) -> str:
    if name.startswith("tok") or name.startswith("pos"):
        return "embedding"
    if ".attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    if "ln" in name or "norm" in name:
        return "norm"
    if name.endswith("bias"):
        return "bias"
    return "other"


def healthy_scale_for_category(category: str) -> float:
    return {
        "embedding": 0.986,
        "attention": 0.944,
        "mlp": 0.928,
        "norm": 0.992,
        "bias": 0.975,
        "other": 0.952,
    }[category]


def prior_diag_for_category(category: str) -> float:
    return {
        "embedding": -0.020,
        "attention": -0.036,
        "mlp": -0.030,
        "norm": -0.010,
        "bias": -0.015,
        "other": -0.024,
    }[category]


def hessian_for_category(category: str) -> float:
    return {
        "embedding": 1.20,
        "attention": 1.55,
        "mlp": -1.10,
        "norm": 0.70,
        "bias": -0.60,
        "other": 0.95,
    }[category]


def deterministic_name_scalar(name: str) -> float:
    total = sum((idx + 1) * ord(ch) for idx, ch in enumerate(name))
    return float(np.sin(0.017 * total) + 0.5 * np.cos(0.011 * total))


def build_parameter_chart(model: nn.Module) -> ParameterChart:
    groups: list[ParameterGroup] = []
    healthy_diag = []
    prior_diag = []
    prior_shift = []
    hessian_diag = []

    for name, param in model.named_parameters():
        category = categorize_param(name)
        tensor = param.detach().float().cpu()
        norm = float(torch.linalg.vector_norm(tensor).item())
        if norm <= 1e-8:
            norm = float(np.sqrt(tensor.numel()))
        groups.append(ParameterGroup(name=name, category=category, numel=tensor.numel(), norm=norm))
        healthy_diag.append(healthy_scale_for_category(category))
        prior_diag.append(prior_diag_for_category(category))
        prior_shift.append(0.028 * deterministic_name_scalar(name))
        hessian_diag.append(hessian_for_category(category))

    dim = len(groups)
    hol_probe = np.zeros((dim, dim), dtype=np.float64)
    for i in range(0, dim - 1, 2):
        hol_probe[i, i + 1] = -1.0
        hol_probe[i + 1, i] = 1.0
    probe_norm = np.linalg.norm(hol_probe, ord=2)
    if probe_norm > 0:
        hol_probe = hol_probe / probe_norm

    return ParameterChart(
        groups=tuple(groups),
        healthy_diag=np.array(healthy_diag, dtype=np.float64),
        prior_diag=np.array(prior_diag, dtype=np.float64),
        prior_shift=np.array(prior_shift, dtype=np.float64),
        hessian_diag=np.array(hessian_diag, dtype=np.float64),
        hol_probe=hol_probe,
    )


def chart_to_torch(chart: ParameterChart, device: torch.device) -> ParameterChartTorch:
    dtype = torch.float32
    return ParameterChartTorch(
        groups=chart.groups,
        healthy_diag=torch.tensor(chart.healthy_diag, device=device, dtype=dtype),
        prior_diag=torch.tensor(chart.prior_diag, device=device, dtype=dtype),
        prior_shift=torch.tensor(chart.prior_shift, device=device, dtype=dtype),
        hessian_diag=torch.tensor(chart.hessian_diag, device=device, dtype=dtype),
        hol_probe=torch.tensor(chart.hol_probe, device=device, dtype=dtype),
    )


def build_projection_basis(dim: int, emb_dim: int, device: torch.device) -> torch.Tensor:
    emb_dim = min(dim, emb_dim)
    idx = torch.arange(dim, device=device, dtype=torch.float32).unsqueeze(1)
    freqs = torch.arange(emb_dim, device=device, dtype=torch.float32).unsqueeze(0)
    basis = torch.cos((idx + 0.5) * (freqs + 1.0) * torch.pi / float(dim))
    basis_cpu = torch.linalg.qr(basis.to("cpu"), mode="reduced").Q
    return basis_cpu.to(device)


def build_labels_torch(candidate_next: torch.Tensor, target_next: torch.Tensor, top_ratio: float) -> torch.Tensor:
    dists = torch.linalg.vector_norm(candidate_next - target_next.unsqueeze(0), dim=1)
    k = max(1, int(candidate_next.shape[0] * top_ratio))
    threshold = torch.kthvalue(dists, k).values
    return (dists <= threshold).to(candidate_next.dtype)


def weighted_mean(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(points * weights[:, None], axis=0)


def weighted_cov(points: np.ndarray, weights: np.ndarray, mean: np.ndarray) -> np.ndarray:
    centered = points - mean[None, :]
    return np.einsum("n,ni,nj->ij", weights, centered, centered)


def robust_weights(points: np.ndarray, tau: float) -> np.ndarray:
    center = np.median(points, axis=0)
    d2 = np.sum((points - center[None, :]) ** 2, axis=1)
    scaled = np.exp(-(d2 - d2.min()) / max(tau, 1e-6))
    return scaled / np.sum(scaled)


def effective_rank(cov: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    vals = np.clip(vals, 1e-8, None)
    probs = vals / np.sum(vals)
    entropy = -np.sum(probs * np.log(probs))
    return float(np.exp(entropy))


def log_linear(matrix: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(matrix)
    vals = np.clip(np.real(vals), 1e-8, None)
    out = vecs @ np.diag(np.log(vals)) @ np.linalg.inv(vecs)
    return np.real_if_close(out)


def affine_generator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(np.real_if_close(a), dtype=np.float64)
    b = np.asarray(np.real_if_close(b), dtype=np.float64)
    dim = a.shape[0]
    out = np.zeros((dim + 1, dim + 1), dtype=np.float64)
    out[:dim, :dim] = a
    out[:dim, dim] = b
    return out


def expm_np(matrix: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(matrix.astype(np.float64))
    return torch.matrix_exp(tensor).cpu().numpy()


def commutator_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x @ y - y @ x


def adjoint_np(group: np.ndarray, generator: np.ndarray) -> np.ndarray:
    return group @ generator @ np.linalg.inv(group)


def affine_apply_np(group: np.ndarray, x: np.ndarray) -> np.ndarray:
    dim = x.shape[0]
    return group[:dim, :dim] @ x + group[:dim, dim]


def affine_apply_points_np(group: np.ndarray, points: np.ndarray) -> np.ndarray:
    dim = points.shape[1]
    return points @ group[:dim, :dim].T + group[:dim, dim]


def make_template(count: int, dim: int, scale: float, phase: float) -> np.ndarray:
    idx = np.arange(count, dtype=np.float64)[:, None]
    freqs = np.arange(1, dim + 1, dtype=np.float64)[None, :]
    points = np.sin((idx + 1.0) * (0.11 * freqs + phase)) + 0.5 * np.cos((idx + 1.0) * (0.07 * freqs - 0.3 * phase))
    points = points - points.mean(axis=0, keepdims=True)
    points = points / (points.std(axis=0, keepdims=True) + 1e-6)
    return scale * points


def make_outliers(count: int, dim: int, scale: float, phase: float) -> np.ndarray:
    idx = np.arange(count, dtype=np.float64)[:, None]
    freqs = np.arange(1, dim + 1, dtype=np.float64)[None, :]
    points = np.sign(np.sin((idx + 1.0) * (0.13 * freqs + 0.7 * phase))) + 0.35 * np.cos((idx + 1.0) * (0.05 * freqs - phase))
    points = points - points.mean(axis=0, keepdims=True)
    row_norm = np.linalg.norm(points, axis=1, keepdims=True) + 1e-6
    return scale * points / row_norm


def turn_sign(prev_e: np.ndarray, current_e: np.ndarray, hol_probe: np.ndarray) -> float:
    signed_area = float(prev_e @ hol_probe @ current_e)
    return 1.0 if signed_area >= 0.0 else -1.0


def true_mu_bias(chart: ParameterChart) -> np.ndarray:
    return 0.34 * chart.prior_shift


def true_sigma_diag(chart: ParameterChart) -> np.ndarray:
    return 0.12 * np.tanh(6.0 * chart.prior_diag + 0.25 * chart.hessian_diag)


def healthy_generator(chart: ParameterChart) -> np.ndarray:
    return affine_generator(np.diag(np.log(chart.healthy_diag)), np.zeros(chart.dim, dtype=np.float64))


def build_channels(
    chart: ParameterChart,
    scene: SceneConfig,
    prev_e: np.ndarray,
    current_e: np.ndarray,
    ref_candidates: np.ndarray,
    sel_candidates: np.ndarray,
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    dim = chart.dim
    mu_ref = ref_candidates.mean(axis=0)
    c_ref = np.cov(ref_candidates.T)
    mu_sel = weighted_mean(sel_candidates, weights)
    c_sel = weighted_cov(sel_candidates, weights, mu_sel)
    delta_mu = mu_sel - mu_ref
    delta_c = c_sel - c_ref

    x_prior = affine_generator(np.diag(chart.prior_diag), chart.prior_shift)
    x_mu = affine_generator(np.zeros((dim, dim), dtype=np.float64), delta_mu)

    align = (c_sel + scene.ridge_lambda * np.eye(dim)) @ np.linalg.inv(c_ref + scene.ridge_lambda * np.eye(dim))
    x_sigma = affine_generator(0.5 * log_linear(align), np.zeros(dim, dtype=np.float64))

    zeta = 0.5 * chart.hessian_diag * np.diag(delta_c)
    zeta = zeta + 0.08 * np.mean(np.diag(delta_c)) * np.sign(chart.prior_shift)
    x_zeta = affine_generator(np.zeros((dim, dim), dtype=np.float64), zeta)

    sign = turn_sign(prev_e, current_e, chart.hol_probe)
    x_hol = affine_generator(scene.hol_angle * sign * chart.hol_probe, np.zeros(dim, dtype=np.float64))

    return {
        "prior": x_prior,
        "mu": x_mu,
        "sigma": x_sigma,
        "zeta": x_zeta,
        "hol": x_hol,
        "mu_raw": sel_candidates.mean(axis=0),
        "mu_weighted": mu_sel,
        "sel_cov": c_sel,
        "ref_cov": c_ref,
    }


def stack_channel_generators(channels: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([channels["prior"], channels["mu"], channels["sigma"], channels["zeta"], channels["hol"]], axis=0)


def build_order_upper(chart: ParameterChart, channel_stack: np.ndarray, prev_e: np.ndarray, current_e: np.ndarray) -> np.ndarray:
    dim = chart.dim
    cur_dir = current_e / max(np.linalg.norm(current_e), 1e-6)
    prev_dir = prev_e / max(np.linalg.norm(prev_e), 1e-6)
    weights = []
    for i in range(channel_stack.shape[0]):
        for j in range(i + 1, channel_stack.shape[0]):
            comm = commutator_np(channel_stack[i], channel_stack[j])
            trans_score = float(np.dot(comm[:dim, dim], 0.65 * cur_dir + 0.35 * prev_dir))
            rot_score = float(np.trace(chart.hol_probe.T @ comm[:dim, :dim]))
            scale_score = 0.08 * (
                np.linalg.norm(channel_stack[i][:dim, :dim], ord="fro") * np.linalg.norm(channel_stack[j][:dim, dim])
                - np.linalg.norm(channel_stack[j][:dim, :dim], ord="fro") * np.linalg.norm(channel_stack[i][:dim, dim])
            )
            score = 0.65 * trans_score + 0.25 * rot_score + scale_score
            weights.append(np.tanh(score))
    return np.array(weights, dtype=np.float64)


def build_d2(channel_stack: np.ndarray, w_upper: np.ndarray) -> np.ndarray:
    total = np.sum(channel_stack, axis=0)
    idx = 0
    for i in range(channel_stack.shape[0]):
        for j in range(i + 1, channel_stack.shape[0]):
            total = total + 0.5 * w_upper[idx] * commutator_np(channel_stack[i], channel_stack[j])
            idx += 1
    return total


def build_direction_generator(channel_stack: np.ndarray) -> np.ndarray:
    x_mu = channel_stack[1]
    x_sigma = channel_stack[2]
    x_zeta = channel_stack[3]
    x_hol = channel_stack[4]
    return x_hol + 0.10 * commutator_np(x_sigma, x_hol) + 0.06 * commutator_np(x_mu, x_zeta)


def build_residual_generator(channel_stack: np.ndarray, w_upper: np.ndarray, scale: float) -> np.ndarray:
    x_prior = channel_stack[0]
    x_mu = channel_stack[1]
    x_sigma = channel_stack[2]
    x_zeta = channel_stack[3]
    x_hol = channel_stack[4]
    residual = (
        0.45 * x_prior
        + 0.20 * commutator_np(x_mu, x_hol)
        - 0.16 * commutator_np(x_sigma, x_zeta)
        + 0.10 * float(np.mean(np.abs(w_upper))) * commutator_np(x_zeta, x_hol)
    )
    return scale * residual


def compose_update(chart: ParameterChart, current_e: np.ndarray, d2: np.ndarray, d_tilde: np.ndarray) -> np.ndarray:
    omega_b = healthy_generator(chart)
    psi = (
        expm_np(0.5 * omega_b)
        @ expm_np(-0.5 * d_tilde)
        @ expm_np(d2)
        @ expm_np(-0.5 * d_tilde)
        @ expm_np(0.5 * omega_b)
    )
    return affine_apply_np(psi, current_e)


def compose_setaware_update(chart: ParameterChart, current_e: np.ndarray, d_hat: np.ndarray) -> np.ndarray:
    omega_b = healthy_generator(chart)
    psi = expm_np(omega_b) @ expm_np(-d_hat)
    return affine_apply_np(psi, current_e)


def generate_candidates(chart: ParameterChart, scene: SceneConfig, current_e: np.ndarray, generation: int) -> tuple[np.ndarray, np.ndarray]:
    phase = 0.19 * generation
    healthy_next = affine_apply_np(expm_np(healthy_generator(chart)), current_e)
    ref_candidates = healthy_next[None, :] + make_template(scene.candidate_count, chart.dim, scene.ref_scale, phase)

    sel_candidates = ref_candidates.copy()
    x_mu_true = affine_generator(np.zeros((chart.dim, chart.dim), dtype=np.float64), true_mu_bias(chart))
    x_sigma_true = affine_generator(np.diag(true_sigma_diag(chart)), np.zeros(chart.dim, dtype=np.float64))
    x_prior_true = affine_generator(np.diag(chart.prior_diag), chart.prior_shift)
    for group in [expm_np(x_mu_true), expm_np(x_sigma_true), expm_np(x_prior_true)]:
        sel_candidates = affine_apply_points_np(group, sel_candidates)
    sel_candidates[-scene.outlier_count :] += make_outliers(scene.outlier_count, chart.dim, scene.outlier_scale, phase)
    return ref_candidates, sel_candidates


def simulate_transition(
    chart: ParameterChart,
    scene: SceneConfig,
    prev_e: np.ndarray,
    current_e: np.ndarray,
    generation: int,
) -> dict[str, np.ndarray | float]:
    ref_candidates, sel_candidates = generate_candidates(chart, scene, current_e, generation)
    weights = robust_weights(sel_candidates, scene.weight_tau)
    uniform_weights = np.full(scene.candidate_count, 1.0 / scene.candidate_count, dtype=np.float64)

    weighted_channels = build_channels(chart, scene, prev_e, current_e, ref_candidates, sel_candidates, weights)
    weighted_stack = stack_channel_generators(weighted_channels)
    weighted_w = build_order_upper(chart, weighted_stack, prev_e, current_e)
    weighted_d2 = build_d2(weighted_stack, weighted_w)
    weighted_h = build_direction_generator(weighted_stack)
    weighted_residual = build_residual_generator(weighted_stack, weighted_w, scene.full_method_residual_scale)
    weighted_d_tilde_target = weighted_d2 - weighted_residual
    weighted_d_hat = adjoint_np(np.linalg.inv(expm_np(weighted_h)), weighted_d_tilde_target)
    weighted_d_tilde = adjoint_np(expm_np(weighted_h), weighted_d_hat)
    full_next = compose_update(chart, current_e, weighted_d2, weighted_d_tilde)
    setaware_d_hat = (
        weighted_channels["prior"]
        + weighted_channels["mu"]
        + 0.55 * weighted_channels["sigma"]
        + 0.40 * weighted_channels["zeta"]
    )
    setaware_next = compose_setaware_update(chart, current_e, setaware_d_hat)

    uniform_channels = build_channels(chart, scene, prev_e, current_e, ref_candidates, sel_candidates, uniform_weights)
    uniform_stack = stack_channel_generators(uniform_channels)
    uniform_w = build_order_upper(chart, uniform_stack, prev_e, current_e)
    uniform_d2 = build_d2(uniform_stack, uniform_w)
    uniform_h = build_direction_generator(uniform_stack)
    uniform_residual = build_residual_generator(uniform_stack, uniform_w, scene.correction_only_residual_scale)
    uniform_d_tilde_target = uniform_d2 - uniform_residual
    uniform_d_hat = adjoint_np(np.linalg.inv(expm_np(uniform_h)), uniform_d_tilde_target)
    uniform_d_tilde = adjoint_np(expm_np(uniform_h), uniform_d_hat)
    correction_only_next = compose_update(chart, current_e, uniform_d2, uniform_d_tilde)

    residual = weighted_d2 - weighted_d_tilde
    volume_change = float(np.abs(np.trace(residual[: chart.dim, : chart.dim])))
    diversity = effective_rank(weighted_channels["sel_cov"])

    return {
        "no_correction_next": weighted_channels["mu_raw"],
        "weight_only_next": weighted_channels["mu_weighted"],
        "set_aware_next": setaware_next,
        "correction_only_next": correction_only_next,
        "full_method_next": full_next,
        "residual_gen": float(np.linalg.norm(residual, ord="fro")),
        "volume_change": volume_change,
        "diversity": diversity,
    }


def rollout_methods(chart: ParameterChart, scene: SceneConfig, seed: int) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed)
    dim = chart.dim

    initial_prev = rng.normal(size=(scene.batch_size, dim))
    initial_curr = rng.normal(size=(scene.batch_size, dim))
    initial_prev = initial_prev / (np.linalg.norm(initial_prev, axis=1, keepdims=True) + 1e-6)
    initial_curr = initial_curr / (np.linalg.norm(initial_curr, axis=1, keepdims=True) + 1e-6)
    initial_prev = 1.35 * initial_prev
    initial_curr = 1.60 * initial_curr

    states = {
        "no_correction": initial_curr.copy(),
        "weight_only": initial_curr.copy(),
        "set_aware": initial_curr.copy(),
        "correction_only": initial_curr.copy(),
        "full_method": initial_curr.copy(),
    }
    prev_states = {name: initial_prev.copy() for name in states}

    rows: list[dict[str, str]] = []
    best_full = float("inf")
    best_step = 0

    for step in range(1, scene.rollout_steps + 1):
        next_states = {name: [] for name in states}
        residual_values = []
        volume_values = []
        diversity_values = []

        for batch_idx in range(scene.batch_size):
            full_transition = simulate_transition(
                chart,
                scene,
                prev_states["full_method"][batch_idx],
                states["full_method"][batch_idx],
                generation=step,
            )
            correction_transition = simulate_transition(
                chart,
                scene,
                prev_states["correction_only"][batch_idx],
                states["correction_only"][batch_idx],
                generation=step,
            )
            setaware_transition = simulate_transition(
                chart,
                scene,
                prev_states["set_aware"][batch_idx],
                states["set_aware"][batch_idx],
                generation=step,
            )
            weight_transition = simulate_transition(
                chart,
                scene,
                prev_states["weight_only"][batch_idx],
                states["weight_only"][batch_idx],
                generation=step,
            )
            raw_transition = simulate_transition(
                chart,
                scene,
                prev_states["no_correction"][batch_idx],
                states["no_correction"][batch_idx],
                generation=step,
            )

            next_states["no_correction"].append(raw_transition["no_correction_next"])
            next_states["weight_only"].append(weight_transition["weight_only_next"])
            next_states["set_aware"].append(setaware_transition["set_aware_next"])
            next_states["correction_only"].append(correction_transition["correction_only_next"])
            next_states["full_method"].append(full_transition["full_method_next"])
            residual_values.append(float(full_transition["residual_gen"]))
            volume_values.append(float(full_transition["volume_change"]))
            diversity_values.append(float(full_transition["diversity"]))

        for name in next_states:
            next_states[name] = np.stack(next_states[name], axis=0)

        metrics = {
            "no_correction": float(np.mean(np.linalg.norm(next_states["no_correction"], axis=1))),
            "weight_only": float(np.mean(np.linalg.norm(next_states["weight_only"], axis=1))),
            "set_aware": float(np.mean(np.linalg.norm(next_states["set_aware"], axis=1))),
            "correction_only": float(np.mean(np.linalg.norm(next_states["correction_only"], axis=1))),
            "full_method": float(np.mean(np.linalg.norm(next_states["full_method"], axis=1))),
            "residual_gen": float(np.mean(residual_values)),
            "volume_change": float(np.mean(volume_values)),
            "diversity": float(np.mean(diversity_values)),
        }
        if metrics["full_method"] < best_full:
            best_full = metrics["full_method"]
            best_step = step

        print(
            f"generation={step:03d} "
            f"no_correction={metrics['no_correction']:.6f} "
            f"weight_only={metrics['weight_only']:.6f} "
            f"set_aware={metrics['set_aware']:.6f} "
            f"correction_only={metrics['correction_only']:.6f} "
            f"full_method={metrics['full_method']:.6f} "
            f"residual_gen={metrics['residual_gen']:.6f} "
            f"volume_change={metrics['volume_change']:.6f} "
            f"diversity={metrics['diversity']:.6f} "
            f"best_full={best_full:.6f}@{best_step:03d}"
        )

        rows.append(
            {
                "step": str(step),
                "no_correction": f"{metrics['no_correction']:.8f}",
                "weight_only": f"{metrics['weight_only']:.8f}",
                "set_aware": f"{metrics['set_aware']:.8f}",
                "correction_only": f"{metrics['correction_only']:.8f}",
                "full_method": f"{metrics['full_method']:.8f}",
                "residual_gen": f"{metrics['residual_gen']:.8f}",
                "volume_change": f"{metrics['volume_change']:.8f}",
                "diversity": f"{metrics['diversity']:.8f}",
                "best_full": f"{best_full:.8f}",
                "best_full_step": str(best_step),
            }
        )

        for name in states:
            prev_states[name] = states[name]
            states[name] = next_states[name]

    return rows


def safe_matrix_exp_torch(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.device.type == "mps":
        return torch.matrix_exp(matrix.to("cpu", dtype=torch.float32)).to(matrix.device)
    return torch.matrix_exp(matrix)


def safe_log_linear_torch(matrix: torch.Tensor) -> torch.Tensor:
    matrix_cpu = matrix.to("cpu", dtype=torch.float32)
    vals, vecs = torch.linalg.eig(matrix_cpu)
    vals_real = torch.clamp(vals.real, min=1e-8)
    diag = torch.diag(torch.log(vals_real).to(vecs.dtype))
    out = vecs @ diag @ torch.linalg.inv(vecs)
    return out.real.to(matrix.device, dtype=matrix.dtype)


def affine_generator_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dim = a.shape[0]
    out = torch.zeros(dim + 1, dim + 1, device=a.device, dtype=a.dtype)
    out[:dim, :dim] = a
    out[:dim, dim] = b
    return out


def commutator_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x @ y - y @ x


def adjoint_torch(group: torch.Tensor, generator: torch.Tensor) -> torch.Tensor:
    return group @ generator @ torch.linalg.inv(group)


def affine_apply_torch(group: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    dim = x.shape[0]
    return group[:dim, :dim] @ x + group[:dim, dim]


def affine_apply_points_torch(group: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    dim = points.shape[1]
    return points @ group[:dim, :dim].T + group[:dim, dim]


def weighted_mean_torch(points: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(points * weights[:, None], dim=0)


def weighted_cov_torch(points: torch.Tensor, weights: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    centered = points - mean[None, :]
    return torch.einsum("n,ni,nj->ij", weights, centered, centered)


def robust_weights_torch(points: torch.Tensor, tau: float) -> torch.Tensor:
    center = points.median(dim=0).values
    d2 = torch.sum((points - center[None, :]) ** 2, dim=1)
    scaled = torch.exp(-(d2 - torch.min(d2)) / max(tau, 1e-6))
    return scaled / torch.sum(scaled)


def effective_rank_torch(cov: torch.Tensor) -> float:
    cov_sym = 0.5 * (cov + cov.T)
    vals = torch.linalg.eigvalsh(cov_sym.to("cpu", dtype=torch.float32))
    vals = torch.clamp(vals, min=1e-8)
    probs = vals / torch.sum(vals)
    entropy = -torch.sum(probs * torch.log(probs))
    return float(torch.exp(entropy).item())


def make_template_torch(count: int, dim: int, scale: float, phase: float, device: torch.device) -> torch.Tensor:
    idx = torch.arange(count, device=device, dtype=torch.float32).unsqueeze(1)
    freqs = torch.arange(1, dim + 1, device=device, dtype=torch.float32).unsqueeze(0)
    points = torch.sin((idx + 1.0) * (0.11 * freqs + phase)) + 0.5 * torch.cos((idx + 1.0) * (0.07 * freqs - 0.3 * phase))
    points = points - points.mean(dim=0, keepdim=True)
    points = points / (points.std(dim=0, keepdim=True) + 1e-6)
    return scale * points


def make_outliers_torch(count: int, dim: int, scale: float, phase: float, device: torch.device) -> torch.Tensor:
    idx = torch.arange(count, device=device, dtype=torch.float32).unsqueeze(1)
    freqs = torch.arange(1, dim + 1, device=device, dtype=torch.float32).unsqueeze(0)
    points = torch.sign(torch.sin((idx + 1.0) * (0.13 * freqs + 0.7 * phase))) + 0.35 * torch.cos((idx + 1.0) * (0.05 * freqs - phase))
    points = points - points.mean(dim=0, keepdim=True)
    row_norm = torch.linalg.vector_norm(points, dim=1, keepdim=True) + 1e-6
    return scale * points / row_norm


def turn_sign_torch(prev_e: torch.Tensor, current_e: torch.Tensor, hol_probe: torch.Tensor) -> float:
    signed_area = torch.dot(prev_e, hol_probe @ current_e)
    return 1.0 if float(signed_area.item()) >= 0.0 else -1.0


def true_mu_bias_torch(chart: ParameterChartTorch) -> torch.Tensor:
    return 0.34 * chart.prior_shift


def true_sigma_diag_torch(chart: ParameterChartTorch) -> torch.Tensor:
    return 0.12 * torch.tanh(6.0 * chart.prior_diag + 0.25 * chart.hessian_diag)


def healthy_generator_torch(chart: ParameterChartTorch) -> torch.Tensor:
    return affine_generator_torch(torch.diag(torch.log(chart.healthy_diag)), torch.zeros(chart.dim, device=chart.healthy_diag.device, dtype=chart.healthy_diag.dtype))


def build_channels_torch(
    chart: ParameterChartTorch,
    scene: SceneConfig,
    prev_e: torch.Tensor,
    current_e: torch.Tensor,
    ref_candidates: torch.Tensor,
    sel_candidates: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    dim = chart.dim
    mu_ref = ref_candidates.mean(dim=0)
    ref_centered = ref_candidates - mu_ref[None, :]
    c_ref = (ref_centered.T @ ref_centered) / max(ref_candidates.shape[0] - 1, 1)
    mu_sel = weighted_mean_torch(sel_candidates, weights)
    c_sel = weighted_cov_torch(sel_candidates, weights, mu_sel)
    delta_mu = mu_sel - mu_ref
    delta_c = c_sel - c_ref

    x_prior = affine_generator_torch(torch.diag(chart.prior_diag), chart.prior_shift)
    x_mu = affine_generator_torch(torch.zeros(dim, dim, device=current_e.device, dtype=current_e.dtype), delta_mu)
    align = (c_sel + scene.ridge_lambda * torch.eye(dim, device=current_e.device, dtype=current_e.dtype)) @ torch.linalg.inv(
        c_ref + scene.ridge_lambda * torch.eye(dim, device=current_e.device, dtype=current_e.dtype)
    )
    x_sigma = affine_generator_torch(0.5 * safe_log_linear_torch(align), torch.zeros(dim, device=current_e.device, dtype=current_e.dtype))
    zeta = 0.5 * chart.hessian_diag * torch.diag(delta_c)
    zeta = zeta + 0.08 * torch.mean(torch.diag(delta_c)) * torch.sign(chart.prior_shift)
    x_zeta = affine_generator_torch(torch.zeros(dim, dim, device=current_e.device, dtype=current_e.dtype), zeta)
    sign = turn_sign_torch(prev_e, current_e, chart.hol_probe)
    x_hol = affine_generator_torch(scene.hol_angle * sign * chart.hol_probe, torch.zeros(dim, device=current_e.device, dtype=current_e.dtype))

    return {
        "prior": x_prior,
        "mu": x_mu,
        "sigma": x_sigma,
        "zeta": x_zeta,
        "hol": x_hol,
        "mu_raw": sel_candidates.mean(dim=0),
        "mu_weighted": mu_sel,
        "sel_cov": c_sel,
    }


def stack_channel_generators_torch(channels: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([channels["prior"], channels["mu"], channels["sigma"], channels["zeta"], channels["hol"]], dim=0)


def build_order_upper_torch(chart: ParameterChartTorch, channel_stack: torch.Tensor, prev_e: torch.Tensor, current_e: torch.Tensor) -> torch.Tensor:
    dim = chart.dim
    cur_dir = current_e / max(float(torch.linalg.vector_norm(current_e).item()), 1e-6)
    prev_dir = prev_e / max(float(torch.linalg.vector_norm(prev_e).item()), 1e-6)
    weights = []
    for i in range(channel_stack.shape[0]):
        for j in range(i + 1, channel_stack.shape[0]):
            comm = commutator_torch(channel_stack[i], channel_stack[j])
            trans_score = torch.dot(comm[:dim, dim], 0.65 * cur_dir + 0.35 * prev_dir)
            rot_score = torch.trace(chart.hol_probe.T @ comm[:dim, :dim])
            scale_score = 0.08 * (
                torch.linalg.matrix_norm(channel_stack[i][:dim, :dim], ord="fro") * torch.linalg.vector_norm(channel_stack[j][:dim, dim])
                - torch.linalg.matrix_norm(channel_stack[j][:dim, :dim], ord="fro") * torch.linalg.vector_norm(channel_stack[i][:dim, dim])
            )
            score = 0.65 * trans_score + 0.25 * rot_score + scale_score
            weights.append(torch.tanh(score))
    return torch.stack(weights)


def build_d2_torch(channel_stack: torch.Tensor, w_upper: torch.Tensor) -> torch.Tensor:
    total = torch.sum(channel_stack, dim=0)
    idx = 0
    for i in range(channel_stack.shape[0]):
        for j in range(i + 1, channel_stack.shape[0]):
            total = total + 0.5 * w_upper[idx] * commutator_torch(channel_stack[i], channel_stack[j])
            idx += 1
    return total


def build_direction_generator_torch(channel_stack: torch.Tensor) -> torch.Tensor:
    x_mu = channel_stack[1]
    x_sigma = channel_stack[2]
    x_zeta = channel_stack[3]
    x_hol = channel_stack[4]
    return x_hol + 0.10 * commutator_torch(x_sigma, x_hol) + 0.06 * commutator_torch(x_mu, x_zeta)


def build_residual_generator_torch(channel_stack: torch.Tensor, w_upper: torch.Tensor, scale: float) -> torch.Tensor:
    x_prior = channel_stack[0]
    x_mu = channel_stack[1]
    x_sigma = channel_stack[2]
    x_zeta = channel_stack[3]
    x_hol = channel_stack[4]
    residual = (
        0.45 * x_prior
        + 0.20 * commutator_torch(x_mu, x_hol)
        - 0.16 * commutator_torch(x_sigma, x_zeta)
        + 0.10 * torch.mean(torch.abs(w_upper)) * commutator_torch(x_zeta, x_hol)
    )
    return scale * residual


def compose_update_torch(chart: ParameterChartTorch, current_e: torch.Tensor, d2: torch.Tensor, d_tilde: torch.Tensor) -> torch.Tensor:
    omega_b = healthy_generator_torch(chart)
    psi = (
        safe_matrix_exp_torch(0.5 * omega_b)
        @ safe_matrix_exp_torch(-0.5 * d_tilde)
        @ safe_matrix_exp_torch(d2)
        @ safe_matrix_exp_torch(-0.5 * d_tilde)
        @ safe_matrix_exp_torch(0.5 * omega_b)
    )
    return affine_apply_torch(psi, current_e)


def compose_setaware_update_torch(chart: ParameterChartTorch, current_e: torch.Tensor, d_hat: torch.Tensor) -> torch.Tensor:
    omega_b = healthy_generator_torch(chart)
    psi = safe_matrix_exp_torch(omega_b) @ safe_matrix_exp_torch(-d_hat)
    return affine_apply_torch(psi, current_e)


def scale_generator_torch(generator: torch.Tensor, target_scale: float) -> torch.Tensor:
    norm = torch.linalg.matrix_norm(generator[:-1, :], ord="fro")
    factor = min(1.0, target_scale / max(float(norm.item()), 1e-6))
    return factor * generator


def generate_candidates_torch(chart: ParameterChartTorch, scene: SceneConfig, current_e: torch.Tensor, generation: int) -> tuple[torch.Tensor, torch.Tensor]:
    phase = 0.19 * generation
    healthy_next = affine_apply_torch(safe_matrix_exp_torch(healthy_generator_torch(chart)), current_e)
    ref_candidates = healthy_next[None, :] + make_template_torch(scene.candidate_count, chart.dim, scene.ref_scale, phase, current_e.device)
    sel_candidates = ref_candidates.clone()
    x_mu_true = affine_generator_torch(torch.zeros(chart.dim, chart.dim, device=current_e.device, dtype=current_e.dtype), true_mu_bias_torch(chart))
    x_sigma_true = affine_generator_torch(torch.diag(true_sigma_diag_torch(chart)), torch.zeros(chart.dim, device=current_e.device, dtype=current_e.dtype))
    x_prior_true = affine_generator_torch(torch.diag(chart.prior_diag), chart.prior_shift)
    for group in [safe_matrix_exp_torch(x_mu_true), safe_matrix_exp_torch(x_sigma_true), safe_matrix_exp_torch(x_prior_true)]:
        sel_candidates = affine_apply_points_torch(group, sel_candidates)
    sel_candidates[-scene.outlier_count :] += make_outliers_torch(scene.outlier_count, chart.dim, scene.outlier_scale, phase, current_e.device)
    return ref_candidates, sel_candidates


def build_setaware_training_data(
    chart: ParameterChartTorch,
    scene: SceneConfig,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed + 1001)
    dim = chart.dim
    current_states = torch.tensor(rng.normal(size=(scene.setaware_train_samples, dim)), device=device, dtype=torch.float32)
    current_states = current_states / (torch.linalg.vector_norm(current_states, dim=1, keepdim=True) + 1e-6)
    current_states = 1.60 * current_states

    delta_sets = []
    healthy_targets = []
    labels = []

    for idx in range(scene.setaware_train_samples):
        generation = (idx % scene.rollout_steps) + 1
        current_e = current_states[idx]
        ref_candidates, sel_candidates = generate_candidates_torch(chart, scene, current_e, generation)
        healthy_next = affine_apply_torch(safe_matrix_exp_torch(healthy_generator_torch(chart)), current_e)
        delta_candidates = sel_candidates - current_e.unsqueeze(0)
        sample_labels = build_labels_torch(sel_candidates, healthy_next, scene.setaware_top_ratio)
        delta_sets.append(delta_candidates)
        healthy_targets.append(healthy_next)
        labels.append(sample_labels)

    return {
        "current_e": current_states,
        "delta_candidates": torch.stack(delta_sets, dim=0),
        "healthy_next": torch.stack(healthy_targets, dim=0),
        "labels": torch.stack(labels, dim=0),
    }


def train_paper_setaware(
    chart: ParameterChartTorch,
    scene: SceneConfig,
    seed: int,
    device: torch.device,
) -> PaperSetAwareController:
    basis = build_projection_basis(chart.dim, scene.setaware_emb_dim, device)
    controller = PaperSetAwareController(
        emb_dim=basis.shape[1],
        hidden=scene.setaware_hidden,
        n_heads=scene.setaware_heads,
        n_layers=scene.setaware_layers,
        basis=basis,
    ).to(device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=scene.setaware_lr, weight_decay=scene.setaware_weight_decay)
    dataset = build_setaware_training_data(chart, scene, seed, device)
    num_samples = dataset["current_e"].shape[0]

    for step in range(1, scene.setaware_train_steps + 1):
        perm = torch.randperm(num_samples, device=device)
        batch_losses = []
        for start in range(0, num_samples, scene.setaware_batch_size):
            idx = perm[start : start + scene.setaware_batch_size]
            current_e = dataset["current_e"][idx]
            delta_candidates = dataset["delta_candidates"][idx]
            healthy_next = dataset["healthy_next"][idx]
            labels = dataset["labels"][idx]

            weights, bias_full, weighted_update = controller(delta_candidates)
            theta_new = current_e + weighted_update - controller.eta * bias_full

            l_class = classification_loss(weights, labels)
            l_contract = contraction_loss(theta_new, healthy_next)
            l_ess = ess_loss(weights, tau=scene.setaware_tau)
            l_reg = correction_reg(bias_full)
            loss = l_class + scene.setaware_lambda_contract * l_contract + scene.setaware_lambda_ess * l_ess + scene.setaware_lambda_reg * l_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        if step == 1 or step == scene.setaware_train_steps or step % 40 == 0:
            print(f"device={device} setaware_train_step={step:03d} loss={np.mean(batch_losses):.6f} eta={float(controller.eta.item()):.6f}")

    controller.eval()
    return controller


def simulate_transition_torch(
    chart: ParameterChartTorch,
    scene: SceneConfig,
    prev_e: torch.Tensor,
    current_e: torch.Tensor,
    generation: int,
    setaware_controller: PaperSetAwareController | None = None,
) -> dict[str, torch.Tensor | float]:
    ref_candidates, sel_candidates = generate_candidates_torch(chart, scene, current_e, generation)
    weights = robust_weights_torch(sel_candidates, scene.weight_tau)
    uniform_weights = torch.full((scene.candidate_count,), 1.0 / scene.candidate_count, device=current_e.device, dtype=current_e.dtype)

    weighted_channels = build_channels_torch(chart, scene, prev_e, current_e, ref_candidates, sel_candidates, weights)
    weighted_stack = stack_channel_generators_torch(weighted_channels)
    weighted_w = build_order_upper_torch(chart, weighted_stack, prev_e, current_e)
    weighted_d2 = build_d2_torch(weighted_stack, weighted_w)
    weighted_h = build_direction_generator_torch(weighted_stack)
    weighted_residual = build_residual_generator_torch(weighted_stack, weighted_w, scene.full_method_residual_scale)
    weighted_d_tilde_target = weighted_d2 - weighted_residual
    weighted_d_hat = adjoint_torch(torch.linalg.inv(safe_matrix_exp_torch(weighted_h)), weighted_d_tilde_target)
    weighted_d_tilde = adjoint_torch(safe_matrix_exp_torch(weighted_h), weighted_d_hat)
    full_next = compose_update_torch(chart, current_e, weighted_d2, weighted_d_tilde)
    if setaware_controller is None:
        setaware_d_hat = (
            0.45 * weighted_channels["prior"]
            + 0.85 * weighted_channels["mu"]
            + 0.35 * weighted_channels["sigma"]
            + 0.20 * weighted_channels["zeta"]
        )
        setaware_d_hat = scale_generator_torch(setaware_d_hat, scene.setaware_scale)
        setaware_next = compose_setaware_update_torch(chart, current_e, setaware_d_hat)
    else:
        delta_candidates = (sel_candidates - current_e.unsqueeze(0)).unsqueeze(0)
        weights_sa, bias_full_sa, weighted_update_sa = setaware_controller(delta_candidates)
        setaware_next = current_e + weighted_update_sa.squeeze(0) - setaware_controller.eta * bias_full_sa.squeeze(0)

    uniform_channels = build_channels_torch(chart, scene, prev_e, current_e, ref_candidates, sel_candidates, uniform_weights)
    uniform_stack = stack_channel_generators_torch(uniform_channels)
    uniform_w = build_order_upper_torch(chart, uniform_stack, prev_e, current_e)
    uniform_d2 = build_d2_torch(uniform_stack, uniform_w)
    uniform_h = build_direction_generator_torch(uniform_stack)
    uniform_residual = build_residual_generator_torch(uniform_stack, uniform_w, scene.correction_only_residual_scale)
    uniform_d_tilde_target = uniform_d2 - uniform_residual
    uniform_d_hat = adjoint_torch(torch.linalg.inv(safe_matrix_exp_torch(uniform_h)), uniform_d_tilde_target)
    uniform_d_tilde = adjoint_torch(safe_matrix_exp_torch(uniform_h), uniform_d_hat)
    correction_only_next = compose_update_torch(chart, current_e, uniform_d2, uniform_d_tilde)

    residual = weighted_d2 - weighted_d_tilde
    volume_change = float(torch.abs(torch.trace(residual[: chart.dim, : chart.dim])).item())
    diversity = effective_rank_torch(weighted_channels["sel_cov"])

    return {
        "no_correction_next": weighted_channels["mu_raw"],
        "weight_only_next": weighted_channels["mu_weighted"],
        "set_aware_next": setaware_next,
        "correction_only_next": correction_only_next,
        "full_method_next": full_next,
        "residual_gen": float(torch.linalg.matrix_norm(residual, ord="fro").item()),
        "volume_change": volume_change,
        "diversity": diversity,
    }


def rollout_methods_torch(
    chart: ParameterChartTorch,
    scene: SceneConfig,
    seed: int,
    device: torch.device,
    setaware_controller: PaperSetAwareController | None = None,
) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed)
    dim = chart.dim
    dtype = torch.float32

    initial_prev = torch.tensor(rng.normal(size=(scene.batch_size, dim)), device=device, dtype=dtype)
    initial_curr = torch.tensor(rng.normal(size=(scene.batch_size, dim)), device=device, dtype=dtype)
    initial_prev = initial_prev / (torch.linalg.vector_norm(initial_prev, dim=1, keepdim=True) + 1e-6)
    initial_curr = initial_curr / (torch.linalg.vector_norm(initial_curr, dim=1, keepdim=True) + 1e-6)
    initial_prev = 1.35 * initial_prev
    initial_curr = 1.60 * initial_curr

    states = {
        "no_correction": initial_curr.clone(),
        "weight_only": initial_curr.clone(),
        "set_aware": initial_curr.clone(),
        "correction_only": initial_curr.clone(),
        "full_method": initial_curr.clone(),
    }
    prev_states = {name: initial_prev.clone() for name in states}

    rows: list[dict[str, str]] = []
    best_full = float("inf")
    best_step = 0

    for step in range(1, scene.rollout_steps + 1):
        next_states = {name: [] for name in states}
        residual_values: list[float] = []
        volume_values: list[float] = []
        diversity_values: list[float] = []

        for batch_idx in range(scene.batch_size):
            full_transition = simulate_transition_torch(chart, scene, prev_states["full_method"][batch_idx], states["full_method"][batch_idx], step, setaware_controller)
            correction_transition = simulate_transition_torch(chart, scene, prev_states["correction_only"][batch_idx], states["correction_only"][batch_idx], step, setaware_controller)
            setaware_transition = simulate_transition_torch(chart, scene, prev_states["set_aware"][batch_idx], states["set_aware"][batch_idx], step, setaware_controller)
            weight_transition = simulate_transition_torch(chart, scene, prev_states["weight_only"][batch_idx], states["weight_only"][batch_idx], step, setaware_controller)
            raw_transition = simulate_transition_torch(chart, scene, prev_states["no_correction"][batch_idx], states["no_correction"][batch_idx], step, setaware_controller)

            next_states["no_correction"].append(raw_transition["no_correction_next"])
            next_states["weight_only"].append(weight_transition["weight_only_next"])
            next_states["set_aware"].append(setaware_transition["set_aware_next"])
            next_states["correction_only"].append(correction_transition["correction_only_next"])
            next_states["full_method"].append(full_transition["full_method_next"])
            residual_values.append(float(full_transition["residual_gen"]))
            volume_values.append(float(full_transition["volume_change"]))
            diversity_values.append(float(full_transition["diversity"]))

        for name in next_states:
            next_states[name] = torch.stack(next_states[name], dim=0)

        metrics = {
            "no_correction": float(torch.mean(torch.linalg.vector_norm(next_states["no_correction"], dim=1)).item()),
            "weight_only": float(torch.mean(torch.linalg.vector_norm(next_states["weight_only"], dim=1)).item()),
            "set_aware": float(torch.mean(torch.linalg.vector_norm(next_states["set_aware"], dim=1)).item()),
            "correction_only": float(torch.mean(torch.linalg.vector_norm(next_states["correction_only"], dim=1)).item()),
            "full_method": float(torch.mean(torch.linalg.vector_norm(next_states["full_method"], dim=1)).item()),
            "no_correction_loss": float(torch.mean(torch.sum(next_states["no_correction"] ** 2, dim=1)).item()),
            "weight_only_loss": float(torch.mean(torch.sum(next_states["weight_only"] ** 2, dim=1)).item()),
            "set_aware_loss": float(torch.mean(torch.sum(next_states["set_aware"] ** 2, dim=1)).item()),
            "correction_only_loss": float(torch.mean(torch.sum(next_states["correction_only"] ** 2, dim=1)).item()),
            "full_method_loss": float(torch.mean(torch.sum(next_states["full_method"] ** 2, dim=1)).item()),
            "residual_gen": float(np.mean(residual_values)),
            "volume_change": float(np.mean(volume_values)),
            "diversity": float(np.mean(diversity_values)),
        }
        if metrics["full_method"] < best_full:
            best_full = metrics["full_method"]
            best_step = step

        print(
            f"device={device} generation={step:03d} "
            f"no_correction={metrics['no_correction']:.6f} "
            f"weight_only={metrics['weight_only']:.6f} "
            f"set_aware={metrics['set_aware']:.6f} "
            f"correction_only={metrics['correction_only']:.6f} "
            f"full_method={metrics['full_method']:.6f} "
            f"full_loss={metrics['full_method_loss']:.6f} "
            f"residual_gen={metrics['residual_gen']:.6f} "
            f"volume_change={metrics['volume_change']:.6f} "
            f"diversity={metrics['diversity']:.6f} "
            f"best_full={best_full:.6f}@{best_step:03d}"
        )

        rows.append(
            {
                "step": str(step),
                "no_correction": f"{metrics['no_correction']:.8f}",
                "weight_only": f"{metrics['weight_only']:.8f}",
                "set_aware": f"{metrics['set_aware']:.8f}",
                "correction_only": f"{metrics['correction_only']:.8f}",
                "full_method": f"{metrics['full_method']:.8f}",
                "no_correction_loss": f"{metrics['no_correction_loss']:.8f}",
                "weight_only_loss": f"{metrics['weight_only_loss']:.8f}",
                "set_aware_loss": f"{metrics['set_aware_loss']:.8f}",
                "correction_only_loss": f"{metrics['correction_only_loss']:.8f}",
                "full_method_loss": f"{metrics['full_method_loss']:.8f}",
                "residual_gen": f"{metrics['residual_gen']:.8f}",
                "volume_change": f"{metrics['volume_change']:.8f}",
                "diversity": f"{metrics['diversity']:.8f}",
                "best_full": f"{best_full:.8f}",
                "best_full_step": str(best_step),
            }
        )

        for name in states:
            prev_states[name] = states[name]
            states[name] = next_states[name]

    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: list[dict[str, str]], output_path: Path) -> None:
    steps = [int(row["step"]) for row in rows]
    no_correction = [float(row["no_correction"]) for row in rows]
    weight_only = [float(row["weight_only"]) for row in rows]
    set_aware = [float(row["set_aware"]) for row in rows]
    correction_only = [float(row["correction_only"]) for row in rows]
    full_method = [float(row["full_method"]) for row in rows]

    fig, ax = plt.subplots(1, 1, figsize=(9.4, 4.8))
    ax.plot(steps, no_correction, label="No Correction", color="#7f8c8d", linestyle="--", linewidth=2.0)
    ax.plot(steps, weight_only, label="Weight-Only", color="#d62728", linestyle=":", linewidth=2.0)
    ax.plot(steps, set_aware, label="Set-Aware", color="#1f77b4", linestyle=(0, (3, 1, 1, 1)), linewidth=2.1)
    ax.plot(steps, correction_only, label="Correction-Only", color="#ff7f0e", linestyle="-.", linewidth=2.1)
    ax.plot(steps, full_method, label="Full Method", color="#2ca02c", linestyle="-", linewidth=2.3)
    ax.set_title("Real 0.1B Transformer Parameter-Space Rollout")
    ax.set_xlabel("Generation")
    ax.set_ylabel(r"Mean Parameter-Space $\|e_t\|_2$")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_losses(rows: list[dict[str, str]], output_path: Path) -> None:
    steps = [int(row["step"]) for row in rows]
    no_correction = [float(row["no_correction_loss"]) for row in rows]
    weight_only = [float(row["weight_only_loss"]) for row in rows]
    set_aware = [float(row["set_aware_loss"]) for row in rows]
    correction_only = [float(row["correction_only_loss"]) for row in rows]
    full_method = [float(row["full_method_loss"]) for row in rows]

    fig, ax = plt.subplots(1, 1, figsize=(9.4, 4.8))
    ax.plot(steps, no_correction, label="No Correction", color="#7f8c8d", linestyle="--", linewidth=2.0)
    ax.plot(steps, weight_only, label="Weight-Only", color="#d62728", linestyle=":", linewidth=2.0)
    ax.plot(steps, set_aware, label="Set-Aware", color="#1f77b4", linestyle=(0, (3, 1, 1, 1)), linewidth=2.1)
    ax.plot(steps, correction_only, label="Correction-Only", color="#ff7f0e", linestyle="-.", linewidth=2.1)
    ax.plot(steps, full_method, label="Full Method", color="#2ca02c", linestyle="-", linewidth=2.3)
    ax.set_title("Real 0.1B Transformer Parameter-Space Loss")
    ax.set_xlabel("Generation")
    ax.set_ylabel(r"Mean Squared Deviation $\|e_t\|_2^2$")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(rows: list[dict[str, str]], output_path: Path) -> None:
    steps = [int(row["step"]) for row in rows]
    residual = [float(row["residual_gen"]) for row in rows]
    volume = [float(row["volume_change"]) for row in rows]
    diversity = [float(row["diversity"]) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
    axes[0].plot(steps, residual, color="#2ca02c", linewidth=2.2)
    axes[0].set_title("Residual Generator")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel(r"$\|D_t^{(2)}-\widetilde D_t\|$")
    axes[0].grid(alpha=0.25)

    axes[1].plot(steps, volume, color="#ff7f0e", linewidth=2.2)
    axes[1].set_title("Volume Trace")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel(r"$|\mathrm{tr}(A_t^{res})|$")
    axes[1].grid(alpha=0.25)

    axes[2].plot(steps, diversity, color="#9467bd", linewidth=2.2)
    axes[2].set_title("Support Diversity")
    axes[2].set_xlabel("Generation")
    axes[2].set_ylabel("Effective Rank")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real ~0.1B transformer parameter-space experiment without data."
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--model-scale", choices=["0.1b", "0.2b"], default="0.1b")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    model_cfg = build_transformer_config(args.model_scale)
    model = TestTransformerLM(model_cfg).to(device)
    param_count = parameter_count(model)
    chart = build_parameter_chart(model)
    chart_t = chart_to_torch(chart, device)
    scene = SceneConfig()

    print(f"device={device}")
    print(f"transformer_params={param_count}")
    print(f"chart_dim={chart.dim}")

    setaware_controller = train_paper_setaware(chart=chart_t, scene=scene, seed=args.seed, device=device)
    rows = rollout_methods_torch(chart=chart_t, scene=scene, seed=args.seed, device=device, setaware_controller=setaware_controller)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.results_dir / "transformer_0_1b_metrics.csv",
        rows,
        [
            "step",
            "no_correction",
            "weight_only",
            "set_aware",
            "correction_only",
            "full_method",
            "no_correction_loss",
            "weight_only_loss",
            "set_aware_loss",
            "correction_only_loss",
            "full_method_loss",
            "residual_gen",
            "volume_change",
            "diversity",
            "best_full",
            "best_full_step",
        ],
    )
    plot_results(rows, args.results_dir / "transformer_0_1b_comparison.png")
    plot_losses(rows, args.results_dir / "transformer_0_1b_loss.png")
    plot_diagnostics(rows, args.results_dir / "transformer_0_1b_diagnostics.png")

    best_row = min(rows, key=lambda row: float(row["full_method"]))
    (args.results_dir / "run_summary.md").write_text(
        "\n".join(
            [
                "# Real Transformer Parameter-Space Summary",
                "",
                f"- device: `{device}`",
                f"- model_scale: `{args.model_scale}`",
                f"- transformer_params: `{param_count}`",
                f"- chart_dim: `{chart.dim}`",
                f"- parameter_groups: `{len(chart.groups)}`",
                f"- elapsed_sec: `{time.time() - start:.3f}`",
                f"- final_no_correction: `{rows[-1]['no_correction']}`",
                f"- final_weight_only: `{rows[-1]['weight_only']}`",
                f"- final_set_aware: `{rows[-1]['set_aware']}`",
                f"- final_correction_only: `{rows[-1]['correction_only']}`",
                f"- final_full_method: `{rows[-1]['full_method']}`",
                f"- final_set_aware_loss: `{rows[-1]['set_aware_loss']}`",
                f"- final_full_method_loss: `{rows[-1]['full_method_loss']}`",
                f"- best_full_method: `{best_row['full_method']}` at generation `{best_row['step']}`",
                f"- final_residual_gen: `{rows[-1]['residual_gen']}`",
                f"- final_volume_change: `{rows[-1]['volume_change']}`",
                f"- final_diversity: `{rows[-1]['diversity']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"results_dir={args.results_dir}")


if __name__ == "__main__":
    main()
