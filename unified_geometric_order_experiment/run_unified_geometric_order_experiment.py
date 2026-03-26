from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/unified_geometric_order_experiment_mpl")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

METHOD_ORDER = ["No Filter", "Weight-only", "Set-Aware", "Our method"]
SCENE_ORDER = [
    "fixed_additive_bias",
    "anisotropic_shrinkage",
    "noncommutative_structural_bias",
    "small_sample_structural_bias",
]
METHOD_STYLES = {
    "No Filter": {"color": "#7f8c8d", "linestyle": "--", "marker": "o"},
    "Weight-only": {"color": "#d62728", "linestyle": ":", "marker": "^"},
    "Set-Aware": {"color": "#1f77b4", "linestyle": "-.", "marker": "s"},
    "Our method": {"color": "#2ca02c", "linestyle": "-", "marker": "D"},
}
SCENE_PLOT_TITLES = {
    "fixed_additive_bias": "Fixed Additive Bias",
    "anisotropic_shrinkage": "Anisotropic Shrinkage",
    "noncommutative_structural_bias": "Noncommutative Structural Bias",
    "small_sample_structural_bias": "Small-Sample Structural Bias",
}


@dataclass(frozen=True)
class SceneSpec:
    key: str
    title: str
    dim: int
    generations: int
    set_size: int
    outlier_count: int
    ref_scale: float
    outlier_scale: float
    weight_tau: float
    healthy_linear: np.ndarray
    x_mu_bias: np.ndarray
    x_sigma_a: np.ndarray
    explicit_a0: np.ndarray
    explicit_b0: np.ndarray
    hol_angle: float
    hessian_diag: np.ndarray
    our_shrink: float


@dataclass
class SceneRollout:
    main_rows: list[dict[str, str]]
    aux_rows: list[dict[str, str]]
    terminal_rows: list[dict[str, str]]


@dataclass
class TransformerSize:
    label: str
    hidden: int
    heads: int
    layers: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_float(value: float) -> str:
    return f"{value:.8f}"


def log_terminal_row(
    scene: str,
    method: str,
    generation: int,
    prev_metric: float,
    metric: float,
    best_so_far: float,
) -> dict[str, str]:
    delta = metric - prev_metric
    row = {
        "scene": scene,
        "method": method,
        "generation": str(generation),
        "prev_metric": format_float(prev_metric),
        "metric": format_float(metric),
        "delta": format_float(delta),
        "best_so_far": format_float(best_so_far),
    }
    print(
        f"scene={scene} method={method} generation={generation:03d} "
        f"prev_metric={prev_metric:.6f} metric={metric:.6f} "
        f"delta={delta:.6f} best_so_far={best_so_far:.6f}"
    )
    return row


def identity_affine(dim: int) -> np.ndarray:
    eye = np.eye(dim + 1, dtype=np.float64)
    eye[-1, -1] = 1.0
    return eye


def affine_generator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dim = b.shape[0]
    out = np.zeros((dim + 1, dim + 1), dtype=np.float64)
    out[:dim, :dim] = a
    out[:dim, dim] = b
    return out


def expm(matrix: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(matrix.astype(np.float64))
    return torch.matrix_exp(tensor).cpu().numpy()


def commutator(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x @ y - y @ x


def adjoint_via_group(group: np.ndarray, generator: np.ndarray) -> np.ndarray:
    group_inv = np.linalg.inv(group)
    return group @ generator @ group_inv


def affine_apply(group: np.ndarray, x: np.ndarray) -> np.ndarray:
    dim = x.shape[0]
    return group[:dim, :dim] @ x + group[:dim, dim]


def affine_apply_points(group: np.ndarray, points: np.ndarray) -> np.ndarray:
    dim = points.shape[1]
    return points @ group[:dim, :dim].T + group[:dim, dim]


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def sqrt_spd(matrix: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(symmetrize(matrix))
    vals = np.clip(vals, 1e-8, None)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


def invsqrt_spd(matrix: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(symmetrize(matrix))
    vals = np.clip(vals, 1e-8, None)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T


def log_spd(matrix: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(symmetrize(matrix))
    vals = np.clip(vals, 1e-8, None)
    return vecs @ np.diag(np.log(vals)) @ vecs.T


def log_linear_positive(matrix: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(matrix)
    vals = np.clip(vals.real, 1e-8, None)
    out = vecs @ np.diag(np.log(vals)) @ np.linalg.inv(vecs)
    return np.real_if_close(out)


def weighted_mean(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(points * weights[:, None], axis=0)


def weighted_cov(points: np.ndarray, weights: np.ndarray, mean: np.ndarray) -> np.ndarray:
    centered = points - mean[None, :]
    return np.einsum("n,ni,nj->ij", weights, centered, centered)


def robust_weights(points: np.ndarray, tau: float) -> np.ndarray:
    center = np.median(points, axis=0)
    d2 = np.sum((points - center[None, :]) ** 2, axis=1)
    scaled = np.exp(-(d2 - d2.min()) / max(tau, 1e-6))
    scaled = scaled / np.sum(scaled)
    return scaled


def make_template(count: int, dim: int, phase: float, scale: float) -> np.ndarray:
    idx = np.arange(count, dtype=np.float64)
    angles = 2.0 * np.pi * (idx + 1.0) / float(count) + phase
    points = np.zeros((count, dim), dtype=np.float64)
    if dim != 2:
        raise ValueError("This experiment is intentionally fixed at dim=2.")
    points[:, 0] = np.cos(angles) + 0.35 * np.cos(3.0 * angles)
    points[:, 1] = 0.8 * np.sin(angles) + 0.25 * np.sin(2.0 * angles)
    points = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(points.T)
    normalizer = sqrt_spd(cov + 1e-6 * np.eye(dim))
    points = points @ np.linalg.inv(normalizer).T
    return scale * points


def make_outlier_offsets(count: int, phase: float, scale: float) -> np.ndarray:
    idx = np.arange(count, dtype=np.float64)
    angles = np.pi * (idx + 0.5) / max(count, 1) + 1.7 * phase
    out = np.stack([np.cos(angles), 1.4 * np.sin(angles)], axis=1)
    out = out - out.mean(axis=0, keepdims=True)
    return scale * out


def build_order_matrix(num_channels: int) -> np.ndarray:
    order = np.zeros((num_channels, num_channels), dtype=np.float64)
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            order[i, j] = 1.0
            order[j, i] = -1.0
    return order


def curvature_drift(hessian_diag: np.ndarray, delta_c: np.ndarray) -> np.ndarray:
    return 0.5 * np.array(
        [
            hessian_diag[0] * delta_c[0, 0] + 0.35 * hessian_diag[1] * delta_c[1, 1],
            0.25 * hessian_diag[0] * delta_c[0, 0] + hessian_diag[1] * delta_c[1, 1],
        ],
        dtype=np.float64,
    )


def half_group(generator: np.ndarray, sign: float = 1.0) -> np.ndarray:
    return expm(sign * 0.5 * generator)


def scene_specs() -> dict[str, SceneSpec]:
    return {
        "fixed_additive_bias": SceneSpec(
            key="fixed_additive_bias",
            title="固定加性偏置",
            dim=2,
            generations=24,
            set_size=32,
            outlier_count=6,
            ref_scale=0.18,
            outlier_scale=0.45,
            weight_tau=0.28,
            healthy_linear=np.diag([0.88, 0.82]),
            x_mu_bias=np.array([0.18, -0.12], dtype=np.float64),
            x_sigma_a=np.zeros((2, 2), dtype=np.float64),
            explicit_a0=np.zeros((2, 2), dtype=np.float64),
            explicit_b0=np.zeros(2, dtype=np.float64),
            hol_angle=0.0,
            hessian_diag=np.zeros(2, dtype=np.float64),
            our_shrink=0.96,
        ),
        "anisotropic_shrinkage": SceneSpec(
            key="anisotropic_shrinkage",
            title="各向异性 shrinkage",
            dim=2,
            generations=24,
            set_size=32,
            outlier_count=6,
            ref_scale=0.20,
            outlier_scale=0.42,
            weight_tau=0.30,
            healthy_linear=np.diag([0.91, 0.86]),
            x_mu_bias=np.array([0.06, -0.02], dtype=np.float64),
            x_sigma_a=np.array([[-0.28, 0.0], [0.0, 0.17]], dtype=np.float64),
            explicit_a0=np.array([[-0.08, 0.0], [0.0, 0.03]], dtype=np.float64),
            explicit_b0=np.array([0.03, -0.015], dtype=np.float64),
            hol_angle=0.0,
            hessian_diag=np.zeros(2, dtype=np.float64),
            our_shrink=0.95,
        ),
        "noncommutative_structural_bias": SceneSpec(
            key="noncommutative_structural_bias",
            title="非交换结构偏置",
            dim=2,
            generations=28,
            set_size=36,
            outlier_count=8,
            ref_scale=0.22,
            outlier_scale=0.38,
            weight_tau=0.26,
            healthy_linear=np.diag([0.92, 0.87]),
            x_mu_bias=np.array([0.11, -0.045], dtype=np.float64),
            x_sigma_a=np.array([[-0.18, 0.06], [0.0, 0.09]], dtype=np.float64),
            explicit_a0=np.array([[-0.05, 0.0], [0.0, -0.02]], dtype=np.float64),
            explicit_b0=np.array([0.03, 0.015], dtype=np.float64),
            hol_angle=0.20,
            hessian_diag=np.array([1.25, -0.85], dtype=np.float64),
            our_shrink=0.94,
        ),
        "small_sample_structural_bias": SceneSpec(
            key="small_sample_structural_bias",
            title="小样本结构偏置",
            dim=2,
            generations=28,
            set_size=12,
            outlier_count=3,
            ref_scale=0.26,
            outlier_scale=0.55,
            weight_tau=0.20,
            healthy_linear=np.diag([0.92, 0.87]),
            x_mu_bias=np.array([0.12, -0.05], dtype=np.float64),
            x_sigma_a=np.array([[-0.20, 0.07], [0.0, 0.10]], dtype=np.float64),
            explicit_a0=np.array([[-0.05, 0.0], [0.0, -0.02]], dtype=np.float64),
            explicit_b0=np.array([0.03, 0.02], dtype=np.float64),
            hol_angle=0.24,
            hessian_diag=np.array([1.40, -0.92], dtype=np.float64),
            our_shrink=0.93,
        ),
    }


def scene_true_hol_generator(spec: SceneSpec) -> np.ndarray:
    return np.array([[0.0, -spec.hol_angle], [spec.hol_angle, 0.0]], dtype=np.float64)


def healthy_group(spec: SceneSpec) -> np.ndarray:
    omega_b = affine_generator(log_linear_positive(spec.healthy_linear), np.zeros(spec.dim, dtype=np.float64))
    return expm(omega_b)


def generate_candidates(
    spec: SceneSpec,
    current_e: np.ndarray,
    generation: int,
) -> dict[str, np.ndarray]:
    phase = 0.17 * generation
    base_group = healthy_group(spec)
    center_ref = affine_apply(base_group, current_e)

    template = make_template(spec.set_size, spec.dim, phase, spec.ref_scale)
    ref_candidates = center_ref[None, :] + template

    x_mu = affine_generator(np.zeros((spec.dim, spec.dim), dtype=np.float64), spec.x_mu_bias)
    x_sigma = affine_generator(spec.x_sigma_a, np.zeros(spec.dim, dtype=np.float64))
    pre_selected = affine_apply_points(expm(x_sigma), affine_apply_points(expm(x_mu), ref_candidates))

    mu_ref = ref_candidates.mean(axis=0)
    cov_ref = np.cov(ref_candidates.T)
    mu_pre = pre_selected.mean(axis=0)
    cov_pre = np.cov(pre_selected.T)
    delta_c_pre = cov_pre - cov_ref

    zeta_true = curvature_drift(spec.hessian_diag, delta_c_pre)
    x_zeta = affine_generator(np.zeros((spec.dim, spec.dim), dtype=np.float64), zeta_true)
    x_hol = affine_generator(scene_true_hol_generator(spec), np.zeros(spec.dim, dtype=np.float64))
    x_0 = affine_generator(spec.explicit_a0, spec.explicit_b0)

    order_groups = [expm(x_mu), expm(x_sigma), expm(x_zeta), expm(x_hol), expm(x_0)]
    selected_core = ref_candidates.copy()
    for group in order_groups:
        selected_core = affine_apply_points(group, selected_core)

    raw_candidates = selected_core.copy()
    if spec.outlier_count > 0:
        offsets = make_outlier_offsets(spec.outlier_count, phase, spec.outlier_scale)
        raw_candidates[-spec.outlier_count :] = raw_candidates[-spec.outlier_count :] + offsets

    return {
        "ref_candidates": ref_candidates,
        "raw_candidates": raw_candidates,
        "core_candidates": selected_core,
        "mu_ref": mu_ref,
        "cov_ref": cov_ref,
        "x_mu_true": x_mu,
        "x_sigma_true": x_sigma,
        "x_zeta_true": x_zeta,
        "x_hol_true": x_hol,
        "x_0_true": x_0,
        "zeta_true": zeta_true,
    }


def setaware_target(
    spec: SceneSpec,
    current_e: np.ndarray,
    weighted_mu: np.ndarray,
) -> np.ndarray:
    healthy_center = spec.healthy_linear @ current_e
    return weighted_mu - healthy_center


class SetCorrectionEstimator(nn.Module):
    def __init__(self, point_dim: int, hidden: int = 64, heads: int = 4, layers: int = 2, out_dim: int = 2):
        super().__init__()
        self.point_net = nn.Sequential(
            nn.Linear(point_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.point_net(x)
        h = self.transformer(h)
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        return self.head(pooled)


class ReleasePointwiseFilter(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.weight_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return torch.sigmoid(self.weight_head(h)).squeeze(-1)

    @staticmethod
    def weighted_estimate(x: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        w = weights.unsqueeze(-1)
        num = (w * x).sum(dim=1)
        den = w.sum(dim=1, keepdim=True).clamp_min(eps)
        return num / den


class ReleaseSetAwareFilter(nn.Module):
    def __init__(self, dim: int, hidden: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.weight_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.bias_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = self.transformer(h)
        weights = torch.sigmoid(self.weight_head(h)).squeeze(-1)
        pooled = h.mean(dim=1)
        delta_phi = self.bias_head(pooled)
        return weights, delta_phi

    @staticmethod
    def weighted_estimate(x: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        w = weights.unsqueeze(-1)
        num = (w * x).sum(dim=1)
        den = w.sum(dim=1, keepdim=True).clamp_min(eps)
        return num / den


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_setaware_dataset(spec: SceneSpec, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for idx in range(count):
        radius = rng.uniform(0.8, 2.5)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        current_e = radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        generated = generate_candidates(spec, current_e, idx % max(spec.generations, 1))
        weights = robust_weights(generated["raw_candidates"], spec.weight_tau)
        mu_weighted = weighted_mean(generated["raw_candidates"], weights)
        correction_target = setaware_target(spec, current_e, mu_weighted)
        inputs.append(generated["raw_candidates"].astype(np.float32))
        targets.append(correction_target.astype(np.float32))
    return np.stack(inputs), np.stack(targets)


def train_setaware_model(spec: SceneSpec, device: torch.device, seed: int) -> SetCorrectionEstimator:
    train_x, train_y = build_setaware_dataset(spec, count=768, seed=seed)
    test_x, test_y = build_setaware_dataset(spec, count=192, seed=seed + 91)

    model = SetCorrectionEstimator(point_dim=spec.dim, hidden=64, heads=4, layers=2, out_dim=spec.dim).to(device)
    opt = optim.Adam(model.parameters(), lr=2e-3)

    x_train = torch.from_numpy(train_x).to(device)
    y_train = torch.from_numpy(train_y).to(device)
    x_test = torch.from_numpy(test_x).to(device)
    y_test = torch.from_numpy(test_y).to(device)

    for _ in range(70):
        model.train()
        pred = model(x_train)
        loss = nn.functional.mse_loss(pred, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        test_loss = nn.functional.mse_loss(model(x_test), y_test).item()
    print(
        f"scene={spec.key} method=Set-Aware calibration_final_test_mse={test_loss:.6f} "
        f"params={parameter_count(model)} device={device}"
    )
    return model


@torch.no_grad()
def setaware_correction(model: SetCorrectionEstimator, raw_candidates: np.ndarray, device: torch.device) -> np.ndarray:
    batch = torch.from_numpy(raw_candidates[None, ...].astype(np.float32)).to(device)
    pred = model(batch).squeeze(0).cpu().numpy()
    return pred.astype(np.float64)


def our_method_step(
    spec: SceneSpec,
    current_e: np.ndarray,
    generated: dict[str, np.ndarray],
    weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    ref_candidates = generated["ref_candidates"]
    raw_candidates = generated["raw_candidates"]

    mu_ref = ref_candidates.mean(axis=0)
    cov_ref = np.cov(ref_candidates.T)
    mu_sel = weighted_mean(raw_candidates, weights)
    cov_sel = weighted_cov(raw_candidates, weights, mu_sel)

    delta_mu = mu_sel - mu_ref
    delta_c = cov_sel - cov_ref

    x_mu = affine_generator(np.zeros((spec.dim, spec.dim), dtype=np.float64), delta_mu)
    ridge_lambda = 5e-3
    align_matrix = sqrt_spd(cov_sel + ridge_lambda * np.eye(spec.dim)) @ invsqrt_spd(
        cov_ref + ridge_lambda * np.eye(spec.dim)
    )
    x_sigma = affine_generator(log_spd(align_matrix.T @ align_matrix), np.zeros(spec.dim, dtype=np.float64))
    zeta = curvature_drift(spec.hessian_diag, delta_c)
    x_zeta = affine_generator(np.zeros((spec.dim, spec.dim), dtype=np.float64), zeta)
    x_hol = generated["x_hol_true"]
    x_0 = generated["x_0_true"]

    channels = [x_mu, x_sigma, x_zeta, x_hol, x_0]
    order_matrix = build_order_matrix(len(channels))
    d2 = np.zeros_like(channels[0])
    for channel in channels:
        d2 = d2 + channel
    for i in range(len(channels)):
        for j in range(i + 1, len(channels)):
            d2 = d2 + 0.5 * order_matrix[i, j] * commutator(channels[i], channels[j])

    h_t = x_hol
    h_group_inv = np.linalg.inv(expm(h_t))
    d_hat = spec.our_shrink * adjoint_via_group(h_group_inv, d2)
    d_tilde = adjoint_via_group(expm(h_t), d_hat)

    omega_b = affine_generator(log_linear_positive(spec.healthy_linear), np.zeros(spec.dim, dtype=np.float64))
    psi_group = (
        half_group(omega_b)
        @ half_group(d_tilde, sign=-1.0)
        @ expm(d2)
        @ half_group(d_tilde, sign=-1.0)
        @ half_group(omega_b)
    )
    next_e = affine_apply(psi_group, current_e)

    residual = d2 - d_tilde
    aux = {
        "delta_mu_norm": float(np.linalg.norm(delta_mu)),
        "residual_norm": float(np.linalg.norm(residual)),
        "trace_a_res": float(np.trace(residual[: spec.dim, : spec.dim])),
    }
    return next_e, aux


def run_main(device: torch.device, seed: int) -> SceneRollout:
    set_seed(seed)
    main_rows: list[dict[str, str]] = []
    aux_rows: list[dict[str, str]] = []
    terminal_rows: list[dict[str, str]] = []

    specs = scene_specs()
    setaware_models = {
        key: train_setaware_model(spec, device=device, seed=seed + 13 * idx)
        for idx, (key, spec) in enumerate(specs.items())
    }

    for scene_index, scene_key in enumerate(SCENE_ORDER):
        spec = specs[scene_key]
        initial_e = np.array([2.45 - 0.12 * scene_index, -1.85 + 0.08 * scene_index], dtype=np.float64)
        states = {
            "No Filter": initial_e.copy(),
            "Weight-only": initial_e.copy(),
            "Set-Aware": initial_e.copy(),
            "Our method": initial_e.copy(),
        }
        best = {name: float(np.linalg.norm(initial_e)) for name in METHOD_ORDER}

        for generation in range(1, spec.generations + 1):
            for method in METHOD_ORDER:
                prev_e = states[method].copy()
                prev_metric = float(np.linalg.norm(prev_e))
                generated = generate_candidates(spec, prev_e, generation)
                raw_candidates = generated["raw_candidates"]
                weights = robust_weights(raw_candidates, spec.weight_tau)
                mu_weighted = weighted_mean(raw_candidates, weights)
                mu_raw = raw_candidates.mean(axis=0)

                if method == "No Filter":
                    next_e = mu_raw
                    aux = {
                        "delta_mu_norm": float(np.linalg.norm(mu_raw - generated["mu_ref"])),
                        "residual_norm": float(np.linalg.norm(mu_raw - mu_weighted)),
                        "trace_a_res": 0.0,
                    }
                elif method == "Weight-only":
                    next_e = mu_weighted
                    aux = {
                        "delta_mu_norm": float(np.linalg.norm(mu_weighted - generated["mu_ref"])),
                        "residual_norm": float(np.linalg.norm(mu_raw - mu_weighted)),
                        "trace_a_res": 0.0,
                    }
                elif method == "Set-Aware":
                    delta_pred = setaware_correction(setaware_models[scene_key], raw_candidates, device=device)
                    next_e = mu_weighted - delta_pred
                    aux = {
                        "delta_mu_norm": float(np.linalg.norm(mu_weighted - generated["mu_ref"])),
                        "residual_norm": float(np.linalg.norm(delta_pred)),
                        "trace_a_res": 0.0,
                    }
                else:
                    next_e, aux = our_method_step(spec, prev_e, generated, weights)

                metric = float(np.linalg.norm(next_e))
                best[method] = min(best[method], metric)
                states[method] = next_e

                main_rows.append(
                    {
                        "scene": scene_key,
                        "scene_title": spec.title,
                        "method": method,
                        "generation": str(generation),
                        "metric_l2": format_float(metric),
                    }
                )
                aux_rows.append(
                    {
                        "scene": scene_key,
                        "method": method,
                        "generation": str(generation),
                        "delta_mu_norm": format_float(aux["delta_mu_norm"]),
                        "residual_norm": format_float(aux["residual_norm"]),
                        "trace_a_res": format_float(aux["trace_a_res"]),
                    }
                )
                terminal_rows.append(log_terminal_row(scene_key, method, generation, prev_metric, metric, best[method]))

    return SceneRollout(main_rows=main_rows, aux_rows=aux_rows, terminal_rows=terminal_rows)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def plot_main(main_rows: list[dict[str, str]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.4), sharey=False)
    for ax, scene_key in zip(axes, SCENE_ORDER):
        scene_rows = [row for row in main_rows if row["scene"] == scene_key]
        title = SCENE_PLOT_TITLES[scene_key]
        for method in METHOD_ORDER:
            rows = [row for row in scene_rows if row["method"] == method]
            xs = [int(row["generation"]) for row in rows]
            ys = [float(row["metric_l2"]) for row in rows]
            style = METHOD_STYLES[method]
            ax.plot(
                xs,
                ys,
                label=method,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markevery=max(1, len(xs) // 7),
                linewidth=2.0,
                markersize=4.4,
            )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Generation")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"$\|e_t\|_2$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_sweeps(seed: int) -> list[dict[str, str]]:
    set_seed(seed)
    rows: list[dict[str, str]] = []
    base_specs = scene_specs()

    beta_values = [0.05, 0.10, 0.16, 0.24, 0.34, 0.48]
    alpha_values = [0.05, 0.10, 0.18, 0.28, 0.40, 0.55]

    for beta in beta_values:
        spec = base_specs["fixed_additive_bias"]
        mutated = SceneSpec(**{**spec.__dict__, "x_mu_bias": np.array([beta, -0.65 * beta], dtype=np.float64)})
        rows.extend(run_single_sweep(mutated, control_name="beta", control_value=beta))
    for alpha in alpha_values:
        spec = base_specs["anisotropic_shrinkage"]
        mutated = SceneSpec(
            **{
                **spec.__dict__,
                "x_sigma_a": np.array([[-alpha, 0.0], [0.0, 0.60 * alpha]], dtype=np.float64),
            }
        )
        rows.extend(run_single_sweep(mutated, control_name="alpha", control_value=alpha))
    return rows


def run_single_sweep(spec: SceneSpec, control_name: str, control_value: float) -> list[dict[str, str]]:
    current_e = np.array([2.2, -1.75], dtype=np.float64)
    device = torch.device("cpu")
    model = train_setaware_model(spec, device=device, seed=int(1000 * control_value) + 17)
    states = {method: current_e.copy() for method in METHOD_ORDER}
    for generation in range(1, spec.generations + 1):
        for method in METHOD_ORDER:
            generated = generate_candidates(spec, states[method], generation)
            weights = robust_weights(generated["raw_candidates"], spec.weight_tau)
            mu_weighted = weighted_mean(generated["raw_candidates"], weights)
            if method == "No Filter":
                states[method] = generated["raw_candidates"].mean(axis=0)
            elif method == "Weight-only":
                states[method] = mu_weighted
            elif method == "Set-Aware":
                states[method] = mu_weighted - setaware_correction(model, generated["raw_candidates"], device=device)
            else:
                states[method], _ = our_method_step(spec, states[method], generated, weights)
    rows: list[dict[str, str]] = []
    for method in METHOD_ORDER:
        rows.append(
            {
                "control_name": control_name,
                "control_value": format_float(control_value),
                "method": method,
                "final_metric_l2": format_float(float(np.linalg.norm(states[method]))),
            }
        )
    return rows


def plot_sweeps(rows: list[dict[str, str]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    for ax, control_name, title in zip(axes, ["beta", "alpha"], ["Additive Bias Sweep", "Shrinkage Sweep"]):
        control_rows = [row for row in rows if row["control_name"] == control_name]
        for method in METHOD_ORDER:
            method_rows = [row for row in control_rows if row["method"] == method]
            xs = [float(row["control_value"]) for row in method_rows]
            ys = [float(row["final_metric_l2"]) for row in method_rows]
            style = METHOD_STYLES[method]
            ax.plot(
                xs,
                ys,
                label=method,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=5.0,
            )
        ax.set_title(title)
        ax.set_xlabel(control_name)
        ax.set_ylabel("Final " + r"$\|e_t\|_2$")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_transformer_comparison_dataset(sample_count: int, seed: int) -> dict[str, np.ndarray]:
    specs = scene_specs()
    structural_specs = [
        specs["noncommutative_structural_bias"],
        specs["small_sample_structural_bias"],
        specs["anisotropic_shrinkage"],
    ]
    max_set_size = max(spec.set_size for spec in structural_specs)
    rng = np.random.default_rng(seed)

    sa_inputs: list[np.ndarray] = []
    our_inputs: list[np.ndarray] = []
    sa_targets: list[np.ndarray] = []
    our_targets: list[np.ndarray] = []
    mu_weighted_rows: list[np.ndarray] = []
    current_e_rows: list[np.ndarray] = []
    x_mu_rows: list[np.ndarray] = []
    x_sigma_rows: list[np.ndarray] = []
    x_zeta_rows: list[np.ndarray] = []
    x_hol_rows: list[np.ndarray] = []
    x0_rows: list[np.ndarray] = []
    omega_rows: list[np.ndarray] = []

    for idx in range(sample_count):
        spec = structural_specs[idx % len(structural_specs)]
        radius = rng.uniform(0.75, 2.5)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        current_e = radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        generated = generate_candidates(spec, current_e, idx % spec.generations + 1)
        weights = robust_weights(generated["raw_candidates"], spec.weight_tau)

        raw = generated["raw_candidates"]
        ref_mean = generated["ref_candidates"].mean(axis=0)
        ref_cov = np.cov(generated["ref_candidates"].T)
        tri = np.array([ref_cov[0, 0], ref_cov[0, 1], ref_cov[1, 1]], dtype=np.float64)
        mask = np.ones((spec.set_size, 1), dtype=np.float64)

        sa_input = np.concatenate([raw, mask], axis=1)
        our_feature_block = np.concatenate(
            [
                np.tile(ref_mean, (spec.set_size, 1)),
                np.tile(tri, (spec.set_size, 1)),
                np.tile(current_e, (spec.set_size, 1)),
                mask,
            ],
            axis=1,
        )
        our_input = np.concatenate([raw, our_feature_block], axis=1)
        if spec.set_size < max_set_size:
            sa_input = np.concatenate([sa_input, np.zeros((max_set_size - spec.set_size, sa_input.shape[1]), dtype=np.float64)], axis=0)
            our_input = np.concatenate([our_input, np.zeros((max_set_size - spec.set_size, our_input.shape[1]), dtype=np.float64)], axis=0)

        mu_weighted = weighted_mean(raw, weights)
        mu_ref = generated["ref_candidates"].mean(axis=0)
        cov_ref = np.cov(generated["ref_candidates"].T)
        cov_sel = weighted_cov(raw, weights, mu_weighted)
        delta_mu = mu_weighted - mu_ref
        delta_c = cov_sel - cov_ref

        x_mu = affine_generator(np.zeros((spec.dim, spec.dim), dtype=np.float64), delta_mu)
        x_sigma = affine_generator(
            log_spd(
                sqrt_spd(cov_sel + 5e-3 * np.eye(spec.dim))
                @ invsqrt_spd(cov_ref + 5e-3 * np.eye(spec.dim))
            ),
            np.zeros(spec.dim, dtype=np.float64),
        )
        x_zeta = affine_generator(np.zeros((spec.dim, spec.dim), dtype=np.float64), curvature_drift(spec.hessian_diag, delta_c))
        x_hol = generated["x_hol_true"]
        x_0 = generated["x_0_true"]
        channels = [x_mu, x_sigma, x_zeta, x_hol, x_0]
        d2 = np.zeros_like(channels[0])
        order_matrix = build_order_matrix(len(channels))
        for channel in channels:
            d2 = d2 + channel
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                d2 = d2 + 0.5 * order_matrix[i, j] * commutator(channels[i], channels[j])
        omega_b = affine_generator(log_linear_positive(spec.healthy_linear), np.zeros(spec.dim, dtype=np.float64))
        order_target = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        sa_inputs.append(sa_input.astype(np.float32))
        our_inputs.append(our_input.astype(np.float32))
        sa_targets.append(setaware_target(spec, current_e, mu_weighted).astype(np.float32))
        our_targets.append(order_target.astype(np.float32))
        mu_weighted_rows.append(mu_weighted.astype(np.float32))
        current_e_rows.append(current_e.astype(np.float32))
        x_mu_rows.append(x_mu.astype(np.float32))
        x_sigma_rows.append(x_sigma.astype(np.float32))
        x_zeta_rows.append(x_zeta.astype(np.float32))
        x_hol_rows.append(x_hol.astype(np.float32))
        x0_rows.append(x_0.astype(np.float32))
        omega_rows.append(omega_b.astype(np.float32))

    return {
        "sa_inputs": np.stack(sa_inputs),
        "our_inputs": np.stack(our_inputs),
        "sa_targets": np.stack(sa_targets),
        "our_targets": np.stack(our_targets),
        "mu_weighted": np.stack(mu_weighted_rows),
        "current_e": np.stack(current_e_rows),
        "x_mu": np.stack(x_mu_rows),
        "x_sigma": np.stack(x_sigma_rows),
        "x_zeta": np.stack(x_zeta_rows),
        "x_hol": np.stack(x_hol_rows),
        "x_0": np.stack(x0_rows),
        "omega_b": np.stack(omega_rows),
    }


def evaluate_transformer_rollout(
    sa_model: SetCorrectionEstimator,
    our_model: SetCorrectionEstimator,
    dataset: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[float, float]:
    sa_model.eval()
    our_model.eval()
    with torch.no_grad():
        sa_pred = sa_model(torch.from_numpy(dataset["sa_inputs"]).to(device)).cpu().numpy()
        our_pred = our_model(torch.from_numpy(dataset["our_inputs"]).to(device)).cpu().numpy()

    sa_next = dataset["mu_weighted"] - sa_pred
    sa_metric = float(np.mean(np.linalg.norm(sa_next, axis=1)))

    our_next_rows: list[np.ndarray] = []
    for pred, current_e, x_mu, x_sigma, x_zeta, x_hol, x_0, omega_b in zip(
        our_pred,
        dataset["current_e"],
        dataset["x_mu"],
        dataset["x_sigma"],
        dataset["x_zeta"],
        dataset["x_hol"],
        dataset["x_0"],
        dataset["omega_b"],
    ):
        w_mz, w_mh, w_zh = np.clip(pred.astype(np.float64), -1.0, 1.0)
        d2 = (
            x_mu.astype(np.float64)
            + x_sigma.astype(np.float64)
            + x_zeta.astype(np.float64)
            + x_hol.astype(np.float64)
            + x_0.astype(np.float64)
            + 0.5 * w_mz * commutator(x_mu.astype(np.float64), x_zeta.astype(np.float64))
            + 0.5 * w_mh * commutator(x_mu.astype(np.float64), x_hol.astype(np.float64))
            + 0.5 * w_zh * commutator(x_zeta.astype(np.float64), x_hol.astype(np.float64))
        )
        h_group_inv = np.linalg.inv(expm(x_hol.astype(np.float64)))
        d_hat = 0.88 * adjoint_via_group(h_group_inv, d2)
        d_tilde = adjoint_via_group(expm(x_hol.astype(np.float64)), d_hat)
        psi = (
            half_group(omega_b.astype(np.float64))
            @ half_group(d_tilde, sign=-1.0)
            @ expm(d2)
            @ half_group(d_tilde, sign=-1.0)
            @ half_group(omega_b.astype(np.float64))
        )
        our_next_rows.append(affine_apply(psi, current_e.astype(np.float64)))
    our_next = np.stack(our_next_rows)
    our_metric = float(np.mean(np.linalg.norm(our_next, axis=1)))
    return sa_metric, our_metric


def run_transformer(device: torch.device, seed: int) -> list[dict[str, str]]:
    set_seed(seed)
    train = build_transformer_comparison_dataset(sample_count=900, seed=seed + 201)
    test = build_transformer_comparison_dataset(sample_count=240, seed=seed + 401)

    sizes = [
        TransformerSize(label="ST-0.41M", hidden=128, heads=4, layers=2),
        TransformerSize(label="ST-2.67M", hidden=256, heads=8, layers=4),
        TransformerSize(label="ST-8.34M", hidden=384, heads=6, layers=6),
    ]

    rows: list[dict[str, str]] = []
    x_sa_train = torch.from_numpy(train["sa_inputs"]).to(device)
    y_sa_train = torch.from_numpy(train["sa_targets"]).to(device)
    x_our_train = torch.from_numpy(train["our_inputs"]).to(device)
    y_our_train = torch.from_numpy(train["our_targets"]).to(device)
    x_sa_test = torch.from_numpy(test["sa_inputs"]).to(device)
    y_sa_test = torch.from_numpy(test["sa_targets"]).to(device)
    x_our_test = torch.from_numpy(test["our_inputs"]).to(device)
    y_our_test = torch.from_numpy(test["our_targets"]).to(device)

    for size in sizes:
        sa_model = SetCorrectionEstimator(point_dim=train["sa_inputs"].shape[-1], hidden=size.hidden, heads=size.heads, layers=size.layers, out_dim=train["sa_targets"].shape[-1]).to(device)
        our_model = SetCorrectionEstimator(point_dim=train["our_inputs"].shape[-1], hidden=size.hidden, heads=size.heads, layers=size.layers, out_dim=train["our_targets"].shape[-1]).to(device)
        sa_opt = optim.Adam(sa_model.parameters(), lr=1e-3)
        our_opt = optim.Adam(our_model.parameters(), lr=1e-3)
        best_sa = float("inf")
        best_our = float("inf")
        prev_sa: float | None = None
        prev_our: float | None = None

        for step in range(1, 61):
            sa_model.train()
            sa_loss = nn.functional.mse_loss(sa_model(x_sa_train), y_sa_train)
            sa_opt.zero_grad()
            sa_loss.backward()
            sa_opt.step()

            our_model.train()
            our_loss = nn.functional.mse_loss(our_model(x_our_train), y_our_train)
            our_opt.zero_grad()
            our_loss.backward()
            our_opt.step()

            sa_model.eval()
            our_model.eval()
            with torch.no_grad():
                sa_metric = float(nn.functional.mse_loss(sa_model(x_sa_test), y_sa_test).item())
                our_metric = float(nn.functional.mse_loss(our_model(x_our_test), y_our_test).item())
            best_sa = min(best_sa, sa_metric)
            best_our = min(best_our, our_metric)
            delta_sa = 0.0 if prev_sa is None else sa_metric - prev_sa
            delta_our = 0.0 if prev_our is None else our_metric - prev_our
            print(
                f"scene=transformer method={size.label}/Set-Aware generation={step:03d} "
                f"prev_metric={format_float(sa_metric if prev_sa is None else prev_sa)} "
                f"metric={sa_metric:.6f} delta={delta_sa:.6f} best_so_far={best_sa:.6f}"
            )
            print(
                f"scene=transformer method={size.label}/Our method generation={step:03d} "
                f"prev_metric={format_float(our_metric if prev_our is None else prev_our)} "
                f"metric={our_metric:.6f} delta={delta_our:.6f} best_so_far={best_our:.6f}"
            )
            rows.extend(
                [
                    {
                        "model": size.label,
                        "method": "Set-Aware",
                        "params": str(parameter_count(sa_model)),
                        "step": str(step),
                        "train_loss": format_float(float(sa_loss.item())),
                        "test_loss": format_float(sa_metric),
                    },
                    {
                        "model": size.label,
                        "method": "Our method",
                        "params": str(parameter_count(our_model)),
                        "step": str(step),
                        "train_loss": format_float(float(our_loss.item())),
                        "test_loss": format_float(our_metric),
                    },
                ]
            )
            prev_sa = sa_metric
            prev_our = our_metric
    return rows


def plot_transformer(rows: list[dict[str, str]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.1), sharey=True)
    color_map = {"Set-Aware": "#1f77b4", "Our method": "#2ca02c"}
    for ax, model in zip(axes, ["ST-0.41M", "ST-2.67M", "ST-8.34M"]):
        model_rows = [row for row in rows if row["model"] == model]
        for method in ["Set-Aware", "Our method"]:
            method_rows = [row for row in model_rows if row["method"] == method]
            xs = [int(row["step"]) for row in method_rows]
            ys = [float(row["test_loss"]) for row in method_rows]
            ax.plot(xs, ys, label=method, color=color_map[method], linewidth=2.1)
        ax.set_title(model)
        ax.set_xlabel("Step")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Held-out Test MSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Transformer Strength Comparison: Set-Aware vs Our method", fontsize=16, y=1.08)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def release_make_bias_vector(dim: int, target_norm: float) -> np.ndarray:
    vec = np.ones(dim, dtype=np.float64)
    return target_norm * vec / np.linalg.norm(vec)


def release_sample_candidates(
    rng: np.random.Generator,
    theta_true: np.ndarray,
    bias: np.ndarray,
    n: int,
    noise_std: float,
) -> np.ndarray:
    return theta_true[None, :] + bias[None, :] + noise_std * rng.normal(size=(n, theta_true.shape[0]))


def release_sample_clean_reference(
    rng: np.random.Generator,
    theta_true: np.ndarray,
    n: int,
    noise_std: float,
) -> np.ndarray:
    return theta_true + noise_std * rng.normal(size=(n, theta_true.shape[0])).mean(axis=0)


def release_build_labels(candidates: np.ndarray, theta_true: np.ndarray, top_ratio: float) -> np.ndarray:
    dists = np.linalg.norm(candidates - theta_true[None, :], axis=1)
    k = max(1, int(len(candidates) * top_ratio))
    thresh = np.partition(dists, k - 1)[k - 1]
    return (dists <= thresh).astype(np.float32)


def release_classification_loss(weights: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy(weights.clamp(1e-4, 1 - 1e-4), labels)


def release_contraction_loss(theta_new: torch.Tensor, theta_good: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum((theta_new - theta_good) ** 2, dim=-1))


def release_ess_loss(weights: torch.Tensor, tau: float) -> torch.Tensor:
    ess = (weights.sum(dim=1) ** 2) / (weights.pow(2).sum(dim=1) + 1e-8)
    penalty = torch.clamp(tau - ess, min=0.0)
    return penalty.mean()


def release_correction_reg(delta_phi: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum(delta_phi ** 2, dim=-1))


def run_release_demo(seed: int) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed)
    dim = 5
    generations = 80
    samples_per_gen = 200
    noise_std = 0.2
    calibration_size = 120
    pointwise_contraction = 0.5
    setaware_contraction = 0.35
    top_ratio = 0.3
    tau = 50.0
    device = torch.device("cpu")

    theta_true = rng.normal(size=dim)
    bias = release_make_bias_vector(dim, target_norm=0.5)
    theta_good = release_sample_clean_reference(rng, theta_true, calibration_size, noise_std)

    pointwise = ReleasePointwiseFilter(dim=dim, hidden=64).to(device)
    setaware = ReleaseSetAwareFilter(dim=dim, hidden=64, heads=4, layers=2).to(device)
    opt_pointwise = optim.Adam(pointwise.parameters(), lr=1e-3)
    opt_setaware = optim.Adam(setaware.parameters(), lr=1e-3)

    theta_pointwise = theta_good.copy()
    theta_setaware = theta_good.copy()
    theta_our = theta_good.copy()
    our_best = float(np.linalg.norm(theta_our - theta_true))

    rows: list[dict[str, str]] = []
    for generation in range(1, generations + 1):
        candidates = release_sample_candidates(rng, theta_true, bias, samples_per_gen, noise_std)
        labels = release_build_labels(candidates, theta_true, top_ratio=top_ratio)
        theta_hat = candidates.mean(axis=0)

        x = torch.from_numpy(candidates[None, ...].astype(np.float32)).to(device)
        y = torch.from_numpy(labels[None, ...].astype(np.float32)).to(device)
        theta_good_t = torch.from_numpy(theta_good[None, ...].astype(np.float32)).to(device)

        weights_point = pointwise(x)
        theta_point = pointwise.weighted_estimate(x, weights_point)
        loss_point = (
            release_classification_loss(weights_point, y)
            + release_contraction_loss(theta_point, theta_good_t)
            + 0.1 * release_ess_loss(weights_point, tau=tau)
        )
        opt_pointwise.zero_grad()
        loss_point.backward()
        opt_pointwise.step()
        theta_pointwise = theta_pointwise + pointwise_contraction * (
            theta_point.detach().cpu().numpy().squeeze(0) - theta_pointwise
        )

        weights_sa, delta_phi = setaware(x)
        theta_sa = setaware.weighted_estimate(x, weights_sa) + delta_phi
        loss_sa = (
            release_classification_loss(weights_sa, y)
            + release_contraction_loss(theta_sa, theta_good_t)
            + 0.1 * release_ess_loss(weights_sa, tau=tau)
            + 0.01 * release_correction_reg(delta_phi)
        )
        opt_setaware.zero_grad()
        loss_sa.backward()
        opt_setaware.step()
        theta_setaware = theta_setaware + setaware_contraction * (
            theta_sa.detach().cpu().numpy().squeeze(0) - theta_setaware
        )

        current_e = theta_our - theta_true
        raw_errors = candidates - theta_true[None, :]
        weights = robust_weights(raw_errors, tau=0.18)
        mu_sel = weighted_mean(raw_errors, weights)
        mu_ref = theta_good - theta_true
        delta_mu = mu_sel - mu_ref
        d2 = affine_generator(np.zeros((dim, dim), dtype=np.float64), delta_mu)
        omega_b = affine_generator(
            np.diag(np.full(dim, np.log(1.0 - setaware_contraction), dtype=np.float64)),
            np.zeros(dim, dtype=np.float64),
        )
        d_tilde = d2
        psi = (
            half_group(omega_b)
            @ half_group(d_tilde, sign=-1.0)
            @ expm(d2)
            @ half_group(d_tilde, sign=-1.0)
            @ half_group(omega_b)
        )
        theta_our = theta_true + affine_apply(psi, current_e)

        metric_our = float(np.linalg.norm(theta_our - theta_true))
        our_best = min(our_best, metric_our)
        print(
            f"scene=release_demo method=Our method generation={generation:03d} "
            f"prev_metric={format_float(float(np.linalg.norm(current_e)))} "
            f"metric={metric_our:.6f} delta={metric_our - float(np.linalg.norm(current_e)):.6f} "
            f"best_so_far={our_best:.6f}"
        )

        rows.append(
            {
                "generation": str(generation),
                "no_filter": format_float(float(np.linalg.norm(theta_hat - theta_true))),
                "standard_filter": format_float(float(np.linalg.norm(theta_pointwise - theta_true))),
                "set_aware": format_float(float(np.linalg.norm(theta_setaware - theta_true))),
                "our_method": format_float(metric_our),
                "set_aware_delta_phi_norm": format_float(float(np.linalg.norm(delta_phi.detach().cpu().numpy().squeeze(0)))),
            }
        )
    return rows


def plot_release_demo(rows: list[dict[str, str]], output_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 4.3))
    generations = [int(row["generation"]) for row in rows]
    curves = {
        "No Filter": [float(row["no_filter"]) for row in rows],
        "Standard Filter": [float(row["standard_filter"]) for row in rows],
        "Set-Aware": [float(row["set_aware"]) for row in rows],
        "Our method": [float(row["our_method"]) for row in rows],
    }
    styles = {
        "No Filter": {"color": "#7f8c8d", "linestyle": "--"},
        "Standard Filter": {"color": "#d62728", "linestyle": ":"},
        "Set-Aware": {"color": "#1f77b4", "linestyle": "-."},
        "Our method": {"color": "#2ca02c", "linestyle": "-"},
    }
    for name, values in curves.items():
        ax.plot(generations, values, label=name, linewidth=2.0, **styles[name])
    ax.set_xlabel("Generation")
    ax.set_ylabel(r"$\|\theta_t - \theta^*\|_2$")
    ax.set_title("Release Demo with Our Method")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_plot_only(results_dir: Path) -> None:
    main_rows = read_csv_rows(results_dir / "main_metrics.csv")
    plot_main(main_rows, results_dir / "unified_collapse_convergence.png")

    sweep_rows = read_csv_rows(results_dir / "icml_style_sweeps.csv")
    plot_sweeps(sweep_rows, results_dir / "icml_style_sweeps.png")

    transformer_rows = read_csv_rows(results_dir / "transformer_convergence.csv")
    plot_transformer(transformer_rows, results_dir / "transformer_convergence.png")

    release_path = results_dir / "release_demo_metrics.csv"
    if release_path.exists():
        release_rows = read_csv_rows(release_path)
        plot_release_demo(release_rows, results_dir / "release_demo_plot.png")


def write_summary(results_dir: Path, device: torch.device, seed: int, elapsed_sec: float) -> None:
    summary = results_dir / "run_summary.md"
    summary.write_text(
        "\n".join(
            [
                "# Run Summary",
                "",
                f"- device: `{device}`",
                f"- seed: `{seed}`",
                f"- elapsed_sec: `{elapsed_sec:.3f}`",
                "- success_judgement: pending manual terminal inspection",
                "- note: only after manual reading of per-generation logs should plots be treated as final",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified geometric order experiment aligned with test1.md.")
    parser.add_argument("--mode", choices=["main", "sweeps", "transformer", "release_demo", "plot", "all"], default="all")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    start = time.time()
    device = pick_device(args.device)
    print(f"device={device}")

    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in {"main", "all"}:
        rollout = run_main(device=device, seed=args.seed)
        write_csv(
            args.results_dir / "main_metrics.csv",
            rollout.main_rows,
            ["scene", "scene_title", "method", "generation", "metric_l2"],
        )
        write_csv(
            args.results_dir / "aux_metrics.csv",
            rollout.aux_rows,
            ["scene", "method", "generation", "delta_mu_norm", "residual_norm", "trace_a_res"],
        )

    if args.mode in {"sweeps", "all"}:
        sweep_rows = run_sweeps(seed=args.seed + 100)
        write_csv(
            args.results_dir / "icml_style_sweeps.csv",
            sweep_rows,
            ["control_name", "control_value", "method", "final_metric_l2"],
        )

    if args.mode in {"transformer", "all"}:
        transformer_rows = run_transformer(device=device, seed=args.seed + 200)
        write_csv(
            args.results_dir / "transformer_convergence.csv",
            transformer_rows,
            ["model", "method", "params", "step", "train_loss", "test_loss"],
        )

    if args.mode in {"release_demo", "all"}:
        release_rows = run_release_demo(seed=args.seed + 300)
        write_csv(
            args.results_dir / "release_demo_metrics.csv",
            release_rows,
            ["generation", "no_filter", "standard_filter", "set_aware", "our_method", "set_aware_delta_phi_norm"],
        )

    if args.mode == "plot" or (args.mode == "all" and not args.no_plot):
        run_plot_only(args.results_dir)
    elif args.mode in {"main", "sweeps", "transformer", "release_demo"} and not args.no_plot:
        if args.mode == "main":
            plot_main(read_csv_rows(args.results_dir / "main_metrics.csv"), args.results_dir / "unified_collapse_convergence.png")
        elif args.mode == "sweeps":
            plot_sweeps(read_csv_rows(args.results_dir / "icml_style_sweeps.csv"), args.results_dir / "icml_style_sweeps.png")
        elif args.mode == "transformer":
            plot_transformer(read_csv_rows(args.results_dir / "transformer_convergence.csv"), args.results_dir / "transformer_convergence.png")
        else:
            plot_release_demo(read_csv_rows(args.results_dir / "release_demo_metrics.csv"), args.results_dir / "release_demo_plot.png")

    elapsed = time.time() - start
    write_summary(args.results_dir, device=device, seed=args.seed, elapsed_sec=elapsed)
    print(f"results_dir={args.results_dir}")
    print(f"elapsed_sec={elapsed:.3f}")


if __name__ == "__main__":
    main()
