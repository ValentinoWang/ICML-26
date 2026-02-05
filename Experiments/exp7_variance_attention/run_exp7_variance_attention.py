import argparse
import json
import pathlib
import sys
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn, optim
from plot_figure11 import save_exp7_visuals

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BASE_TABLES_DIR = ROOT / "Total_results" / "Tables" / SCRIPT_DIR.name
BASE_FIGURES_DIR = ROOT / "Total_results" / "Figures" / SCRIPT_DIR.name
TABLES_DIR = BASE_TABLES_DIR / "results"
FIGURES_DIR = BASE_FIGURES_DIR / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from filter.losses import correction_reg  # noqa: E402
from filter.set_aware.model import SetAwareBiasRobustFilter  # noqa: E402


class CorrectionMLP(nn.Module):
    """Mean-pooling MLP that predicts a global correction (no set interaction)."""

    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)  # [B, D]
        h = torch.relu(self.fc1(pooled))
        h = self.dropout(h)
        h = torch.relu(self.fc2(h))
        h = self.dropout(h)
        out = self.fc3(h)
        return out  # [B, D]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        h = torch.relu(self.fc1(pooled))
        h = self.dropout(h)
        h = torch.relu(self.fc2(h))
        h = self.dropout(h)
        return h  # [B, hidden]


class PointwiseMLP(nn.Module):
    """Point-wise MLP applied to each element, then mean-pooled for global correction."""

    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        h = torch.relu(self.fc1(x))
        h = self.dropout(h)
        h = torch.relu(self.fc2(h))
        h = self.dropout(h)
        out = self.fc3(h)  # [B, N, D]
        return out.mean(dim=1)  # [B, D]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        h = self.dropout(h)
        h = torch.relu(self.fc2(h))
        h = self.dropout(h)
        return h.mean(dim=1)  # [B, hidden]


class BatchStatsMLP(nn.Module):
    """Point-wise MLP with batch mean/variance concatenated, then mean-pooled."""

    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim * 3, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        mean = mean.expand_as(x)
        var = var.expand_as(x)
        return torch.cat([x, mean, var], dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._augment(x)
        h = torch.relu(self.fc1(feats))
        h = self.dropout(h)
        h = torch.relu(self.fc2(h))
        h = self.dropout(h)
        out = self.fc3(h)
        return out.mean(dim=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._augment(x)
        h = torch.relu(self.fc1(feats))
        h = self.dropout(h)
        h = torch.relu(self.fc2(h))
        h = self.dropout(h)
        return h.mean(dim=1)


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def unit_vector(dim: int) -> np.ndarray:
    v = np.ones(dim)
    return v / (np.linalg.norm(v) + 1e-8)


def sample_variance_batch(
    rng: np.random.Generator,
    batch_size: int,
    n_candidates: int,
    dim: int,
    var_min: float,
    var_max: float,
    bias_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate candidate sets with controlled variance. Mean is fixed at theta_true=1,
    variance level sampled in [var_min, var_max]. Ground-truth correction norm is proportional to variance.
    Returns:
        candidates: [B, N, D]
        delta_true: [B, D]
        var_levels: [B]
    """
    var_levels = rng.uniform(low=var_min, high=var_max, size=batch_size)
    noise = rng.normal(size=(batch_size, n_candidates, dim))
    theta_true = np.ones(dim)
    candidates = theta_true + (var_levels[:, None, None] * noise).astype(np.float32)
    direction = unit_vector(dim)
    delta_true = (bias_scale * var_levels[:, None] * direction).astype(np.float32)
    return candidates, delta_true, var_levels.astype(np.float32)


def sample_pair_batch(
    rng: np.random.Generator,
    batch_size: int,
    n_candidates: int,
    dim: int,
    gap_min: float,
    gap_max: float,
    bias_scale: float,
    omega: float,
    base_std: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate candidate sets where bias follows a sinusoidal function of pair gap:
        ||delta|| = bias_scale * (sin(omega * gap) + 1).
    Two special points are placed symmetrically at +-gap/2 along a fixed direction; others are small noise around origin.
    Returns:
        candidates: [B, N, D]
        delta_true: [B, D]
        gaps: [B]
    """
    gaps = rng.uniform(low=gap_min, high=gap_max, size=batch_size)
    direction = unit_vector(dim)
    candidates = []
    delta_true = []
    theta_true = np.zeros(dim, dtype=np.float32)
    for g in gaps:
        # Two special points
        p1 = theta_true + (g / 2.0) * direction
        p2 = theta_true - (g / 2.0) * direction
        # Remaining points with small isotropic noise
        rest = base_std * rng.normal(size=(n_candidates - 2, dim))
        c = np.concatenate([rest, p1[None, :], p2[None, :]], axis=0)
        rng.shuffle(c, axis=0)
        candidates.append(c.astype(np.float32))
        delta_true.append((bias_scale * (np.sin(omega * g) + 1.0)) * direction)
    return np.stack(candidates, axis=0), np.stack(delta_true, axis=0), gaps.astype(np.float32)


def train_models(
    args: argparse.Namespace, device: torch.device
) -> Tuple[CorrectionMLP, PointwiseMLP, BatchStatsMLP, SetAwareBiasRobustFilter]:
    rng = set_seed(args.seed)
    mlp = CorrectionMLP(dim=args.dim, hidden=args.hidden, dropout=args.dropout).to(device)
    pw_mlp = PointwiseMLP(dim=args.dim, hidden=args.hidden, dropout=args.dropout).to(device)
    bs_mlp = BatchStatsMLP(dim=args.dim, hidden=args.hidden, dropout=args.dropout).to(device)
    sa = SetAwareBiasRobustFilter(dim=args.dim, hidden=args.hidden, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout).to(device)
    opt_mlp = optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_pw = optim.Adam(pw_mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_bs = optim.Adam(bs_mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_sa = optim.Adam(sa.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for step in range(args.train_steps):
        if args.mode == "variance":
            cands, delta_true_np, _ = sample_variance_batch(
                rng,
                batch_size=args.batch_size,
                n_candidates=args.candidates_per_set,
                dim=args.dim,
                var_min=args.var_min,
                var_max=args.var_max,
                bias_scale=args.bias_scale,
            )
        else:
            cands, delta_true_np, _ = sample_pair_batch(
                rng,
                batch_size=args.batch_size,
                n_candidates=args.candidates_per_set,
                dim=args.dim,
                gap_min=args.gap_min,
                gap_max=args.gap_max,
                bias_scale=args.bias_scale,
                omega=args.omega,
                base_std=args.pair_base_std,
            )
        x = torch.from_numpy(cands).float().to(device)
        delta_true = torch.from_numpy(delta_true_np).float().to(device)

        # MLP + correction
        pred_mlp = mlp(x)
        loss_mlp = ((pred_mlp - delta_true) ** 2).mean() + args.lambda_reg * correction_reg(pred_mlp.unsqueeze(1))
        opt_mlp.zero_grad()
        loss_mlp.backward()
        opt_mlp.step()

        # Point-wise MLP + correction
        pred_pw = pw_mlp(x)
        loss_pw = ((pred_pw - delta_true) ** 2).mean() + args.lambda_reg * correction_reg(pred_pw.unsqueeze(1))
        opt_pw.zero_grad()
        loss_pw.backward()
        opt_pw.step()

        # Point-wise + batch stats MLP + correction
        pred_bs = bs_mlp(x)
        loss_bs = ((pred_bs - delta_true) ** 2).mean() + args.lambda_reg * correction_reg(pred_bs.unsqueeze(1))
        opt_bs.zero_grad()
        loss_bs.backward()
        opt_bs.step()

        # Set-aware correction-only path
        pred_sa = sa.forward_correction_only(x)
        loss_sa = ((pred_sa - delta_true) ** 2).mean() + args.lambda_reg * correction_reg(pred_sa.unsqueeze(1))
        opt_sa.zero_grad()
        loss_sa.backward()
        opt_sa.step()

        if (step + 1) % 200 == 0:
            print(f"Train step {step+1}/{args.train_steps} done.")

    return mlp, pw_mlp, bs_mlp, sa


@torch.no_grad()
def eval_response(
    mlp: CorrectionMLP,
    pw_mlp: PointwiseMLP,
    bs_mlp: BatchStatsMLP,
    sa: SetAwareBiasRobustFilter,
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> Dict[str, Dict[str, np.ndarray]]:
    if args.mode == "variance":
        grid = np.linspace(args.var_min, args.var_max, args.eval_points, dtype=np.float32)
    else:
        grid = np.linspace(args.gap_min, args.gap_max, args.eval_points, dtype=np.float32)
    records = {"metric": [], "gt": [], "mlp": [], "pw": [], "bs": [], "sa": []}
    direction = unit_vector(args.dim)
    for v in grid:
        preds_mlp = []
        preds_pw = []
        preds_bs = []
        preds_sa = []
        for _ in range(args.eval_reps):
            if args.mode == "variance":
                noise = rng.normal(size=(1, args.candidates_per_set, args.dim))
                candidates = np.ones((1, args.candidates_per_set, args.dim)) + v * noise
                delta_true = args.bias_scale * v * direction
            else:
                # pair mode: gap = v
                theta_true = np.zeros(args.dim)
                p1 = theta_true + (v / 2.0) * direction
                p2 = theta_true - (v / 2.0) * direction
                rest = args.pair_base_std * rng.normal(size=(args.candidates_per_set - 2, args.dim))
                cands = np.concatenate([rest, p1[None, :], p2[None, :]], axis=0)
                rng.shuffle(cands, axis=0)
                candidates = cands[None, ...]
                delta_true = args.bias_scale * (np.sin(args.omega * v) + 1.0) * direction
            x = torch.from_numpy(candidates.astype(np.float32)).float().to(device)
            pm = mlp(x).cpu().numpy().squeeze(0)
            ppw = pw_mlp(x).cpu().numpy().squeeze(0)
            pbs = bs_mlp(x).cpu().numpy().squeeze(0)
            ps = sa.forward_correction_only(x).cpu().numpy().squeeze(0)
            preds_mlp.append(np.linalg.norm(pm))
            preds_pw.append(np.linalg.norm(ppw))
            preds_bs.append(np.linalg.norm(pbs))
            preds_sa.append(np.linalg.norm(ps))
        records["metric"].append(v)
        records["gt"].append(np.linalg.norm(delta_true))
        records["mlp"].append(np.mean(preds_mlp))
        records["pw"].append(np.mean(preds_pw))
        records["bs"].append(np.mean(preds_bs))
        records["sa"].append(np.mean(preds_sa))
    for k in records:
        records[k] = np.array(records[k])
    return records


@torch.no_grad()
def extract_attention(sa: SetAwareBiasRobustFilter, x: torch.Tensor) -> torch.Tensor:
    """
    Returns averaged attention map over heads from the last Transformer layer. Shape [N, N].
    """
    h = sa.encoder(x)
    # Pass through all but last layer
    for layer in sa.transformer.layers[:-1]:
        h = layer(h)
    last = sa.transformer.layers[-1]
    attn_out, attn_weights = last.self_attn(h, h, h, need_weights=True, average_attn_weights=False)
    # Continue with layer norm to follow data path
    h = h + last.dropout1(attn_out)
    h = last.norm1(h)
    attn_weights = attn_weights.detach()  # [B, num_heads, N, N]
    w = attn_weights.mean(dim=1).squeeze(0)
    return w.cpu()


@torch.no_grad()
def collect_latents(
    mlp: CorrectionMLP,
    pw_mlp: PointwiseMLP,
    bs_mlp: BatchStatsMLP,
    sa: SetAwareBiasRobustFilter,
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
    n_batches: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect hidden features (h_global) and true variances for t-SNE visualization.
    Returns:
        variances: [n]
        feats_mlp: [n, hidden]
        feats_pw: [n, hidden]
        feats_sa: [n, hidden_sa]
    """
    variances = []
    feats_mlp = []
    feats_pw = []
    feats_sa = []
    for _ in range(n_batches):
        if args.mode == "variance":
            cands, _, var_levels = sample_variance_batch(
                rng,
                batch_size=1,
                n_candidates=args.candidates_per_set,
                dim=args.dim,
                var_min=args.var_min,
                var_max=args.var_max,
                bias_scale=args.bias_scale,
            )
        else:
            cands, _, var_levels = sample_pair_batch(
                rng,
                batch_size=1,
                n_candidates=args.candidates_per_set,
                dim=args.dim,
                gap_min=args.gap_min,
                gap_max=args.gap_max,
                bias_scale=args.bias_scale,
                omega=args.omega,
                base_std=args.pair_base_std,
            )
        x = torch.from_numpy(cands).float().to(device)
        variances.append(var_levels.item())
        feats_mlp.append(mlp.forward_features(x).cpu().numpy().squeeze(0))
        feats_pw.append(pw_mlp.forward_features(x).cpu().numpy().squeeze(0))
        # Set-aware h_global after transformer mean-pooling
        h0 = sa.encoder(x)
        h = sa.transformer(h0)
        feats_sa.append(h.mean(dim=1).cpu().numpy().squeeze(0))
    return np.array(variances), np.array(feats_mlp), np.array(feats_pw), np.array(feats_sa)


def main():
    parser = argparse.ArgumentParser(description="Exp7: variance response & attention comparison (MLP vs set-aware).")
    parser.add_argument("--mode", choices=["variance", "pair"], default="pair", help="Variance-driven (old) or pair-gap-driven bias.")
    parser.add_argument("--dim", type=int, default=50)
    parser.add_argument("--candidates-per-set", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--var-min", type=float, default=0.05)
    parser.add_argument("--var-max", type=float, default=0.8)
    parser.add_argument("--gap-min", type=float, default=0.05, help="Min pair gap for pair-mode bias.")
    parser.add_argument("--gap-max", type=float, default=0.8, help="Max pair gap for pair-mode bias.")
    parser.add_argument("--pair-base-std", type=float, default=0.01, help="Noise std for background points in pair mode.")
    parser.add_argument("--bias-scale", type=float, default=10.0, help="Scale for ripple bias (sin(omega*gap)+1).")
    parser.add_argument("--omega", type=float, default=15.0, help="Frequency for ripple bias in pair mode.")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-reg", type=float, default=0.001)
    parser.add_argument("--eval-points", type=int, default=30)
    parser.add_argument("--eval-reps", type=int, default=4)
    parser.add_argument("--attn-low", type=float, default=0.1)
    parser.add_argument("--attn-high", type=float, default=0.7)
    parser.add_argument("--tsne-batches", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out-dir", type=pathlib.Path, default=TABLES_DIR, help="Directory to store CSV/NPZ outputs.")
    parser.add_argument("--fig-dir", type=pathlib.Path, default=FIGURES_DIR, help="Directory to store figures.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    mlp, pw_mlp, bs_mlp, sa = train_models(args, device=device)

    rng_eval = set_seed(args.seed + 123)
    records = eval_response(mlp, pw_mlp, bs_mlp, sa, args, device=device, rng=rng_eval)

    # Attention maps
    with torch.no_grad():
        noise_low = torch.randn(1, args.candidates_per_set, args.dim) * args.attn_low + 1.0
        noise_high = torch.randn(1, args.candidates_per_set, args.dim) * args.attn_high + 1.0
        x_low = noise_low.to(device)
        x_high = noise_high.to(device)
        attn_low = extract_attention(sa, x_low)
        attn_high = extract_attention(sa, x_high)

    out_dir = args.out_dir
    fig_dir = args.fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # t-SNE latent visualization inputs
    rng_tsne = set_seed(args.seed + 999)
    variances, feats_mlp, feats_pw, feats_sa = collect_latents(
        mlp,
        pw_mlp,
        bs_mlp,
        sa,
        args,
        device=device,
        rng=rng_tsne,
        n_batches=args.tsne_batches,
    )
    save_exp7_visuals(
        records=records,
        attn_low=attn_low,
        attn_high=attn_high,
        variances=variances,
        feats_mlp=feats_mlp,
        feats_pw=feats_pw,
        feats_sa=feats_sa,
        mode=args.mode,
        seed=args.seed,
        table_dir=out_dir,
        figure_dir=fig_dir,
    )

    runtime = {
        "device": str(device),
        "train_steps": args.train_steps,
        "seed": args.seed,
        "var_range": [args.var_min, args.var_max],
    }
    with (out_dir / "runtime_exp7.json").open("w") as f:
        json.dump(runtime, f, indent=2)

    print(f"Saved Exp7 response & attention visuals to {out_dir} (tables) and {fig_dir} (figures)")


if __name__ == "__main__":
    main()
