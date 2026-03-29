#!/usr/bin/env python3
"""
exp10: computational cost analysis.

Measure forward/backward time per batch (ms) and GPU memory for:
- ResNet-18 backbone only
- ResNet-18 + pointwise (MLP) filter
- ResNet-18 + set-aware (Transformer) filter
- GPT-2 LM backbone only
- GPT-2 + pointwise filter
- GPT-2 + set-aware filter

Timing uses torch.cuda.Event for precise GPU measurements (falls back to perf_counter on CPU).
Results are saved as a CSV table with relative overhead vs. the no-filter baseline for each backbone.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import models
from transformers import GPT2Config, GPT2LMHeadModel

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "Total_results" / "Tables" / SCRIPT_DIR.name / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from filter.set_aware.model import SetAwareBiasRobustFilter  # noqa: E402
from filter.standard.model import StandardFilter  # noqa: E402


@dataclass
class BenchmarkResult:
    backbone: str
    method: str
    component: str
    time_ms: float
    std_ms: float
    gpu_memory_mb: float
    relative_overhead: float


def weighted_estimate(tokens: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute weighted mean over the sequence dimension -> [B, D].
    """
    w = weights.unsqueeze(-1)
    num = (w * tokens).sum(dim=1)
    den = weights.sum(dim=1).clamp_min(eps).unsqueeze(-1)
    return num / den


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def benchmark_step(
    step_fn: Callable[[], torch.Tensor],
    optimizer: optim.Optimizer,
    device: torch.device,
    iterations: int,
    warmup: int,
) -> Tuple[float, float, float]:
    """
    Run warmup + timed iterations, returning (mean_ms, std_ms, gpu_memory_mb).
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    times: List[float] = []
    for i in range(warmup + iterations):
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
        else:
            t0 = time.perf_counter()

        loss = step_fn()
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            ender.record()
            torch.cuda.synchronize(device)
            if i >= warmup:
                times.append(starter.elapsed_time(ender))
        else:
            t1 = time.perf_counter()
            if i >= warmup:
                times.append((t1 - t0) * 1000.0)

    mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    return mean_ms, std_ms, mem_mb


def build_resnet_components(num_classes: int, filter_dim: int) -> tuple[nn.Module, nn.Module, nn.Module]:
    """
    Returns feature extractor (up to layer4), 1x1 projector, and a classifier head.
    """
    backbone = models.resnet18(weights=None)
    feature_extractor = nn.Sequential(*list(backbone.children())[:-2])  # spatial map before avgpool
    projector = nn.Conv2d(512, filter_dim, kernel_size=1)
    head = nn.Linear(filter_dim, num_classes)
    return feature_extractor, projector, head


def benchmark_resnet(args: argparse.Namespace, device: torch.device) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []

    # Shared synthetic batch
    images = torch.randn(args.resnet_batch_size, 3, args.resnet_image_size, args.resnet_image_size, device=device)
    labels = torch.randint(0, args.resnet_num_classes, (args.resnet_batch_size,), device=device)

    # Baseline: vanilla ResNet-18
    resnet_model = models.resnet18(weights=None, num_classes=args.resnet_num_classes).to(device)
    resnet_model.train()
    optimizer = optim.Adam(resnet_model.parameters(), lr=args.resnet_lr)

    def resnet_step() -> torch.Tensor:
        logits = resnet_model(images)
        return F.cross_entropy(logits, labels)

    mean_ms, std_ms, mem_mb = benchmark_step(resnet_step, optimizer, device, args.iterations, args.warmup)
    baseline_time = mean_ms
    results.append(
        BenchmarkResult(
            backbone="ResNet-18",
            method="No Filter",
            component="Backbone",
            time_ms=mean_ms,
            std_ms=std_ms,
            gpu_memory_mb=mem_mb,
            relative_overhead=0.0,
        )
    )

    # Pointwise (MLP) filter
    feat, proj, head = build_resnet_components(args.resnet_num_classes, args.resnet_filter_dim)
    mlp_filter = StandardFilter(dim=args.resnet_filter_dim, hidden=args.resnet_filter_hidden, dropout=args.filter_dropout)
    params = list(feat.parameters()) + list(proj.parameters()) + list(mlp_filter.parameters()) + list(head.parameters())
    mlp_filter = mlp_filter.to(device)
    feat = feat.to(device)
    proj = proj.to(device)
    head = head.to(device)
    optimizer = optim.Adam(params, lr=args.resnet_lr)

    def mlp_step() -> torch.Tensor:
        fmap = feat(images)  # [B, 512, H, W]
        tokens = proj(fmap).flatten(2).transpose(1, 2)  # [B, N, D]
        weights = mlp_filter(tokens)
        pooled = weighted_estimate(tokens, weights)
        logits = head(pooled)
        return F.cross_entropy(logits, labels)

    mean_ms, std_ms, mem_mb = benchmark_step(mlp_step, optimizer, device, args.iterations, args.warmup)
    results.append(
        BenchmarkResult(
            backbone="ResNet-18",
            method="Pointwise",
            component="MLP Filter",
            time_ms=mean_ms,
            std_ms=std_ms,
            gpu_memory_mb=mem_mb,
            relative_overhead=(mean_ms - baseline_time) / baseline_time * 100.0,
        )
    )

    # Set-aware Transformer filter
    feat_sa, proj_sa, head_sa = build_resnet_components(args.resnet_num_classes, args.resnet_filter_dim)
    sa_filter = SetAwareBiasRobustFilter(
        dim=args.resnet_filter_dim,
        hidden=args.resnet_filter_hidden,
        n_heads=args.resnet_filter_heads,
        n_layers=args.resnet_filter_layers,
        dropout=args.filter_dropout,
    )
    params = (
        list(feat_sa.parameters()) + list(proj_sa.parameters()) + list(sa_filter.parameters()) + list(head_sa.parameters())
    )
    sa_filter = sa_filter.to(device)
    feat_sa = feat_sa.to(device)
    proj_sa = proj_sa.to(device)
    head_sa = head_sa.to(device)
    optimizer = optim.Adam(params, lr=args.resnet_lr)

    def sa_step() -> torch.Tensor:
        fmap = feat_sa(images)
        tokens = proj_sa(fmap).flatten(2).transpose(1, 2)
        weights, delta = sa_filter(tokens)
        pooled = weighted_estimate(tokens, weights) + delta
        logits = head_sa(pooled)
        return F.cross_entropy(logits, labels)

    mean_ms, std_ms, mem_mb = benchmark_step(sa_step, optimizer, device, args.iterations, args.warmup)
    results.append(
        BenchmarkResult(
            backbone="ResNet-18",
            method="Set-Aware",
            component="Transformer Filter",
            time_ms=mean_ms,
            std_ms=std_ms,
            gpu_memory_mb=mem_mb,
            relative_overhead=(mean_ms - baseline_time) / baseline_time * 100.0,
        )
    )

    del resnet_model, feat, proj, head, mlp_filter, feat_sa, proj_sa, head_sa, sa_filter
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def build_gpt2_model(args: argparse.Namespace, device: torch.device) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=args.vocab_size,
        n_positions=args.gpt2_seq_len,
        n_ctx=args.gpt2_seq_len,
        n_embd=args.gpt2_hidden,
        n_layer=args.gpt2_layers,
        n_head=args.gpt2_heads,
    )
    model = GPT2LMHeadModel(config)
    return model.to(device)


def benchmark_gpt2(args: argparse.Namespace, device: torch.device) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    input_ids = torch.randint(
        0, args.vocab_size, (args.gpt2_batch_size, args.gpt2_seq_len), device=device, dtype=torch.long
    )
    attention_mask = torch.ones_like(input_ids)

    # Baseline GPT-2 LM step
    gpt2_base = build_gpt2_model(args, device)
    gpt2_base.train()
    optimizer = optim.AdamW(gpt2_base.parameters(), lr=args.gpt2_lr)

    def gpt2_step() -> torch.Tensor:
        outputs = gpt2_base(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        return outputs.loss

    mean_ms, std_ms, mem_mb = benchmark_step(gpt2_step, optimizer, device, args.iterations, args.warmup)
    baseline_time = mean_ms
    results.append(
        BenchmarkResult(
            backbone="GPT-2",
            method="No Filter",
            component="Backbone",
            time_ms=mean_ms,
            std_ms=std_ms,
            gpu_memory_mb=mem_mb,
            relative_overhead=0.0,
        )
    )

    # Pointwise filter on top of hidden states
    gpt2_mlp = build_gpt2_model(args, device)
    proj = nn.Linear(args.gpt2_hidden, args.gpt2_filter_dim).to(device)
    mlp_filter = StandardFilter(dim=args.gpt2_filter_dim, hidden=args.gpt2_filter_hidden, dropout=args.filter_dropout).to(
        device
    )
    head = nn.Linear(args.gpt2_filter_dim, args.vocab_size).to(device)
    params = list(gpt2_mlp.parameters()) + list(proj.parameters()) + list(mlp_filter.parameters()) + list(head.parameters())
    optimizer = optim.AdamW(params, lr=args.gpt2_lr)

    def gpt2_mlp_step() -> torch.Tensor:
        outputs = gpt2_mlp(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]  # [B, T, H]
        tokens = proj(hidden)  # [B, T, D]
        weights = mlp_filter(tokens)
        pooled = weighted_estimate(tokens, weights)
        logits = head(pooled)  # [B, V]
        aux_labels = input_ids[:, 0]  # lightweight auxiliary target
        aux_loss = F.cross_entropy(logits, aux_labels)
        return outputs.loss + args.filter_aux_weight * aux_loss

    mean_ms, std_ms, mem_mb = benchmark_step(gpt2_mlp_step, optimizer, device, args.iterations, args.warmup)
    results.append(
        BenchmarkResult(
            backbone="GPT-2",
            method="Pointwise",
            component="MLP Filter",
            time_ms=mean_ms,
            std_ms=std_ms,
            gpu_memory_mb=mem_mb,
            relative_overhead=(mean_ms - baseline_time) / baseline_time * 100.0,
        )
    )

    # Set-aware Transformer filter on hidden states
    gpt2_sa = build_gpt2_model(args, device)
    proj_sa = nn.Linear(args.gpt2_hidden, args.gpt2_filter_dim).to(device)
    sa_filter = SetAwareBiasRobustFilter(
        dim=args.gpt2_filter_dim,
        hidden=args.gpt2_filter_hidden,
        n_heads=args.gpt2_filter_heads,
        n_layers=args.gpt2_filter_layers,
        dropout=args.filter_dropout,
    ).to(device)
    head_sa = nn.Linear(args.gpt2_filter_dim, args.vocab_size).to(device)
    params = list(gpt2_sa.parameters()) + list(proj_sa.parameters()) + list(sa_filter.parameters()) + list(head_sa.parameters())
    optimizer = optim.AdamW(params, lr=args.gpt2_lr)

    def gpt2_sa_step() -> torch.Tensor:
        outputs = gpt2_sa(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        tokens = proj_sa(hidden)
        weights, delta = sa_filter(tokens)
        pooled = weighted_estimate(tokens, weights) + delta
        logits = head_sa(pooled)
        aux_labels = input_ids[:, 0]
        aux_loss = F.cross_entropy(logits, aux_labels)
        return outputs.loss + args.filter_aux_weight * aux_loss

    mean_ms, std_ms, mem_mb = benchmark_step(gpt2_sa_step, optimizer, device, args.iterations, args.warmup)
    results.append(
        BenchmarkResult(
            backbone="GPT-2",
            method="Set-Aware",
            component="Transformer Filter",
            time_ms=mean_ms,
            std_ms=std_ms,
            gpu_memory_mb=mem_mb,
            relative_overhead=(mean_ms - baseline_time) / baseline_time * 100.0,
        )
    )

    del gpt2_base, gpt2_mlp, gpt2_sa, proj, mlp_filter, head, proj_sa, sa_filter, head_sa
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def save_results(rows: List[BenchmarkResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["backbone", "method", "component", "time_ms", "std_ms", "gpu_memory_mb", "relative_overhead_pct"])
        for r in rows:
            writer.writerow(
                [
                    r.backbone,
                    r.method,
                    r.component,
                    f"{r.time_ms:.3f}",
                    f"{r.std_ms:.3f}",
                    f"{r.gpu_memory_mb:.1f}",
                    f"{r.relative_overhead:.2f}",
                ]
            )


def print_table(rows: List[BenchmarkResult]) -> None:
    by_backbone = {}
    for r in rows:
        by_backbone.setdefault(r.backbone, []).append(r)
    for backbone, items in by_backbone.items():
        print(f"\n{backbone} (per-iteration, forward+backward)")
        print(f"{'Method':<12} {'Component':<22} {'Time (ms)':>10} {'Mem (MB)':>10} {'Overhead':>10}")
        for r in items:
            overhead = "-" if r.relative_overhead == 0 else f"{r.relative_overhead:+.1f}%"
            print(
                f"{r.method:<12} {r.component:<22} {r.time_ms:>10.2f} {r.gpu_memory_mb:>10.1f} {overhead:>10}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="exp10: benchmark computational cost of filters vs. backbones.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--iterations", type=int, default=20, help="timed iterations (after warmup)")
    parser.add_argument("--warmup", type=int, default=5, help="warmup iterations before timing")
    parser.add_argument("--seed", type=int, default=0)

    # ResNet settings
    parser.add_argument("--resnet-batch-size", type=int, default=16)
    parser.add_argument("--resnet-image-size", type=int, default=224)
    parser.add_argument("--resnet-num-classes", type=int, default=1000)
    parser.add_argument("--resnet-lr", type=float, default=1e-3)
    parser.add_argument("--resnet-filter-dim", type=int, default=128)
    parser.add_argument("--resnet-filter-hidden", type=int, default=256)
    parser.add_argument("--resnet-filter-heads", type=int, default=4)
    parser.add_argument("--resnet-filter-layers", type=int, default=2)

    # GPT-2 settings (trimmed by default to keep runtime practical; bump layers/heads for full GPT-2)
    parser.add_argument("--gpt2-batch-size", type=int, default=4)
    parser.add_argument("--gpt2-seq-len", type=int, default=128)
    parser.add_argument("--gpt2-hidden", type=int, default=768)
    parser.add_argument("--gpt2-layers", type=int, default=8)
    parser.add_argument("--gpt2-heads", type=int, default=12)
    parser.add_argument("--gpt2-lr", type=float, default=5e-4)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--gpt2-filter-dim", type=int, default=256)
    parser.add_argument("--gpt2-filter-hidden", type=int, default=256)
    parser.add_argument("--gpt2-filter-heads", type=int, default=4)
    parser.add_argument("--gpt2-filter-layers", type=int, default=2)

    parser.add_argument("--filter-dropout", type=float, default=0.0)
    parser.add_argument("--filter-aux-weight", type=float, default=0.1, help="scale for auxiliary filter loss")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=TABLES_DIR / "exp10_time_cost.csv",
        help="CSV path for saving the timing table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    resnet_results = benchmark_resnet(args, device)
    gpt2_results = benchmark_gpt2(args, device)
    all_results = resnet_results + gpt2_results

    save_results(all_results, args.results_path)
    print_table(all_results)
    print(f"\nSaved table to {args.results_path}")


if __name__ == "__main__":
    main()
