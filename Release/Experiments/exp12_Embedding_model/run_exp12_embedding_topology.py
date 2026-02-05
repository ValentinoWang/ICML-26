#!/usr/bin/env python3
"""
Exp12: Embedding Topology (manifold volume) analysis for recursive GPT-2 generations.

Outputs:
  - Vendi score by generation (line chart + CSV)
  - t-SNE map for Generation 4 (scatter + CSV)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import os

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:
    raise SystemExit(f"sentence-transformers is required: {exc}")


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f]
    return [line for line in lines if line]


def sample_lines(lines: List[str], n: int, seed: int) -> List[str]:
    if len(lines) < n:
        return lines
    rng = random.Random(seed)
    return rng.sample(lines, n)


def vendi_score(embeddings: np.ndarray) -> float:
    # L2 normalize for cosine kernel
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    x = embeddings / norms
    k = x @ x.T
    k = (k + k.T) * 0.5
    eigvals = np.linalg.eigvalsh(k)
    eigvals = np.clip(eigvals, 0.0, None)
    total = eigvals.sum()
    if total <= 0:
        return 0.0
    p = eigvals / total
    # Avoid log(0)
    p = np.clip(p, 1e-12, None)
    return float(np.exp(-np.sum(p * np.log(p))))


def load_generation_samples(
    data_root: Path,
    seed: int,
    method: str,
    generation: int,
    split: str,
    n: int,
    sample_seed: int,
) -> List[str]:
    name = f"g{generation}_{method}_{split}.txt"
    path = data_root / str(seed) / name
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    lines = read_lines(path)
    return sample_lines(lines, n, sample_seed)


def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_vendi(
    rows: List[Dict[str, float]],
    out_path: Path,
    colors: Dict[str, str],
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for method in sorted(set(r["method"] for r in rows)):
        if method == "reference":
            continue
        subset = [r for r in rows if r["method"] == method]
        subset.sort(key=lambda r: r["generation"])
        gens = [r["generation"] for r in subset]
        vals = [r["vendi"] for r in subset]
        ax.plot(
            gens,
            vals,
            marker="o",
            linewidth=2.4,
            markersize=5,
            color=colors.get(method, "#333333"),
            label=method.replace("_", "-"),
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Vendi Score")
    ax.set_xticks(sorted(set(r["generation"] for r in rows)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_tsne(
    coords: Dict[str, np.ndarray],
    out_path: Path,
    colors: Dict[str, str],
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KernelDensity

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    all_xy = np.vstack([xy for xy in coords.values()])
    x_min, x_max = float(all_xy[:, 0].min()), float(all_xy[:, 0].max())
    y_min, y_max = float(all_xy[:, 1].min()), float(all_xy[:, 1].max())
    x_pad = max((x_max - x_min) * 0.05, 1e-3)
    y_pad = max((y_max - y_min) * 0.05, 1e-3)
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    grid_size = 200
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    grid = np.vstack([xx.ravel(), yy.ravel()]).T

    def density_map(xy: np.ndarray) -> np.ndarray:
        std = np.std(xy, axis=0)
        mean_std = float(np.mean(std)) if np.all(np.isfinite(std)) else 1.0
        bandwidth = max(0.1, 0.3 * mean_std)
        kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
        kde.fit(xy)
        log_density = kde.score_samples(grid)
        return np.exp(log_density).reshape(xx.shape)

    legend_handles = []
    legend_labels = []
    for label, xy in coords.items():
        density = density_map(xy)
        vals = density[density > 0]
        if vals.size == 0:
            continue
        if label == "reference":
            level = float(np.quantile(vals, 0.4))
            cs = ax.contour(
                xx,
                yy,
                density,
                levels=[level],
                colors="black",
                linewidths=1.0,
                linestyles="--",
            )
            handle = plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.0)
        else:
            levels = np.quantile(vals, [0.6, 0.75, 0.9])
            levels = np.unique(levels)
            if levels.size < 3:
                levels = np.linspace(vals.min(), vals.max(), 4)[1:]
            color = colors.get(label, "#333333")
            cs = ax.contour(
                xx,
                yy,
                density,
                levels=levels,
                colors=color,
                linewidths=1.0,
                linestyles="-",
            )
            handle = plt.Line2D([0], [0], color=color, linestyle="-", linewidth=1.0)
        legend_handles.append(handle)
        legend_labels.append(label.replace("_", "-"))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.legend(legend_handles, legend_labels, frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    step2_root = Path(__file__).resolve().parents[2]
    default_data_root = (
        step2_root
        / "Experiments"
        / "Total_results"
        / "Tables"
        / "exp11_gpt2_model"
        / "generations"
    )
    default_out_root = step2_root / "Experiments" / "Total_results"

    parser = argparse.ArgumentParser(description="Exp12 Embedding Topology analysis.")
    parser.add_argument("--model", default="Alibaba-NLP/gte-large-en-v1.5")
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--out-root", type=Path, default=default_out_root)
    parser.add_argument("--methods", nargs="+", default=["no_filter", "pointwise", "dispersion", "set_aware"])
    parser.add_argument("--reference", default="no_filter")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1088, 2195, 4960])
    parser.add_argument("--gens", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--split", default="train", choices=["train", "candidates"])
    parser.add_argument("--tsne-seed", type=int, default=1088)
    parser.add_argument("--tsne-ref-gen", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tsne-perplexity", type=int, default=30)
    parser.add_argument("--tsne-iter", type=int, default=1000)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args()

    colors = {
        "pointwise": "#C44E52",
        "set_aware": "#4C72B0",
        "no_filter": "#E69F00",
        "dispersion": "#55A868",
        "reference": "#E69F00",
        args.reference: "#E69F00",
    }

    # Ensure model weights are cached in the system cache.
    cache_dir = Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_dir / "sentence_transformers"))

    out_tables = args.out_root / "Tables" / "exp12_Embedding_model"
    out_figs = args.out_root / "Figures" / "exp12_Embedding_model"
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    model = SentenceTransformer(args.model, trust_remote_code=args.trust_remote_code)

    # Vendi score across generations (average across seeds).
    vendi_rows: List[Dict[str, float]] = []
    for method in args.methods:
        for gen in args.gens:
            scores = []
            for seed in args.seeds:
                samples = load_generation_samples(
                    args.data_root,
                    seed=seed,
                    method=method,
                    generation=gen,
                    split=args.split,
                    n=args.n,
                    sample_seed=seed + gen,
                )
                embeddings = encode_texts(model, samples, args.batch_size)
                scores.append(vendi_score(embeddings))
            vendi_rows.append(
                {
                    "method": method,
                    "generation": gen,
                    "vendi": float(np.mean(scores)),
                    "vendi_std": float(np.std(scores)),
                }
            )

    vendi_csv = out_tables / "vendi_scores.csv"
    with vendi_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "generation", "vendi", "vendi_std"])
        writer.writeheader()
        writer.writerows(vendi_rows)

    vendi_plot = out_figs / "vendi_scores.png"
    plot_vendi(vendi_rows, vendi_plot, colors)

    # t-SNE for generation 4: reference vs selected methods
    tsne_gen = max(args.gens)
    ref_samples = load_generation_samples(
        args.data_root,
        seed=args.tsne_seed,
        method=args.reference,
        generation=args.tsne_ref_gen,
        split=args.split,
        n=args.n,
        sample_seed=args.tsne_seed + args.tsne_ref_gen,
    )
    method_samples = {}
    for idx, method in enumerate(args.methods):
        method_samples[method] = load_generation_samples(
            args.data_root,
            seed=args.tsne_seed,
            method=method,
            generation=tsne_gen,
            split=args.split,
            n=args.n,
            sample_seed=args.tsne_seed + tsne_gen + 11 + idx * 7,
        )

    all_texts = list(ref_samples)
    for samples in method_samples.values():
        all_texts.extend(samples)
    embeddings = encode_texts(model, all_texts, args.batch_size)

    # PCA pre-reduction for t-SNE stability
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    pca = PCA(n_components=min(50, embeddings.shape[1]), random_state=42)
    emb_pca = pca.fit_transform(embeddings)
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=args.tsne_perplexity,
        n_iter=args.tsne_iter,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(emb_pca)

    n_ref = len(ref_samples)
    coords_map = {"reference": coords[:n_ref]}
    offset = n_ref
    for method, samples in method_samples.items():
        count = len(samples)
        coords_map[method] = coords[offset : offset + count]
        offset += count

    coords_csv = out_tables / "tsne_g4_coords.csv"
    with coords_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "x", "y"])
        for label, xy in coords_map.items():
            for row in xy:
                writer.writerow([label, float(row[0]), float(row[1])])

    tsne_plot = out_figs / "tsne_g4.png"
    plot_tsne(coords_map, tsne_plot, colors)

    meta = {
        "model": args.model,
        "data_root": str(args.data_root),
        "split": args.split,
        "n": args.n,
        "seeds": args.seeds,
        "gens": args.gens,
        "tsne_seed": args.tsne_seed,
        "tsne_gen": tsne_gen,
        "tsne_ref_gen": args.tsne_ref_gen,
    }
    with (out_tables / "exp12_metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {vendi_csv} {vendi_plot} {coords_csv} {tsne_plot}")


if __name__ == "__main__":
    main()
