from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import random
import sys
from typing import Dict, List

import mauve
import torch
from mauve import utils as mauve_utils
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_HF_HOME = pathlib.Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_HF_HOME / "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_HF_HOME / "datasets"))


def _hf_hub_cache() -> str | None:
    cache = os.environ.get("HF_HUB_CACHE")
    if cache:
        return cache
    return str(DEFAULT_HF_HOME / "hub")

from exp11_gpt2_model.data import load_wikitext_subset  # noqa: E402


def read_lines(path: pathlib.Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def read_arrow_lines(path: pathlib.Path, column: str) -> List[str]:
    from datasets import Dataset

    ds = Dataset.from_file(str(path))
    if column not in ds.column_names:
        raise ValueError(f"Column '{column}' not found in {path} (columns: {ds.column_names})")
    return [text.strip() for text in ds[column] if text and text.strip()]


def sample_texts(texts: List[str], sample_size: int, rng: random.Random) -> List[str]:
    if len(texts) <= sample_size:
        return texts
    indices = rng.sample(range(len(texts)), sample_size)
    return [texts[i] for i in indices]


def tokenize_texts(tokenizer, texts: List[str], max_text_length: int) -> List[torch.LongTensor]:
    return [
        tokenizer.encode(t, return_tensors="pt", truncation=True, max_length=max_text_length)
        for t in texts
    ]


def featurize_texts(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int,
    max_text_length: int,
    name: str,
) -> torch.Tensor:
    tokenized = tokenize_texts(tokenizer, texts, max_text_length=max_text_length)
    return mauve_utils.featurize_tokens_from_model(
        model=model,
        tokenized_texts=tokenized,
        batch_size=batch_size,
        name=name,
        verbose=False,
    )


def compute_mauve_score(
    p_features,
    q_features,
    kmeans_num_redo: int,
    kmeans_max_iter: int,
    num_buckets,
) -> Dict[str, float]:
    out = mauve.compute_mauve(
        p_features=p_features,
        q_features=q_features,
        kmeans_num_redo=kmeans_num_redo,
        kmeans_max_iter=kmeans_max_iter,
        num_buckets=num_buckets,
        verbose=False,
    )
    mauve_std = getattr(out, "mauve_std", None)
    mauve_star = getattr(out, "mauve_star", None)
    divergence_curve = getattr(out, "divergence_curve", None)
    divergence = float(divergence_curve.mean()) if divergence_curve is not None else None
    p_count = int(p_features.shape[0]) if hasattr(p_features, "shape") else len(p_features)
    q_count = int(q_features.shape[0]) if hasattr(q_features, "shape") else len(q_features)
    return {
        "mauve": float(out.mauve),
        "mauve_std": float(mauve_std) if mauve_std is not None else None,
        "mauve_star": float(mauve_star) if mauve_star is not None else None,
        "divergence": divergence,
        "p_samples": p_count,
        "q_samples": q_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MAUVE for exp11 GPT-2 generations.")
    parser.add_argument(
        "--generations-dir",
        type=pathlib.Path,
        default=pathlib.Path("Total_results/Tables/exp11_gpt2_model/generations"),
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("exp11_gpt2_model/MAUVE"),
    )
    parser.add_argument("--seeds", type=str, default="1088,2195,4960")
    parser.add_argument("--methods", type=str, default="no_filter,pointwise,set_aware")
    parser.add_argument("--min-gen", type=int, default=0)
    parser.add_argument("--max-gen", type=int, default=4)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--model-name", type=str, default="gpt2-large")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--kmeans-num-redo", type=int, default=1)
    parser.add_argument("--kmeans-max-iter", type=int, default=100)
    parser.add_argument("--num-buckets", type=str, default="auto")
    parser.add_argument(
        "--ref-size",
        type=int,
        default=1000,
        help="Number of reference (validation) texts to sample for MAUVE.",
    )
    parser.add_argument(
        "--reference-file",
        type=pathlib.Path,
        default=None,
        help="Optional plaintext reference file (one sample per line). Overrides dataset loading.",
    )
    parser.add_argument(
        "--reference-arrow",
        type=pathlib.Path,
        default=None,
        help="Optional Arrow file to load reference texts from (e.g., wikitext-validation.arrow).",
    )
    parser.add_argument(
        "--reference-column",
        type=str,
        default="text",
        help="Column name to read from the reference Arrow file.",
    )
    parser.add_argument(
        "--reference-split",
        type=str,
        default="validation",
        help="Dataset split for reference sampling when no file is provided.",
    )
    parser.add_argument(
        "--ref-seed-offset",
        type=int,
        default=1,
        help="Reference seed = seed + offset (matches exp11 validation sampling).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.reference_file and args.reference_arrow:
        raise ValueError("Use only one of --reference-file or --reference-arrow.")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    device_id = args.device_id if args.device_id >= 0 else None

    cache_dir = _hf_hub_cache()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=True,
        cache_dir=cache_dir,
    )
    model = AutoModel.from_pretrained(
        args.model_name,
        local_files_only=True,
        cache_dir=cache_dir,
    )
    if device_id is not None and torch.cuda.is_available() and device_id >= 0:
        model = model.to(f"cuda:{device_id}")
    model = model.eval()

    csv_path = args.output_dir / "mauve_g0_g4.csv"
    json_path = args.output_dir / "mauve_g0_g4.json"

    results: List[Dict[str, float | int | str]] = []

    for seed in seeds:
        ref_seed = seed + args.ref_seed_offset
        if args.reference_file:
            if not args.reference_file.exists():
                raise FileNotFoundError(f"Reference file not found: {args.reference_file}")
            ref_texts = read_lines(args.reference_file)
        elif args.reference_arrow:
            if not args.reference_arrow.exists():
                raise FileNotFoundError(f"Reference Arrow file not found: {args.reference_arrow}")
            ref_texts = read_arrow_lines(args.reference_arrow, args.reference_column)
        else:
            ref_texts = load_wikitext_subset(args.reference_split, n_samples=args.ref_size, seed=ref_seed)
        if not ref_texts:
            raise ValueError("Reference texts are empty. Check reference inputs.")
        ref_rng = random.Random(ref_seed)
        ref_texts = sample_texts(ref_texts, args.ref_size, ref_rng)
        p_features = featurize_texts(
            model=model,
            tokenizer=tokenizer,
            texts=ref_texts,
            batch_size=args.batch_size,
            max_text_length=args.max_text_length,
            name=f"p/seed{seed}",
        ).cpu().numpy()

        for gen in range(args.min_gen, args.max_gen + 1):
            for method in methods:
                gen_file = args.generations_dir / str(seed) / f"g{gen}_{method}_candidates.txt"
                if not gen_file.exists():
                    raise FileNotFoundError(f"Missing generation file: {gen_file}")
                q_texts = read_lines(gen_file)
                q_rng = random.Random(seed * 10_000 + gen)
                q_texts = sample_texts(q_texts, args.sample_size, q_rng)
                q_features = featurize_texts(
                    model=model,
                    tokenizer=tokenizer,
                    texts=q_texts,
                    batch_size=args.batch_size,
                    max_text_length=args.max_text_length,
                    name=f"q/seed{seed}/g{gen}/{method}",
                ).cpu().numpy()

                metrics = compute_mauve_score(
                    p_features=p_features,
                    q_features=q_features,
                    kmeans_num_redo=args.kmeans_num_redo,
                    kmeans_max_iter=args.kmeans_max_iter,
                    num_buckets=args.num_buckets,
                )
                row: Dict[str, float | int | str] = {
                    "seed": seed,
                    "method": method,
                    "generation": gen,
                    "mauve": metrics["mauve"],
                    "mauve_std": metrics["mauve_std"],
                    "mauve_star": metrics["mauve_star"],
                    "divergence": metrics["divergence"],
                    "p_samples": metrics["p_samples"],
                    "q_samples": metrics["q_samples"],
                    "ref_seed": ref_seed,
                    "max_text_length": args.max_text_length,
                    "model_name": args.model_name,
                    "batch_size": args.batch_size,
                    "kmeans_num_redo": args.kmeans_num_redo,
                    "kmeans_max_iter": args.kmeans_max_iter,
                    "num_buckets": args.num_buckets,
                }
                results.append(row)
                print(
                    f"[seed {seed}][G{gen}][{method}] "
                    f"MAUVE={metrics['mauve']:.4f} (p={metrics['p_samples']}, q={metrics['q_samples']})"
                )

    with csv_path.open("w", newline="") as f_csv:
        fieldnames = [
            "seed",
            "method",
            "generation",
            "mauve",
            "mauve_std",
            "mauve_star",
            "divergence",
            "p_samples",
            "q_samples",
            "ref_seed",
            "max_text_length",
            "model_name",
            "batch_size",
            "kmeans_num_redo",
            "kmeans_max_iter",
            "num_buckets",
        ]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    args_dict = {}
    for key, value in vars(args).items():
        if isinstance(value, pathlib.Path):
            args_dict[key] = str(value)
        else:
            args_dict[key] = value

    with json_path.open("w") as f_json:
        json.dump({"args": args_dict, "results": results}, f_json, indent=2)

    print(f"Saved MAUVE CSV to {csv_path}")
    print(f"Saved MAUVE JSON to {json_path}")


if __name__ == "__main__":
    main()
