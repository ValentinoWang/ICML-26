import argparse
import csv
import json
import pathlib
from typing import Iterable, List

import mauve


def load_texts(path: pathlib.Path, max_samples: int | None = None) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    texts: List[str] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = obj.get("text", "")
                if text:
                    texts.append(text)
                if max_samples and len(texts) >= max_samples:
                    break
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    texts.append(line.strip())
                if max_samples and len(texts) >= max_samples:
                    break
    if not texts:
        raise ValueError(f"No texts found in {path}")
    return texts


def parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute MAUVE for Qwen2 recursive candidates vs validation.")
    parser.add_argument(
        "--run-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing candidates_g{gen}_{method}.jsonl files.",
    )
    parser.add_argument(
        "--val-jsonl",
        type=pathlib.Path,
        required=True,
        help="Validation JSONL with text field (used as q distribution).",
    )
    parser.add_argument("--gens", type=str, default="5,6")
    parser.add_argument("--methods", type=str, default="base,pointwise,dispersion,set_aware")
    parser.add_argument("--out-csv", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-samples", type=int, default=1000)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--num-buckets", type=int, default=25)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--max-text-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--kmeans-num-redo", type=int, default=5)
    parser.add_argument("--kmeans-max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=25)
    args = parser.parse_args()

    gens = [int(item) for item in parse_list(args.gens)]
    methods = parse_list(args.methods)

    val_texts = load_texts(args.val_jsonl, max_samples=args.val_samples)

    rows: List[dict[str, str | int | float]] = []
    for gen in gens:
        for method in methods:
            candidate_path = args.run_dir / f"candidates_g{gen}_{method}.jsonl"
            candidate_texts = load_texts(candidate_path, max_samples=args.candidate_samples)
            out = mauve.compute_mauve(
                p_text=candidate_texts,
                q_text=val_texts,
                num_buckets=args.num_buckets,
                device_id=args.device_id,
                max_text_length=args.max_text_length,
                batch_size=args.batch_size,
                kmeans_num_redo=args.kmeans_num_redo,
                kmeans_max_iter=args.kmeans_max_iter,
                seed=args.seed,
                verbose=False,
            )
            divergence_curve = getattr(out, "divergence_curve", None)
            divergence = float(divergence_curve.mean()) if divergence_curve is not None else None
            p_file = str(candidate_path).lstrip("/")
            row = {
                "generation": gen,
                "method": method,
                "mauve": float(out.mauve),
                "mauve_std": float(getattr(out, "mauve_std", 0.0) or 0.0),
                "mauve_star": float(getattr(out, "mauve_star", 0.0) or 0.0),
                "divergence": divergence,
                "p_samples": len(candidate_texts),
                "q_samples": len(val_texts),
                "num_buckets": args.num_buckets,
                "p_file": p_file,
            }
            rows.append(row)
            print(f"[G{gen}][{method}] MAUVE={row['mauve']:.4f}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "generation",
                "method",
                "mauve",
                "mauve_std",
                "mauve_star",
                "divergence",
                "p_samples",
                "q_samples",
                "num_buckets",
                "p_file",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
