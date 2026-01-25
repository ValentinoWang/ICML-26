import argparse
import json
import os
import pathlib
import sys
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BASE_TABLES_DIR = ROOT / "Total_results" / "Tables" / SCRIPT_DIR.name
DEFAULT_OUT_DIR = BASE_TABLES_DIR / "effective_zone"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp11_gpt2_model.generation import embed_texts  # noqa: E402


def load_lines(path: pathlib.Path) -> List[str]:
    with path.open() as f:
        return [line.strip() for line in f if line.strip()]


def maybe_sample(texts: Sequence[str], max_n: int, rng: np.random.Generator) -> List[str]:
    if max_n <= 0 or len(texts) <= max_n:
        return list(texts)
    idx = rng.choice(len(texts), size=max_n, replace=False)
    return [texts[i] for i in idx]


def load_encoder(cache_dir: str | None, local_only: bool) -> tuple[AutoTokenizer, AutoModel]:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    encoder_tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only, cache_dir=cache_dir)
    encoder_model = AutoModel.from_pretrained(model_name, local_files_only=local_only, cache_dir=cache_dir)
    return encoder_tokenizer, encoder_model


def compute_projection_ratios(embeddings: np.ndarray, bias: np.ndarray, dims: Sequence[int], seed: int) -> Dict[int, float]:
    max_dim = min(max(dims), embeddings.shape[1])
    pca = PCA(n_components=max_dim, random_state=seed)
    pca.fit(embeddings)
    comps = pca.components_
    ratios: Dict[int, float] = {}
    bias_norm = np.linalg.norm(bias) + 1e-12
    for d in dims:
        d_eff = min(d, comps.shape[0])
        comp = comps[:d_eff]
        proj = comp.T @ (comp @ bias)
        ratios[d] = float(np.linalg.norm(proj) / bias_norm)
    return ratios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Effective geometric zone scan on Exp11 generations.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1088, 2195, 4960])
    parser.add_argument("--generations", type=str, default="0,4")
    parser.add_argument("--method", type=str, default="set_aware")
    parser.add_argument("--max-candidates", type=int, default=2000)
    parser.add_argument("--max-selected", type=int, default=2000)
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--embed-max-length", type=int, default=128)
    parser.add_argument("--dims", type=str, default="8,16,32,64,128,256,384")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--local-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gens = [int(g.strip()) for g in args.generations.split(",") if g.strip()]
    dims = [int(d.strip()) for d in args.dims.split(",") if d.strip()]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_dir = os.environ.get("HF_HUB_CACHE")
    if args.local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        encoder_tokenizer, encoder_model = load_encoder(cache_dir, local_only=args.local_only)
    except Exception:
        if args.local_only:
            raise
        encoder_tokenizer, encoder_model = load_encoder(cache_dir, local_only=False)
    encoder_model = encoder_model.to(device)

    rows: List[Dict[str, float]] = []
    rng = np.random.default_rng(0)
    for seed in args.seeds:
        seed_dir = BASE_TABLES_DIR / "generations" / str(seed)
        for gen in gens:
            cand_path = seed_dir / f"g{gen}_{args.method}_candidates.txt"
            train_path = seed_dir / f"g{gen}_{args.method}_train.txt"
            if not cand_path.exists() or not train_path.exists():
                raise FileNotFoundError(f"Missing generation files for seed={seed}, gen={gen}: {cand_path}, {train_path}")
            candidates = load_lines(cand_path)
            selected = load_lines(train_path)
            candidates = maybe_sample(candidates, args.max_candidates, rng)
            selected = maybe_sample(selected, args.max_selected, rng)

            emb_cand = embed_texts(
                encoder=encoder_model,
                encoder_tokenizer=encoder_tokenizer,
                texts=candidates,
                device=device,
                batch_size=args.embed_batch_size,
                max_length=args.embed_max_length,
            ).cpu().numpy()
            emb_sel = embed_texts(
                encoder=encoder_model,
                encoder_tokenizer=encoder_tokenizer,
                texts=selected,
                device=device,
                batch_size=args.embed_batch_size,
                max_length=args.embed_max_length,
            ).cpu().numpy()

            bias = emb_sel.mean(axis=0) - emb_cand.mean(axis=0)
            ratios = compute_projection_ratios(emb_cand, bias, dims, seed=seed + gen)
            bias_norm = float(np.linalg.norm(bias))
            for d, ratio in ratios.items():
                rows.append(
                    {
                        "seed": float(seed),
                        "generation": float(gen),
                        "dim": float(d),
                        "proj_ratio": float(ratio),
                        "bias_norm": bias_norm,
                        "n_candidates": float(len(candidates)),
                        "n_selected": float(len(selected)),
                    }
                )

    # Summaries across seeds
    summary: Dict[str, Dict[str, float]] = {}
    for gen in gens:
        for d in dims:
            filtered = [r for r in rows if int(r["generation"]) == gen and int(r["dim"]) == d]
            if not filtered:
                continue
            vals = np.array([r["proj_ratio"] for r in filtered], dtype=np.float64)
            key = f"g{gen}_d{d}"
            summary[key] = {
                "proj_ratio_mean": float(vals.mean()),
                "proj_ratio_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            }

    clean_args = {}
    for key, value in vars(args).items():
        clean_args[key] = str(value) if isinstance(value, pathlib.Path) else value
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "effective_zone_set_aware.csv"
    json_path = out_dir / "effective_zone_set_aware.json"
    with csv_path.open("w") as f:
        keys = ["seed", "generation", "dim", "proj_ratio", "bias_norm", "n_candidates", "n_selected"]
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(f"{row[k]:.6f}" for k in keys) + "\n")
    payload = {"args": clean_args, "summary": summary, "rows": rows}
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved effective zone scan to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
