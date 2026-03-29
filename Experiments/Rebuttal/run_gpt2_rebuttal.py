import argparse
import copy
import csv
import json
import pathlib
import sys
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel, logging as hf_logging

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENTS_DIR.parent
for path in (REPO_ROOT, EXPERIMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from exp11_gpt2_model import run_exp11_gpt2_model as gpt2_base  # noqa: E402
from exp11_gpt2_model.data import _hf_hub_cache, _manual_model_dir, load_wikitext_subset, prepare_tokenizer_model  # noqa: E402
from exp11_gpt2_model.filter_module import apply_ppl_leash, select_training_indices  # noqa: E402
from exp11_gpt2_model.generation import embed_texts, generate_texts  # noqa: E402
from exp11_gpt2_model.text_quality import compute_text_quality  # noqa: E402
from exp11_gpt2_model.training import compute_perplexities, eval_validation_ppl, fine_tune  # noqa: E402
from Tools.deterministic import set_deterministic  # noqa: E402


BASE_RESULTS_DIR = SCRIPT_DIR / "results" / "gpt2"
METHOD_SEED_IDS: Dict[str, int] = {
    "no_filter": 0,
    "pointwise": 1,
    "set_aware": 2,
    "clean_mix": 3,
    "batch_mean": 4,
    "unsup_set_aware": 5,
    "clean_only": 6,
    "clean_ft": 7,
    "kl_reg": 8,
}
PHASE_SEED_IDS: Dict[str, int] = {"round": 0, "train": 1, "eval": 2}


def resolve_encoder_source() -> tuple[str, bool]:
    manual = _manual_model_dir("all-MiniLM-L6-v2")
    if manual is not None and (manual / "config.json").exists():
        if (manual / "model.safetensors").exists() and (manual / "model.safetensors").stat().st_size > 10_000_000:
            return str(manual), True
        if (manual / "pytorch_model.bin").exists() and (manual / "pytorch_model.bin").stat().st_size > 10_000_000:
            return str(manual), True
    return "sentence-transformers/all-MiniLM-L6-v2", False


def derive_phase_seed(base_seed: int, method: str, generation: int, phase: str) -> int:
    if method not in METHOD_SEED_IDS:
        raise ValueError(f"Unknown method={method!r}")
    if phase not in PHASE_SEED_IDS:
        raise ValueError(f"Unknown phase={phase!r}")
    return (int(base_seed) + METHOD_SEED_IDS[method] * 10_000_000 + int(generation) * 10_000 + PHASE_SEED_IDS[phase]) % (
        2**32
    )


def seed_results_complete(results_csv_path: pathlib.Path, methods: Sequence[str], generations: int) -> bool:
    if not results_csv_path.exists():
        return False
    required = {(m, g) for m in methods for g in range(int(generations))}
    seen: set[tuple[str, int]] = set()
    with results_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = str(row.get("method", "")).strip()
            if not method:
                continue
            try:
                gen = int(float(row.get("generation", -1)))
            except (TypeError, ValueError):
                continue
            seen.add((method, gen))
    return required.issubset(seen)


def checkpoint_path(results_path_seed: pathlib.Path, method: str, generation: int) -> pathlib.Path:
    return results_path_seed.parent / f"{method}_gen{generation}.pt"


def generation_artifacts_complete(generations_dir: pathlib.Path, method: str, generation: int) -> bool:
    cand_path = generations_dir / f"g{generation}_{method}_candidates.txt"
    train_path = generations_dir / f"g{generation}_{method}_train.txt"
    return cand_path.exists() and train_path.exists()


def _parse_history_value(text: Any) -> Any:
    if isinstance(text, (int, float)):
        return text
    if text in ("", None):
        return text
    raw = str(text).strip()
    if not raw:
        return ""
    try:
        value = float(raw)
    except ValueError:
        return raw
    if value.is_integer():
        return int(value)
    return value


def load_partial_history(results_path_seed: pathlib.Path) -> Dict[str, List[Dict[str, Any]]]:
    json_path = results_path_seed
    csv_path = results_path_seed.with_suffix(".csv")
    history: Dict[str, List[Dict[str, Any]]] = {}

    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f_json:
            payload = json.load(f_json)
        raw_history = payload.get("history", {}) if isinstance(payload, dict) else {}
        if isinstance(raw_history, dict):
            for method, records in raw_history.items():
                clean_records: List[Dict[str, Any]] = []
                for rec in records or []:
                    if not isinstance(rec, dict):
                        continue
                    row = dict(rec)
                    if "generation" in row:
                        row["generation"] = int(row["generation"])
                    clean_records.append(row)
                clean_records.sort(key=lambda rec: int(rec.get("generation", -1)))
                history[str(method)] = clean_records
        return history

    if not csv_path.exists():
        return history

    with csv_path.open("r", encoding="utf-8", newline="") as f_csv:
        reader = csv.DictReader(f_csv)
        for row in reader:
            method = str(row.get("method", "")).strip()
            if not method:
                continue
            record: Dict[str, Any] = {}
            for key, value in row.items():
                if key == "method":
                    continue
                parsed = _parse_history_value(value)
                if parsed == "":
                    continue
                record[key] = parsed
            if "generation" not in record:
                continue
            record["generation"] = int(record["generation"])
            history.setdefault(method, []).append(record)
    for records in history.values():
        records.sort(key=lambda rec: int(rec.get("generation", -1)))
    return history


def discover_resume_state(
    results_path_seed: pathlib.Path,
    generations_dir: pathlib.Path,
    methods: Sequence[str],
    generations: int,
    needs_generation_files: bool,
) -> tuple[Dict[str, List[Dict[str, Any]]], int]:
    history = load_partial_history(results_path_seed)
    if not history:
        return {str(method): [] for method in methods}, 0

    history_by_method_gen: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for method in methods:
        records = list(history.get(str(method), []))
        history_by_method_gen[str(method)] = {int(rec["generation"]): rec for rec in records if "generation" in rec}

    last_complete_generation = -1
    for generation in range(int(generations)):
        complete = True
        for method in methods:
            method = str(method)
            if generation not in history_by_method_gen.get(method, {}):
                complete = False
                break
            if not checkpoint_path(results_path_seed, method, generation).exists():
                complete = False
                break
            if needs_generation_files and not generation_artifacts_complete(generations_dir, method, generation):
                complete = False
                break
        if not complete:
            break
        last_complete_generation = generation

    next_generation = last_complete_generation + 1
    trimmed_history = {
        str(method): [
            dict(rec)
            for rec in history.get(str(method), [])
            if int(rec.get("generation", -1)) < next_generation
        ]
        for method in methods
    }
    return trimmed_history, next_generation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuttal GPT-2 baselines for clean-reference mixing and batch-mean aggregation.")
    parser.add_argument("--wikitext-train-size", type=int, default=50000)
    parser.add_argument("--wikitext-val-size", type=int, default=5000)
    parser.add_argument("--validation-source-size", type=int, default=8000)
    parser.add_argument("--validation-unit", type=str, default="lines", choices=["lines", "chunks"])
    parser.add_argument("--eval-split", type=str, default="validation", choices=["validation", "test"])
    parser.add_argument("--clean-support-split", type=str, default="validation", choices=["validation", "test"])
    parser.add_argument("--prompt-pool-size", type=int, default=5000)
    parser.add_argument("--candidate-pool", type=int, default=10000)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--clean-ref-size", type=int, default=500)
    parser.add_argument("--clean-val-size", type=int, default=None)
    parser.add_argument("--strict-clean-val-split", action="store_true")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--generation-batch", type=int, default=768)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--embed-batch-size", type=int, default=768)
    parser.add_argument("--embed-max-length", type=int, default=128)
    parser.add_argument("--epochs-per-gen", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--initial-epochs", type=int, default=1)
    parser.add_argument("--initial-max-steps", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--eval-sample-size", type=int, default=512)
    parser.add_argument("--val-eval-size", type=int, default=512)

    parser.add_argument("--filter-set-size", type=int, default=1024)
    parser.add_argument("--filter-steps", type=int, default=200)
    parser.add_argument("--filter-lr", type=float, default=1e-3)
    parser.add_argument("--filter-hidden", type=int, default=128)
    parser.add_argument("--filter-heads", type=int, default=4)
    parser.add_argument("--filter-layers", type=int, default=2)
    parser.add_argument("--filter-dropout", type=float, default=0.1)
    parser.add_argument("--filter-knn", type=int, default=32)
    parser.add_argument("--filter-temperature", type=float, default=1.0)
    parser.add_argument("--filter-ess-tau", type=float, default=256.0)
    parser.add_argument("--filter-ess-weight", type=float, default=0.01)
    parser.add_argument("--delta-phi-scale", type=float, default=1.0)
    parser.add_argument("--clean-val-geom-scale", type=float, default=0.0)

    parser.add_argument("--ppl-leash-tau", type=float, default=0.7)
    parser.add_argument("--ppl-leash-mode", type=str, default="upper", choices=["upper", "abs", "lower"])
    parser.add_argument("--ppl-leash-strength", type=float, default=1.0)
    parser.add_argument("--ppl-leash-ref-mode", type=str, default="sliding", choices=["sliding", "fixed"])
    parser.add_argument("--rep-filter-n", type=int, default=4)
    parser.add_argument("--rep-filter-threshold", type=float, default=0.6)
    parser.add_argument("--ppl-safety-min-weight", type=float, default=0.5)

    parser.add_argument(
        "--methods",
        type=str,
        default="set_aware,clean_mix,batch_mean",
        help="Comma-separated methods: no_filter, pointwise, set_aware, unsup_set_aware, clean_mix, clean_only, clean_ft, batch_mean, kl_reg.",
    )
    parser.add_argument(
        "--clean-mix-generated-source",
        type=str,
        default="first_k",
        choices=["first_k", "no_filter", "pointwise"],
    )
    parser.add_argument("--batch-mean-scale", type=float, default=1.0)
    parser.add_argument("--kl-coef", type=float, default=0.1)

    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1088)
    parser.add_argument("--seeds", type=str, default="1088,2195,4960")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=None,
        help="Optional per-process CUDA memory fraction in (0,1], e.g. 0.8 for 80%% of one GPU.",
    )
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--save-generations", action="store_true")
    parser.add_argument(
        "--smoke-mode",
        action="store_true",
        help="Override expensive defaults with a tiny 1-generation configuration for pipeline verification.",
    )
    parser.add_argument("--generations-dir", type=pathlib.Path, default=BASE_RESULTS_DIR / "generations")
    parser.add_argument("--results-path", type=pathlib.Path, default=BASE_RESULTS_DIR / "metrics_diversity_ppl.json")
    return parser.parse_args()


def resolve_clean_val_size(args: argparse.Namespace) -> int:
    if args.clean_val_size is None:
        return int(args.clean_ref_size)
    return max(0, int(args.clean_val_size))


def apply_smoke_mode(args: argparse.Namespace) -> None:
    if not bool(getattr(args, "smoke_mode", False)):
        return
    overrides = {
        "wikitext_train_size": 256,
        "wikitext_val_size": 32,
        "validation_source_size": 128,
        "prompt_pool_size": 8,
        "candidate_pool": 8,
        "train_samples": 4,
        "generations": 1,
        "max_new_tokens": 8,
        "generation_batch": 2,
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "embed_batch_size": 2,
        "embed_max_length": 64,
        "epochs_per_gen": 1,
        "initial_epochs": 0,
        "initial_max_steps": 0,
        "eval_sample_size": 8,
        "val_eval_size": 4,
        "filter_set_size": 8,
        "filter_steps": 2,
        "filter_hidden": 16,
        "filter_heads": 1,
        "filter_layers": 1,
        "filter_knn": 4,
        "block_size": 64,
    }
    for key, value in overrides.items():
        setattr(args, key, value)
    args.clean_ref_size = min(int(args.clean_ref_size), 4)
    if args.clean_val_size is not None:
        args.clean_val_size = min(int(args.clean_val_size), 4)


def chunk_texts_nonoverlap(tokenizer, texts: Sequence[str], block_size: int) -> List[str]:
    joined = "\n\n".join(str(t).strip() for t in texts if str(t).strip())
    if not joined:
        return []
    token_ids = tokenizer.encode(joined, add_special_tokens=False)
    if not token_ids:
        return []
    chunks: List[str] = []
    for start in range(0, len(token_ids), int(block_size)):
        block = token_ids[start : start + int(block_size)]
        if not block:
            continue
        text = tokenizer.decode(block, clean_up_tokenization_spaces=True).strip()
        if text:
            chunks.append(text)
    return chunks


def load_text_units(
    split: str,
    tokenizer,
    n_samples: int,
    source_size: int,
    seed: int,
    unit: str,
    block_size: int,
) -> List[str]:
    raw = load_wikitext_subset(split, n_samples=max(int(n_samples), int(source_size)), seed=seed)
    if unit == "chunks":
        return chunk_texts_nonoverlap(tokenizer, raw, block_size=block_size)
    return list(raw)


def load_clean_and_eval_texts(tokenizer, args: argparse.Namespace, seed: int) -> tuple[List[str], List[str], List[str]]:
    clean_val_size = resolve_clean_val_size(args)
    clean_pool_size = max(clean_val_size, int(args.clean_ref_size))
    eval_seed = seed + 1
    clean_seed = seed + 1
    if args.strict_clean_val_split and args.clean_support_split == args.eval_split and clean_pool_size > 0:
        shared_needed = clean_pool_size + int(args.wikitext_val_size)
        shared_pool = load_text_units(
            split=args.eval_split,
            tokenizer=tokenizer,
            n_samples=shared_needed,
            source_size=max(int(args.validation_source_size), shared_needed),
            seed=eval_seed,
            unit=args.validation_unit,
            block_size=args.block_size,
        )
        if len(shared_pool) < shared_needed:
            raise ValueError(
                f"Need at least {shared_needed} {args.eval_split} units for a strict split, but only found {len(shared_pool)}. "
                "Increase --validation-source-size or reduce --wikitext-val-size/--clean-val-size."
            )
        clean_pool = list(shared_pool[:clean_pool_size])
        eval_texts = list(shared_pool[clean_pool_size : clean_pool_size + int(args.wikitext_val_size)])
    else:
        eval_texts = load_text_units(
            split=args.eval_split,
            tokenizer=tokenizer,
            n_samples=int(args.wikitext_val_size),
            source_size=int(args.wikitext_val_size),
            seed=eval_seed,
            unit=args.validation_unit,
            block_size=args.block_size,
        )
        if clean_pool_size > 0:
            clean_pool = load_text_units(
                split=args.clean_support_split,
                tokenizer=tokenizer,
                n_samples=clean_pool_size,
                source_size=max(int(args.validation_source_size), clean_pool_size),
                seed=clean_seed,
                unit=args.validation_unit,
                block_size=args.block_size,
            )
            if len(clean_pool) < clean_pool_size:
                raise ValueError(
                    f"Need at least {clean_pool_size} {args.clean_support_split} units for clean support, but only found {len(clean_pool)}."
                )
        else:
            clean_pool = []
    clean_refs = list(clean_pool[: int(args.clean_ref_size)])
    clean_val_texts = list(clean_pool[:clean_val_size])
    return clean_refs, clean_val_texts, eval_texts


def make_filter_args(args: argparse.Namespace, ppl_ref: float | None) -> Dict[str, Any]:
    return {
        "set_size": args.filter_set_size,
        "steps": args.filter_steps,
        "lr": args.filter_lr,
        "hidden": args.filter_hidden,
        "heads": args.filter_heads,
        "layers": args.filter_layers,
        "dropout": args.filter_dropout,
        "knn": args.filter_knn,
        "temperature": args.filter_temperature,
        "ess_tau": args.filter_ess_tau,
        "ess_weight": args.filter_ess_weight,
        "delta_phi_scale": args.delta_phi_scale,
        "clean_val_geom_scale": args.clean_val_geom_scale,
        "ppl_ref": ppl_ref,
        "ppl_leash_tau": args.ppl_leash_tau,
        "ppl_leash_mode": args.ppl_leash_mode,
        "ppl_leash_strength": args.ppl_leash_strength,
        "ppl_safety_min_weight": args.ppl_safety_min_weight,
        "rep_filter_n": args.rep_filter_n,
        "rep_filter_threshold": args.rep_filter_threshold,
    }


def select_batch_mean_indices(
    train_size: int,
    ppl_scores: Sequence[float],
    embeddings: torch.Tensor,
    filter_args: Dict[str, Any],
    device: torch.device,
    scale: float,
) -> tuple[List[int], Dict[str, float]]:
    emb = embeddings.detach().cpu().numpy().astype(np.float32)
    ppl = np.asarray(ppl_scores, dtype=np.float64)
    mean_vec = emb.mean(axis=0, keepdims=True)
    mean_vec = mean_vec / (np.linalg.norm(mean_vec, axis=1, keepdims=True) + 1e-8)
    emb_unit = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    align = (emb_unit * mean_vec).sum(axis=1)
    pointwise = -np.log(np.maximum(ppl, 1e-8))
    raw = pointwise - float(scale) * align.astype(np.float64)
    centered = raw - float(raw.mean())
    scaled = centered / (float(raw.std()) + 1e-8)
    weights = torch.sigmoid(torch.tensor(scaled, dtype=torch.float32, device=device))
    weights = apply_ppl_leash(
        weights,
        ppl_scores=list(ppl_scores),
        ppl_ref=filter_args.get("ppl_ref"),
        tau=float(filter_args.get("ppl_leash_tau", 1.0)),
        mode=str(filter_args.get("ppl_leash_mode", "upper")),
        strength=float(filter_args.get("ppl_leash_strength", 0.0)),
    )
    chosen = torch.topk(weights, k=min(int(train_size), weights.numel())).indices.tolist()
    extra = {
        "batch_mean_avg_align": float(np.mean(align[chosen])) if chosen else 0.0,
        "batch_mean_avg_weight": float(weights[chosen].mean().item()) if chosen else 0.0,
    }
    return chosen, extra


def round_payload(
    model: GPT2LMHeadModel,
    tokenizer,
    encoder,
    encoder_tokenizer,
    prompt_pool: Sequence[str],
    device: torch.device,
    args: argparse.Namespace,
    method: str,
    ppl_ref: float | None,
    clean_refs: Sequence[str],
    clean_val_embeddings: torch.Tensor | None,
    clean_val_size: int,
    progress: tqdm | None = None,
) -> Dict[str, Any]:
    selection_method = "no_filter" if method == "kl_reg" else method
    if method in {"clean_only", "clean_ft"}:
        if not clean_refs:
            raise ValueError(f"method={method!r} requires non-empty clean references; increase --clean-ref-size.")
        train_texts = list(clean_refs)
        train_quality = compute_text_quality(train_texts)
        return {
            "train_texts": train_texts,
            "all_candidates": [],
            "train_quality": {
                "train_unique_line_ratio": train_quality.unique_line_ratio,
                "train_rep4_intra": train_quality.rep4_intra,
                "train_gzip_ratio": train_quality.gzip_ratio,
                "train_avg_words": train_quality.avg_words,
                "clean_val_size": float(clean_val_size),
                "train_clean_refs": float(len(clean_refs)),
            },
        }

    candidates = generate_texts(
        model=model,
        tokenizer=tokenizer,
        prompt_pool=prompt_pool,
        n_samples=args.candidate_pool,
        batch_size=args.generation_batch,
        max_new_tokens=args.max_new_tokens,
        device=device,
        temperature=args.temperature,
        top_p=args.top_p,
        progress=progress,
    )
    ppl_scores = compute_perplexities(
        model=model,
        tokenizer=tokenizer,
        texts=candidates,
        device=device,
        batch_size=args.eval_batch_size,
    )
    embeddings = embed_texts(
        encoder=encoder,
        encoder_tokenizer=encoder_tokenizer,
        texts=candidates,
        device=device,
        batch_size=args.embed_batch_size,
        max_length=args.embed_max_length,
    )
    filter_args = make_filter_args(args=args, ppl_ref=ppl_ref)

    extra: Dict[str, float] = {}
    if selection_method == "clean_mix":
        if args.clean_mix_generated_source == "first_k":
            idx = list(range(min(int(args.train_samples), len(candidates))))
        else:
            idx = select_training_indices(
                method=args.clean_mix_generated_source,
                train_size=args.train_samples,
                ppl_scores=ppl_scores,
                embeddings=embeddings,
                texts=candidates,
                filter_args=filter_args,
                device=device,
            )
        train_texts = [candidates[i] for i in idx] + list(clean_refs)
        extra["train_clean_refs"] = float(len(clean_refs))
    elif selection_method == "batch_mean":
        idx, extra = select_batch_mean_indices(
            train_size=args.train_samples,
            ppl_scores=ppl_scores,
            embeddings=embeddings,
            filter_args=filter_args,
            device=device,
            scale=args.batch_mean_scale,
        )
        train_texts = [candidates[i] for i in idx]
    else:
        idx = select_training_indices(
            method=selection_method,
            train_size=args.train_samples,
            ppl_scores=ppl_scores,
            embeddings=embeddings,
            texts=candidates,
            filter_args=filter_args,
            device=device,
            clean_embeddings=clean_val_embeddings,
        )
        train_texts = [candidates[i] for i in idx]

    train_quality = compute_text_quality(train_texts)
    payload = {
        "train_texts": train_texts,
        "all_candidates": candidates,
        "train_quality": {
            "train_unique_line_ratio": train_quality.unique_line_ratio,
            "train_rep4_intra": train_quality.rep4_intra,
            "train_gzip_ratio": train_quality.gzip_ratio,
            "train_avg_words": train_quality.avg_words,
            "clean_val_size": float(clean_val_size),
            **extra,
        },
    }
    return payload


def main() -> None:
    args = parse_args()
    apply_smoke_mode(args)
    hf_logging.set_verbosity_error()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and args.gpu_memory_fraction is not None:
        frac = float(args.gpu_memory_fraction)
        if not (0.0 < frac <= 1.0):
            raise ValueError("--gpu-memory-fraction must be in (0, 1].")
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(frac, device=device_index)
    seeds_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else [args.seed]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = {
        "no_filter",
        "pointwise",
        "set_aware",
        "unsup_set_aware",
        "clean_mix",
        "clean_only",
        "clean_ft",
        "batch_mean",
        "kl_reg",
    }
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"Unknown methods in --methods: {unknown}. Allowed: {sorted(allowed)}")
    if args.smoke_mode:
        print(
            "Smoke mode: "
            f"generations={args.generations}, candidate_pool={args.candidate_pool}, train_samples={args.train_samples}, "
            f"prompt_pool={args.prompt_pool_size}, eval_sample_size={args.eval_sample_size}"
        )

    seed_pbar = tqdm(seeds_list, desc="Seeds", position=0, leave=True)
    for seed in seed_pbar:
        seed_pbar.set_postfix_str(f"seed={seed}")
        results_path_seed = args.results_path.resolve().parent / f"{seed}" / args.results_path.name
        seed_generations_dir = args.generations_dir / str(seed)
        needs_generation_files = bool(args.save_generations)
        has_generation_files = True
        if needs_generation_files:
            for method in methods:
                for gen in range(int(args.generations)):
                    cand_path = seed_generations_dir / f"g{gen}_{method}_candidates.txt"
                    if not cand_path.exists():
                        has_generation_files = False
                        break
                if not has_generation_files:
                    break
        if seed_results_complete(results_path_seed.with_suffix(".csv"), methods=methods, generations=args.generations) and (
            not needs_generation_files or has_generation_files
        ):
            print(f"Skip seed={seed}: found complete results at {results_path_seed.with_suffix('.csv')}")
            continue
        history, start_generation = discover_resume_state(
            results_path_seed=results_path_seed,
            generations_dir=seed_generations_dir,
            methods=methods,
            generations=int(args.generations),
            needs_generation_files=needs_generation_files,
        )
        if start_generation > 0:
            print(
                f"Resume seed={seed} from generation {start_generation} "
                f"using {results_path_seed.with_suffix('.csv')} and per-generation checkpoints."
            )
        set_deterministic(seed)
        tokenizer, base_model = prepare_tokenizer_model()
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        base_model.to(device)

        cache_dir = _hf_hub_cache()
        encoder_source, encoder_local_only = resolve_encoder_source()
        encoder_tokenizer = AutoTokenizer.from_pretrained(
            encoder_source,
            local_files_only=encoder_local_only,
            cache_dir=cache_dir,
        )
        encoder_model = AutoModel.from_pretrained(
            encoder_source,
            local_files_only=encoder_local_only,
            cache_dir=cache_dir,
        ).to(device)

        train_texts = load_wikitext_subset("train", n_samples=args.wikitext_train_size, seed=seed)
        clean_refs, clean_val_texts, eval_val_texts = load_clean_and_eval_texts(tokenizer, args=args, seed=seed)
        prompt_pool = train_texts[: args.prompt_pool_size] if len(train_texts) >= args.prompt_pool_size else train_texts
        clean_val_embeddings: torch.Tensor | None = None
        if clean_val_texts and float(args.clean_val_geom_scale) != 0.0:
            encoder_model = encoder_model.to(device)
            clean_val_embeddings = embed_texts(
                encoder=encoder_model,
                encoder_tokenizer=encoder_tokenizer,
                texts=clean_val_texts,
                device=device,
                batch_size=args.embed_batch_size,
                max_length=args.embed_max_length,
            )

        if args.initial_epochs > 0:
            base_model = fine_tune(
                model=base_model,
                tokenizer=tokenizer,
                texts=train_texts[: args.prompt_pool_size],
                device=device,
                epochs=args.initial_epochs,
                batch_size=args.train_batch_size,
                lr=args.lr,
                warmup_steps=args.warmup_steps,
                max_length=args.block_size,
                max_steps=args.initial_max_steps,
            )
        base_model = base_model.to("cpu")
        torch.cuda.empty_cache()

        ref_model = copy.deepcopy(base_model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad_(False)

        models: Dict[str, GPT2LMHeadModel] = {m: copy.deepcopy(base_model) for m in methods}
        history = {m: list(history.get(m, [])) for m in methods}
        if start_generation > 0:
            for method in methods:
                method_ckpt = checkpoint_path(results_path_seed, method, start_generation - 1)
                state = torch.load(method_ckpt, map_location="cpu")
                models[method].load_state_dict(state["model_state"])
        prev_val_ppl: Dict[str, float | None] = {m: None for m in methods}
        if args.ppl_leash_strength != 0.0:
            if start_generation > 0 and args.ppl_leash_ref_mode == "sliding":
                for method in methods:
                    if history[method]:
                        prev_val_ppl[method] = float(history[method][-1]["val_ppl"])
            else:
                for method in methods:
                    model = models[method].to(device)
                    prev_val_ppl[method] = eval_validation_ppl(
                        model=model,
                        tokenizer=tokenizer,
                        val_texts=eval_val_texts,
                        device=device,
                        batch_size=args.eval_batch_size,
                        max_eval=args.val_eval_size,
                    )
                    models[method] = model.to("cpu")
                    torch.cuda.empty_cache()

        generations_dir = args.generations_dir / str(seed)
        if args.save_generations:
            generations_dir.mkdir(parents=True, exist_ok=True)

        gen_pbar = tqdm(total=int(args.generations), initial=int(start_generation), desc=f"Seed {seed} | Generations", position=1, leave=True)
        for gen in range(int(start_generation), int(args.generations)):
            for method in methods:
                set_deterministic(derive_phase_seed(seed, method, gen, "round"))
                model = models[method].to(device)
                phase_gen = tqdm(
                    total=args.candidate_pool,
                    desc=f"G{gen} {method} | generate",
                    position=2,
                    leave=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
                )
                round_data = round_payload(
                    model=model,
                    tokenizer=tokenizer,
                    encoder=encoder_model,
                    encoder_tokenizer=encoder_tokenizer,
                    prompt_pool=prompt_pool,
                    device=device,
                    args=args,
                    method=method,
                    ppl_ref=prev_val_ppl.get(method),
                    clean_refs=clean_refs,
                    clean_val_embeddings=clean_val_embeddings,
                    clean_val_size=len(clean_val_texts),
                    progress=phase_gen,
                )
                phase_gen.close()

                if args.save_generations:
                    cand_path = generations_dir / f"g{gen}_{method}_candidates.txt"
                    with cand_path.open("w", encoding="utf-8") as f_cand:
                        f_cand.write("\n".join(round_data["all_candidates"]))
                    sel_path = generations_dir / f"g{gen}_{method}_train.txt"
                    with sel_path.open("w", encoding="utf-8") as f_sel:
                        f_sel.write("\n".join(round_data["train_texts"]))

                set_deterministic(derive_phase_seed(seed, method, gen, "train"))
                ref_model_device = None
                kl_coef = 0.0
                if method == "kl_reg":
                    ref_model_device = ref_model.to(device)
                    ref_model_device.eval()
                    kl_coef = float(args.kl_coef)
                model = fine_tune(
                    model=model,
                    tokenizer=tokenizer,
                    texts=round_data["train_texts"],
                    device=device,
                    epochs=args.epochs_per_gen,
                    batch_size=args.train_batch_size,
                    lr=args.lr,
                    warmup_steps=args.warmup_steps,
                    max_length=args.block_size,
                    progress_desc=f"G{gen} {method} | finetune",
                    ref_model=ref_model_device,
                    kl_coef=kl_coef,
                )
                if ref_model_device is not None:
                    ref_model = ref_model_device.to("cpu")
                    torch.cuda.empty_cache()

                set_deterministic(derive_phase_seed(seed, method, gen, "eval"))
                phase_eval = tqdm(
                    total=args.eval_sample_size,
                    desc=f"G{gen} {method} | eval",
                    position=2,
                    leave=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
                )
                metrics = gpt2_base.compute_generation_metrics(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_pool=prompt_pool,
                    val_texts=eval_val_texts,
                    args=args,
                    device=device,
                    progress=phase_eval,
                )
                metrics.update(round_data["train_quality"])
                metrics["generation"] = gen
                history[method].append(metrics)
                if args.ppl_leash_ref_mode == "sliding":
                    prev_val_ppl[method] = float(metrics["val_ppl"])
                phase_eval.close()
                tqdm.write(
                    f"[G{gen}][{method}] PPL={metrics['val_ppl']:.2f} "
                    f"D2={metrics['distinct2']:.3f} D3={metrics['distinct3']:.3f} D4={metrics['distinct4']:.3f}"
                )
                if args.save_checkpoints:
                    ckpt_dir = results_path_seed.parent
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = checkpoint_path(results_path_seed, method, gen)
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "seed": int(seed),
                            "method": str(method),
                            "generation": int(gen),
                            "val_ppl": float(metrics["val_ppl"]),
                        },
                        ckpt_path,
                    )
                models[method] = model.to("cpu")
                torch.cuda.empty_cache()

            gpt2_base.save_results_partial(history, vars(args).copy(), results_path_seed)
            print(f"Saved results to {results_path_seed}")
            print(f"Saved flat metrics to {results_path_seed.with_suffix('.csv')}")
            gen_pbar.update(1)
        gen_pbar.close()
    seed_pbar.close()


if __name__ == "__main__":
    main()
