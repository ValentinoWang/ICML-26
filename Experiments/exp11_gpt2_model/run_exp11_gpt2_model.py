import argparse
import copy
import csv
import json
import pathlib
import os
import sys
from typing import Dict, List, Sequence, Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import logging as hf_logging
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BASE_TABLES_DIR = ROOT / "Total_results" / "Tables" / SCRIPT_DIR.name
RESULTS_DIR = BASE_TABLES_DIR / "Results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp11_gpt2_model.data import load_wikitext_subset, prepare_tokenizer_model  # noqa: E402
from exp11_gpt2_model.filter_module import select_training_indices  # noqa: E402
from exp11_gpt2_model.generation import embed_texts, generate_texts  # noqa: E402
from exp11_gpt2_model.metrics import distinct_n  # noqa: E402
from exp11_gpt2_model.text_quality import compute_text_quality  # noqa: E402
from exp11_gpt2_model.training import compute_perplexities, eval_validation_ppl, fine_tune  # noqa: E402
from Tools.deterministic import set_deterministic  # noqa: E402


# Keep stable IDs for the original methods to preserve determinism across versions.
METHOD_SEED_IDS: Dict[str, int] = {
    "no_filter": 0,
    "pointwise": 1,
    "set_aware": 2,
    "ppl_leash": 3,
    "rep_filter": 4,
    "dispersion": 5,
    "ppl_safety": 6,
}
PHASE_SEED_IDS: Dict[str, int] = {"round": 0, "train": 1, "eval": 2}


def derive_phase_seed(base_seed: int, method: str, generation: int, phase: str) -> int:
    """
    Derive a deterministic per-(seed, method, generation, phase) seed so each method's
    randomness is independent of which other methods are run in the same process.
    """
    if method not in METHOD_SEED_IDS:
        raise ValueError(f"Unknown method={method!r} for seed derivation")
    if phase not in PHASE_SEED_IDS:
        raise ValueError(f"Unknown phase={phase!r} for seed derivation")
    # Keep within uint32 for compatibility across RNG backends.
    return (int(base_seed) + METHOD_SEED_IDS[method] * 10_000_000 + int(generation) * 10_000 + PHASE_SEED_IDS[phase]) % (
        2**32
    )


def run_generation_round(
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    encoder: AutoModel,
    encoder_tokenizer: AutoTokenizer,
    prompt_pool: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    method: str,
    ppl_ref: float | None = None,
    progress: tqdm | None = None,
) -> Dict[str, Any]:
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
    filter_args = {
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
        "ppl_ref": ppl_ref,
        "ppl_leash_tau": args.ppl_leash_tau,
        "ppl_leash_mode": args.ppl_leash_mode,
        "ppl_leash_strength": args.ppl_leash_strength,
        "ppl_safety_min_weight": args.ppl_safety_min_weight,
        "rep_filter_n": args.rep_filter_n,
        "rep_filter_threshold": args.rep_filter_threshold,
    }
    idx = select_training_indices(
        method=method,
        train_size=args.train_samples,
        ppl_scores=ppl_scores,
        embeddings=embeddings,
        texts=candidates,
        filter_args=filter_args,
        device=device,
    )
    train_texts = [candidates[i] for i in idx]
    train_quality = compute_text_quality(train_texts)
    return {
        "train_texts": train_texts,
        "train_quality": {
            "train_unique_line_ratio": train_quality.unique_line_ratio,
            "train_rep4_intra": train_quality.rep4_intra,
            "train_gzip_ratio": train_quality.gzip_ratio,
            "train_avg_words": train_quality.avg_words,
        },
        "eval_candidates": candidates[: args.eval_sample_size],
        "all_candidates": candidates,
    }


def compute_generation_metrics(
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    prompt_pool: Sequence[str],
    val_texts: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    progress: tqdm | None = None,
) -> Dict[str, float]:
    eval_texts = generate_texts(
        model=model,
        tokenizer=tokenizer,
        prompt_pool=prompt_pool,
        n_samples=args.eval_sample_size,
        batch_size=args.generation_batch,
        max_new_tokens=args.max_new_tokens,
        device=device,
        temperature=args.temperature,
        top_p=args.top_p,
        progress=progress,
    )
    d2 = distinct_n(eval_texts, 2)
    d3 = distinct_n(eval_texts, 3)
    d4 = distinct_n(eval_texts, 4)
    val_ppl = eval_validation_ppl(
        model=model,
        tokenizer=tokenizer,
        val_texts=val_texts,
        device=device,
        batch_size=args.eval_batch_size,
        max_eval=args.val_eval_size,
    )
    return {"distinct2": d2, "distinct3": d3, "distinct4": d4, "val_ppl": val_ppl}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="exp11: GPT-2 recursive collapse with set-aware mitigation.")
    parser.add_argument("--wikitext-train-size", type=int, default=50000)
    parser.add_argument("--wikitext-val-size", type=int, default=5000)
    parser.add_argument("--prompt-pool-size", type=int, default=5000)
    parser.add_argument("--candidate-pool", type=int, default=10000)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--generation-batch", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--embed-batch-size", type=int, default=512)
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
    parser.add_argument(
        "--delta-phi-scale",
        type=float,
        default=1.0,
        help="Δϕ logit-correction scale η used inside the set-aware filter; set 0.0 to disable.",
    )
    parser.add_argument(
        "--ppl-leash-tau",
        type=float,
        default=0.7,
        help="Semantic leash temperature on log-perplexity distance (smaller = stronger penalty).",
    )
    parser.add_argument(
        "--ppl-leash-mode",
        type=str,
        default="upper",
        choices=["upper", "abs", "lower"],
        help="How to penalize candidate perplexity relative to clean val_ppl reference.",
    )
    parser.add_argument(
        "--ppl-leash-strength",
        type=float,
        default=0.0,
        help="Semantic leash strength multiplier (0 disables the leash).",
    )
    parser.add_argument(
        "--ppl-leash-ref-mode",
        type=str,
        default="sliding",
        choices=["sliding", "fixed"],
        help="Reference PPL mode for the leash: sliding uses prev-generation val_ppl; fixed anchors to G0.",
    )
    parser.add_argument(
        "--mix-original-ratio",
        type=float,
        default=0.0,
        help="Ratio of original Wikitext samples to mix into each generation's training set (e.g., 0.1 for 10%%).",
    )
    # Default results under Total_results/Tables/exp11_gpt2_model/Results with semantic filename.
    parser.add_argument(
        "--results-path",
        type=pathlib.Path,
        default=RESULTS_DIR / "metrics_diversity_ppl.json",
        help="Base results path (JSON); per-seed outputs are stored under <base_dir>/<seed>/<base_name>.{json,csv}",
    )
    parser.add_argument("--seed", type=int, default=1088)
    parser.add_argument(
        "--seeds",
        type=str,
        default="1088,2195,4960",
        help="Comma-separated seeds; runs one experiment per seed and saves under <results-path parent>/<seed>/.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="no_filter,pointwise,set_aware",
        help="Comma-separated methods to run: no_filter, pointwise, ppl_leash, ppl_safety, rep_filter, dispersion, set_aware.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--rep-filter-n",
        type=int,
        default=4,
        help="N-gram size used by method=rep_filter (default: 4).",
    )
    parser.add_argument(
        "--rep-filter-threshold",
        type=float,
        default=0.6,
        help="Max intra-n-gram repetition allowed by method=rep_filter (lower => stricter).",
    )
    parser.add_argument(
        "--ppl-safety-min-weight",
        type=float,
        default=0.5,
        help="Minimum leash weight to be considered 'safe' in method=ppl_safety.",
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument(
        "--save-generations",
        action="store_true",
        help="If set, save generated candidates and selected train texts per generation to generations/<seed>/",
    )
    parser.add_argument(
        "--generations-dir",
        type=pathlib.Path,
        default=BASE_TABLES_DIR / "generations",
        help="Root directory for saved generations when --save-generations is set.",
    )
    return parser.parse_args()


def save_results_partial(history: Dict[str, List[Dict[str, float]]], args_dict: Dict, results_path: pathlib.Path) -> None:
    """
    Persist current results to JSON + CSV after each generation to avoid loss on interruption.
    """
    clean_args = {}
    for key, value in args_dict.items():
        if isinstance(value, pathlib.Path):
            clean_args[key] = str(value)
        else:
            clean_args[key] = value
    results = {"args": clean_args, "history": history}
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = results_path.with_suffix(".csv")
    base_fields = ["method", "generation", "distinct2", "distinct3", "distinct4", "val_ppl"]
    extra_fields = set()
    for records in history.values():
        for rec in records:
            extra_fields.update(rec.keys())
    extra_fields.difference_update({"generation"})
    extra_fields = sorted(extra_fields.difference(set(base_fields)))
    fieldnames = base_fields + list(extra_fields)

    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for method, records in history.items():
            for rec in records:
                row = {k: "" for k in fieldnames}
                row["method"] = method
                row["generation"] = rec.get("generation", 0)
                for k in fieldnames:
                    if k in {"method", "generation"}:
                        continue
                    if k in rec:
                        row[k] = rec[k]
                writer.writerow(row)


def main() -> None:
    args = parse_args()
    # Silence transformer advisory warnings.
    hf_logging.set_verbosity_error()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Parse seeds (comma-separated); default runs three seeds.
    seeds_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else [args.seed]
    base_results_path = args.results_path

    seed_pbar = tqdm(seeds_list, desc="Seeds", position=0, leave=True)
    for seed in seed_pbar:
        seed_pbar.set_postfix_str(f"seed={seed}")
        set_deterministic(seed)

        # Models
        tokenizer, base_model = prepare_tokenizer_model()
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        base_model.to(device)
        cache_dir = os.environ.get("HF_HUB_CACHE")
        encoder_tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True,
            cache_dir=cache_dir,
        )
        encoder_model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True,
            cache_dir=cache_dir,
        ).to(device)

        # Data
        print("Loading Wikitext-103 subsets...")
        train_texts = load_wikitext_subset("train", n_samples=args.wikitext_train_size, seed=seed)
        val_texts = load_wikitext_subset("validation", n_samples=args.wikitext_val_size, seed=seed + 1)
        prompt_pool = train_texts[: args.prompt_pool_size] if len(train_texts) >= args.prompt_pool_size else train_texts

        # Optional initial adaptation
        if args.initial_epochs > 0:
            print("Running initial adaptation on Wikitext subset...")
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

        requested = [m.strip() for m in args.methods.split(",") if m.strip()]
        allowed = {"no_filter", "pointwise", "ppl_leash", "ppl_safety", "rep_filter", "dispersion", "set_aware"}
        unknown = sorted(set(requested) - allowed)
        if unknown:
            raise ValueError(f"Unknown methods in --methods: {unknown}. Allowed: {sorted(allowed)}")
        methods = requested
        models: Dict[str, GPT2LMHeadModel] = {m: copy.deepcopy(base_model) for m in methods}
        history: Dict[str, List[Dict[str, float]]] = {m: [] for m in methods}
        prev_val_ppl: Dict[str, float | None] = {m: None for m in methods}

        args_dict = vars(args).copy()
        args_dict["results_path"] = str(base_results_path)
        args_dict["seed"] = seed

        # Derive per-seed path from user-specified base (default: Total_results/Tables/exp11_gpt2_model/Results/metrics_diversity_ppl.json)
        base_results_path = args.results_path.resolve()
        results_path_seed = base_results_path.parent / f"{seed}" / base_results_path.name
        args_dict["results_path"] = str(results_path_seed)
        # Generations output dir (optional)
        generations_dir = args.generations_dir / str(seed)
        if args.save_generations:
            generations_dir.mkdir(parents=True, exist_ok=True)

        # Initialize semantic-leash reference (clean val_ppl) before generation 0 so the leash can act immediately.
        if args.ppl_leash_strength != 0.0:
            for method in methods:
                model = models[method].to(device)
                prev_val_ppl[method] = eval_validation_ppl(
                    model=model,
                    tokenizer=tokenizer,
                    val_texts=val_texts,
                    device=device,
                    batch_size=args.eval_batch_size,
                    max_eval=args.val_eval_size,
                )
                models[method] = model.to("cpu")
                torch.cuda.empty_cache()

        gen_pbar = tqdm(range(args.generations), desc=f"Seed {seed} | Generations", position=1, leave=True)
        for gen in gen_pbar:
            for method in methods:
                # Ensure per-method determinism does not depend on which other methods are executed.
                set_deterministic(derive_phase_seed(seed, method, gen, phase="round"))
                # Phase: generate
                phase_gen = tqdm(
                    total=args.candidate_pool,
                    desc=f"G{gen} {method} | generate",
                    position=2,
                    leave=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
                )
                model = models[method].to(device)
                round_data = run_generation_round(
                    model=model,
                    tokenizer=tokenizer,
                    encoder=encoder_model,
                    encoder_tokenizer=encoder_tokenizer,
                    prompt_pool=prompt_pool,
                    args=args,
                    device=device,
                    method=method,
                    ppl_ref=prev_val_ppl.get(method),
                    progress=phase_gen,
                )
                phase_gen.close()
                if args.save_generations:
                    cand_path = generations_dir / f"g{gen}_{method}_candidates.txt"
                    with cand_path.open("w") as f_cand:
                        f_cand.write("\n".join(round_data["all_candidates"]))
                    sel_path = generations_dir / f"g{gen}_{method}_train.txt"
                    with sel_path.open("w") as f_sel:
                        f_sel.write("\n".join(round_data["train_texts"]))

                # Phase: finetune (per-batch progress inside fine_tune)
                set_deterministic(derive_phase_seed(seed, method, gen, phase="train"))
                train_texts_gen = round_data["train_texts"]
                if args.mix_original_ratio > 0.0 and len(train_texts) > 0:
                    n_mix = max(1, int(len(train_texts_gen) * args.mix_original_ratio))
                    mix_indices = np.random.choice(len(train_texts), size=n_mix, replace=False)
                    mix_samples = [train_texts[int(i)] for i in mix_indices]
                    train_texts_gen = train_texts_gen + mix_samples
                    np.random.shuffle(train_texts_gen)
                model = fine_tune(
                    model=model,
                    tokenizer=tokenizer,
                    texts=train_texts_gen,
                    device=device,
                    epochs=args.epochs_per_gen,
                    batch_size=args.train_batch_size,
                    lr=args.lr,
                    warmup_steps=args.warmup_steps,
                    max_length=args.block_size,
                    progress_desc=f"G{gen} {method} | finetune",
                )

                # Phase: eval
                set_deterministic(derive_phase_seed(seed, method, gen, phase="eval"))
                phase_eval = tqdm(
                    total=args.eval_sample_size,
                    desc=f"G{gen} {method} | eval",
                    position=2,
                    leave=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
                )
                metrics = compute_generation_metrics(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_pool=prompt_pool,
                    val_texts=val_texts,
                    args=args,
                    device=device,
                    progress=phase_eval,
                )
                if "train_quality" in round_data:
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
                    ckpt_path = ckpt_dir / f"{method}_gen{gen}.pt"
                    torch.save({"model_state": model.state_dict()}, ckpt_path)
                models[method] = model.to("cpu")
                torch.cuda.empty_cache()

            # Persist progress after each generation for this seed
            save_results_partial(history, args_dict, results_path_seed)
            print(f"Saved results to {results_path_seed}")
            print(f"Saved flat metrics to {results_path_seed.with_suffix('.csv')}")
        gen_pbar.close()
    seed_pbar.close()


if __name__ == "__main__":
    main()
