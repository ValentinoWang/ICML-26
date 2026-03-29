import argparse
import os
import pathlib
import subprocess
import sys
from typing import List


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RUNNER = SCRIPT_DIR / "run_gpt2_rebuttal.py"
MAUVE_EVAL = REPO_ROOT / "Experiments" / "exp11_gpt2_model" / "mauve_eval.py"
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "gpt2_cleanval_size"


def parse_sizes(text: str) -> List[int]:
    sizes = sorted({int(part.strip()) for part in str(text).split(",") if part.strip()})
    if not sizes:
        raise ValueError("At least one clean-val size is required.")
    if any(size < 0 for size in sizes):
        raise ValueError("Clean-val sizes must be non-negative.")
    return sizes


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Sweep GPT-2 rebuttal runs over clean validation sizes.")
    parser.add_argument("--clean-val-sizes", type=str, default="50,100,500,1000")
    parser.add_argument("--results-root", type=pathlib.Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--methods", type=str, default="set_aware")
    parser.add_argument("--seeds", type=str, default="1088,2195,4960")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--wikitext-val-size",
        type=int,
        default=900,
        help="Validation pool size passed to run_gpt2_rebuttal. Keep clean-val-size + wikitext-val-size within the available strict-split chunk budget.",
    )
    parser.add_argument("--run-mauve", action="store_true")
    parser.add_argument("--mauve-ref-size", type=int, default=1000)
    parser.add_argument("--mauve-batch-size", type=int, default=32)
    parser.add_argument("--smoke-mode", action="store_true")
    args, extra = parser.parse_known_args()
    return args, extra


def main() -> None:
    args, extra = parse_args()
    sizes = parse_sizes(args.clean_val_sizes)
    results_root = pathlib.Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env["OMP_NUM_THREADS"] = "1"
    child_env["MKL_NUM_THREADS"] = "1"
    child_env["OPENBLAS_NUM_THREADS"] = "1"
    child_env["NUMEXPR_NUM_THREADS"] = "1"
    child_env["TOKENIZERS_PARALLELISM"] = "false"
    child_env["TRANSFORMERS_NO_TF"] = "1"
    child_env["USE_TF"] = "0"

    for clean_val_size in sizes:
        run_root = results_root / f"n{clean_val_size}"
        run_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(RUNNER),
            "--device",
            args.device,
            "--seeds",
            args.seeds,
            "--methods",
            args.methods,
            "--results-path",
            str(run_root / "metrics_diversity_ppl.json"),
            "--generations-dir",
            str(run_root / "generations"),
            "--clean-ref-size",
            "0",
            "--clean-val-size",
            str(clean_val_size),
            "--strict-clean-val-split",
            "--validation-unit",
            "chunks",
            "--wikitext-val-size",
            str(args.wikitext_val_size),
            "--clean-val-geom-scale",
            "1.0",
            "--candidate-pool",
            "10000",
            "--train-samples",
            "2000",
            "--generations",
            "5",
            "--temperature",
            "0.8",
            "--top-p",
            "0.9",
            "--ppl-leash-strength",
            "1.0",
        ]
        cmd.append("--save-generations")
        cmd.append("--save-checkpoints")
        if args.smoke_mode:
            cmd.append("--smoke-mode")
        cmd.extend(extra)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO_ROOT, env=child_env, check=True)

        if not args.run_mauve:
            continue

        mauve_cmd = [
            sys.executable,
            str(MAUVE_EVAL),
            "--generations-dir",
            str(run_root / "generations"),
            "--output-dir",
            str(run_root / "mauve"),
            "--seeds",
            args.seeds,
            "--methods",
            args.methods,
            "--min-gen",
            "0",
            "--max-gen",
            "4",
            "--ref-size",
            str(args.mauve_ref_size),
            "--batch-size",
            str(args.mauve_batch_size),
        ]
        print(" ".join(mauve_cmd), flush=True)
        subprocess.run(mauve_cmd, cwd=REPO_ROOT, env=child_env, check=True)


if __name__ == "__main__":
    main()
