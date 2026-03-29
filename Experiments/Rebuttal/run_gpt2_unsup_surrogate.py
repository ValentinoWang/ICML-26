import argparse
import os
import pathlib
import subprocess
import sys


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RUNNER = SCRIPT_DIR / "run_gpt2_rebuttal.py"
MAUVE_EVAL = REPO_ROOT / "Experiments" / "exp11_gpt2_model" / "mauve_eval.py"
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "gpt2_unsup_surrogate"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run GPT-2 unsupervised surrogate rebuttal experiments.")
    parser.add_argument("--results-root", type=pathlib.Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--methods", type=str, default="unsup_set_aware,no_filter,kl_reg,pointwise")
    parser.add_argument("--seeds", type=str, default="1088,2195,4960")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run-mauve", action="store_true")
    parser.add_argument("--mauve-ref-size", type=int, default=1000)
    parser.add_argument("--mauve-batch-size", type=int, default=32)
    parser.add_argument("--smoke-mode", action="store_true")
    args, extra = parser.parse_known_args()
    return args, extra


def main() -> None:
    args, extra = parse_args()
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
        str(results_root / "metrics_diversity_ppl.json"),
        "--generations-dir",
        str(results_root / "generations"),
        "--eval-split",
        "test",
        "--clean-support-split",
        "validation",
        "--clean-ref-size",
        "0",
        "--clean-val-size",
        "0",
        "--clean-val-geom-scale",
        "0.0",
        "--ppl-leash-strength",
        "0.0",
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
        "--save-generations",
        "--save-checkpoints",
    ]
    if args.smoke_mode:
        cmd.append("--smoke-mode")
    cmd.extend(extra)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=child_env, check=True)

    if not args.run_mauve:
        return

    mauve_cmd = [
        sys.executable,
        str(MAUVE_EVAL),
        "--generations-dir",
        str(results_root / "generations"),
        "--output-dir",
        str(results_root / "mauve"),
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
        "--model-name",
        "gpt2",
        "--reference-split",
        "test",
    ]
    print(" ".join(mauve_cmd), flush=True)
    subprocess.run(mauve_cmd, cwd=REPO_ROOT, env=child_env, check=True)


if __name__ == "__main__":
    main()
