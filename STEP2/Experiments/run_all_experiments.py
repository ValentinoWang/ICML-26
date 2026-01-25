from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path, *, dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"$ (cd {cwd} && {printable})")
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run experiment entrypoints under STEP2/Experiments.\n\n"
            "Note: some experiments are heavy (e.g., exp9/exp11/exp13) and may require extra deps/GPU."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only these experiment folders (e.g. --only exp1_bias_sources exp2_bias_sensitivity).",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Skip these experiment folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    root = Path(__file__).resolve().parent

    experiments: list[tuple[str, list[str]]] = [
        ("exp1_bias_sources", [sys.executable, str(root / "exp1_bias_sources" / "run_exp1_bias_sources.py")]),
        ("exp2_bias_sensitivity", [sys.executable, str(root / "exp2_bias_sensitivity" / "run_exp2_bias_sensitivity.py")]),
        ("exp3_data_efficiency", [sys.executable, str(root / "exp3_data_efficiency" / "run_exp3_data_efficiency.py")]),
        (
            "exp4_bias_correction_visualization",
            [sys.executable, str(root / "exp4_bias_correction_visualization" / "run_exp4_bias_correction_visualization.py")],
        ),
        ("exp5_high_dim_scalability", [sys.executable, str(root / "exp5_high_dim_scalability" / "run_exp5_high_dim_scalability.py")]),
        ("exp6_arch_ablation", [sys.executable, str(root / "exp6_arch_ablation" / "run_exp6_arch_ablation.py")]),
        ("exp6_arch_ablation_gated", [sys.executable, str(root / "exp6_arch_ablation_gated" / "run_exp6_arch_ablation.py")]),
        ("exp7_recursive_regression", [sys.executable, str(root / "exp7_recursive_regression" / "run_exp7_recursive_regression.py")]),
        ("exp7_variance_attention", [sys.executable, str(root / "exp7_variance_attention" / "run_exp7_variance_attention.py")]),
        ("exp8_mnist_recursive", [sys.executable, str(root / "exp8_mnist_recursive" / "run_exp8_mnist_recursive.py")]),
        # Heavier / external-dep experiments (opt-in via --only):
        ("exp9_cifar10_setaware", [sys.executable, str(root / "exp9_cifar10_setaware" / "run_exp9_cifar10_setaware.py")]),
        ("exp10_time_cost", [sys.executable, str(root / "exp10_time_cost" / "run_exp10_time_cost.py")]),
        ("exp11_gpt2_model", [sys.executable, str(root / "exp11_gpt2_model" / "run_exp11_gpt2_model.py")]),
        ("exp12_Embedding_model", [sys.executable, str(root / "exp12_Embedding_model" / "run_exp12_embedding_topology.py")]),
        ("exp13_Llama_model", [sys.executable, str(root / "exp13_Llama_model" / "run_exp12_oneshot.py")]),
    ]
    exp_map = {name: cmd for name, cmd in experiments}
    default_selected = [
        "exp1_bias_sources",
        "exp2_bias_sensitivity",
        "exp3_data_efficiency",
        "exp4_bias_correction_visualization",
        "exp5_high_dim_scalability",
        "exp6_arch_ablation",
        "exp6_arch_ablation_gated",
        "exp7_recursive_regression",
        "exp7_variance_attention",
        "exp8_mnist_recursive",
    ]

    if args.only is None:
        selected = default_selected
    else:
        selected = args.only

    selected = [name for name in selected if name not in set(args.skip)]

    unknown = [name for name in selected if name not in exp_map]
    if unknown:
        known = ", ".join(name for name, _ in experiments)
        raise SystemExit(f"Unknown experiments: {', '.join(unknown)}\nKnown: {known}")

    for name in selected:
        run(exp_map[name], cwd=root, dry_run=args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
