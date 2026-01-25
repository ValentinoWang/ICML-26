import subprocess
import sys
from pathlib import Path


def run(cmd, cwd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    root = Path(__file__).resolve().parent

    # Experiment 1: stability under bias (set-aware filter)
    exp1_dir = root / "exp1_bias_stability"
    exp1_cmd = [
        sys.executable,
        str(exp1_dir / "run_exp1_bias_stability.py"),
        "--use-set-filter",
        "--samples-per-gen",
        "200",
        "--noise-std",
        "0.2",
        "--generations",
        "1000",
        "--out-dir",
        str(exp1_dir / "results_set_filter_via_run_exp1"),
    ]
    run(exp1_cmd, cwd=root)

    # Experiment 2: bias sensitivity sweep
    exp2_dir = root / "exp2_bias_sensitivity"
    exp2_cmd = [
        sys.executable,
        str(exp2_dir / "run_exp2_bias_sensitivity.py"),
    ]
    run(exp2_cmd, cwd=root)

    # Experiment 3: contraction vs. sample size (linear, fixed, superlinear)
    exp3_dir = root / "exp3_contraction_vs_samples"
    exp3_cmd = [
        sys.executable,
        str(exp3_dir / "run_exp3_contraction_vs_samples.py"),
        "--save-csv",
    ]
    run(exp3_cmd, cwd=root)

    # Experiment 4: bias correction visualization (stronger correction settings)
    exp4_dir = root / "exp4_bias_correction_visualization"
    exp4_cmd = [
        sys.executable,
        str(exp4_dir / "run_exp4_bias_correction_visualization.py"),
        "--correction-clip",
        "1.0",
        "--ours-contraction",
        "0.6",
        "--lambda-contract",
        "0.5",
        "--calibration-size",
        "200",
        "--top-ratio",
        "0.5",
        "--save-csv",
    ]
    run(exp4_cmd, cwd=root)

    print("All experiments completed.")


if __name__ == "__main__":
    main()
