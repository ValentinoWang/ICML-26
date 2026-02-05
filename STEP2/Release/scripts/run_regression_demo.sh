#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Keep the demo CPU-only even on GPU machines (and avoid noisy CUDA probing on CPU-only setups).
export CUDA_VISIBLE_DEVICES=""

python filter/run_filter_experiment.py --cpu --out-dir results_demo "$@"
