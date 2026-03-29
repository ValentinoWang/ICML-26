#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

python Experiments/Rebuttal/run_cifar_rebuttal.py \
  --device "${DEVICE:-cuda}" \
  --results-dir Experiments/Rebuttal/results/cifar10_fraction_sweep \
  --seeds 1088 2195 4960 \
  --modes fraction_sweep \
  --fraction-sweep-alphas 0.5 0.75 0.9 0.975 1.0 \
  --generations 5 \
  --per-gen-add 4000 \
  --filter-candidate-pool 12000

python Experiments/Rebuttal/summarize_cifar_fraction_sweep.py \
  --results-dir Experiments/Rebuttal/results/cifar10_fraction_sweep \
  --out-dir Experiments/Rebuttal/results/summary_fraction_sweep \
  --generation 5 \
  --total-size 4000
