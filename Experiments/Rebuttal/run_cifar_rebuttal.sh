#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

python Experiments/Rebuttal/run_cifar_rebuttal.py \
  --device "${DEVICE:-cuda}" \
  --seeds 1088 2195 4960 \
  --modes set_aware clean_only clean_ft clean_mix batch_mean \
  --clean-set-size 100 \
  --generations 5 \
  --per-gen-add 4000 \
  --filter-candidate-pool 12000 \
  --set-aware-balance-alpha 0.3
