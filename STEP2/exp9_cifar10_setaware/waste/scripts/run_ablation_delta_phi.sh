#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Δϕ ablation under the same meta + balance setting.
# Run two variants into separate output dirs so each produces a clean merged CSV.
# If re-run and a per-seed merged CSV is already complete, the Python runner will skip that seed.

COMMON_ARGS=(
  --modes set_aware
  --seeds 1088 2195 4960
  --meta-clean-val
  --clean-val-size 100
  --clean-val-strategy stratified
  --clean-val-seed 0
  --set-aware-balance-alpha 0.5
)

python STEP2/exp9_cifar10_setaware/run_exp9_cifar10_setaware.py \
  --results-dir STEP2/exp9_cifar10_setaware/results_meta_balance_alpha05_dphi0 \
  --delta-phi-scale 0.0 \
  "${COMMON_ARGS[@]}"

python STEP2/exp9_cifar10_setaware/run_exp9_cifar10_setaware.py \
  --results-dir STEP2/exp9_cifar10_setaware/results_meta_balance_alpha05_dphi1 \
  --delta-phi-scale 1.0 \
  "${COMMON_ARGS[@]}"
