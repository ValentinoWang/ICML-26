#!/usr/bin/env bash
set -euo pipefail

# Baseline (Top-Conf) run for clean-val ablation context.
# Uses the same 3x batch settings as the set_aware runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS="${SEEDS:-1088}"
CLEAN_VAL_SIZES=(${CLEAN_VAL_SIZES:-1000})
CLEAN_VAL_NOISES=(${CLEAN_VAL_NOISES:-0.0})

for SIZE in "${CLEAN_VAL_SIZES[@]}"; do
  for NOISE in "${CLEAN_VAL_NOISES[@]}"; do
    OUT="${REPO_ROOT}/Total_results/Tables/exp9_cifar10_setaware/cleanval_ablation_baseline/size${SIZE}_noise${NOISE}"
    echo "=== Baseline clean-val size=${SIZE}, noise=${NOISE} -> ${OUT} ==="
    python "$REPO_ROOT/exp9_cifar10_setaware/run_exp9_cifar10_setaware.py" \
      --results-dir "$OUT" \
      --modes baseline \
      --seeds ${SEEDS} \
      --meta-clean-val \
      --clean-val-source train_holdout \
      --clean-val-size "$SIZE" \
      --clean-val-strategy stratified \
      --clean-val-seed 0 \
      --clean-val-noise-rate "$NOISE" \
      --clean-val-noise-seed 0 \
      --batch-size 768 \
      --eval-batch-size 1536 \
      --overwrite-results
  done
done
