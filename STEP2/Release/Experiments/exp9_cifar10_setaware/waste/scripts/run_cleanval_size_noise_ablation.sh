#!/usr/bin/env bash
set -euo pipefail

# Clean-val size + label-noise ablation for Meta/Proxy-mode (CIFAR-10).
# Default is single-seed to keep runtime modest; override SEEDS for full runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS="${SEEDS:-1088}"
CLEAN_VAL_SIZES=(${CLEAN_VAL_SIZES:-100 1000 10000})
CLEAN_VAL_NOISES=(${CLEAN_VAL_NOISES:-0.0 0.1 0.2})

for SIZE in "${CLEAN_VAL_SIZES[@]}"; do
  for NOISE in "${CLEAN_VAL_NOISES[@]}"; do
    OUT="${REPO_ROOT}/Total_results/Tables/exp9_cifar10_setaware/cleanval_ablation/size${SIZE}_noise${NOISE}"
    echo "=== Clean-val size=${SIZE}, noise=${NOISE} -> ${OUT} ==="
    python "$REPO_ROOT/exp9_cifar10_setaware/run_exp9_cifar10_setaware.py" \
      --results-dir "$OUT" \
      --modes set_aware \
      --seeds ${SEEDS} \
      --meta-clean-val \
      --clean-val-source train_holdout \
      --clean-val-size "$SIZE" \
      --clean-val-strategy stratified \
      --clean-val-seed 0 \
      --clean-val-noise-rate "$NOISE" \
      --clean-val-noise-seed 0 \
      --set-aware-balance-alpha 0.5 \
      --set-aware-score-mode weight_conf \
      --lambda-conf 1.0 \
      --lambda-balance 0.0 \
      --meta-every 5 \
      --meta-lambda 2.0 \
      --meta-inner-lr 0.1 \
      --meta-set-size 256 \
      --delta-phi-scale 1.0 \
      --batch-size 768 \
      --eval-batch-size 1536 \
      --overwrite-results
  done
done
