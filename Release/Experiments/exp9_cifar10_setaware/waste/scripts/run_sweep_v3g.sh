#!/usr/bin/env bash
set -euo pipefail

# Grid search for score_topk hyperparams on CIFAR exp9 set_aware (single seed=1088).
# Symmetric around current best (floor=0.4, k=400, add=4000).
FLOORS=(0.30 0.35 0.40 0.45 0.50)
KS=(200 300 400 500 600)
ADDS=(3000 3500 4000 4500 5000)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP9_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${EXP9_DIR}"

for F in "${FLOORS[@]}"; do
  for K in "${KS[@]}"; do
    for ADD in "${ADDS[@]}"; do
      OUT="grid/results_sweep/floor${F}_k${K}_add${ADD}"
      echo "=== Running floor=${F}, k=${K}, add=${ADD} -> ${OUT} ==="
      python run_exp9_cifar10_setaware.py \
        --results-dir "$OUT" \
        --modes set_aware \
        --seeds 1088 \
        --meta-clean-val \
        --clean-val-source train_holdout \
        --clean-val-size 100 \
        --clean-val-strategy stratified \
        --clean-val-seed 0 \
        --set-aware-balance-alpha 0.5 \
        --set-aware-score-mode weight_conf \
        --set-aware-conf-threshold 0.0 \
        --set-aware-threshold-mode score_topk \
        --set-aware-score-floor "$F" \
        --set-aware-per-class-k "$K" \
        --per-gen-add "$ADD" \
        --lambda-conf 1.2 \
        --lambda-proto 0.6 \
        --proto-topk 200 \
        --proto-temp 0.2 \
        --proto-conf-power 1.0 \
        --lambda-balance 0.0 \
        --meta-every 5 \
        --meta-lambda 1.5 \
        --meta-inner-lr 0.1 \
        --meta-set-size 256 \
        --delta-phi-scale 1.0 \
        --overwrite-results
    done
  done
done
