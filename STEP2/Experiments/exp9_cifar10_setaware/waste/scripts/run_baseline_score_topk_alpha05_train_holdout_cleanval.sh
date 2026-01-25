#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Strict control for v3g: use the SAME score_topk keep-mask (floor + per-class-k) and the SAME
# selection-time class-balance α, but score candidates by confidence only (no set-aware learned weights).
#
# Clean-val comes from TRAIN holdout (unlabeled pool), NOT from test (avoids leakage).

python run_exp9_cifar10_setaware.py \
  --results-dir results_meta_balance_alpha05_baseline_score_topk_train_holdout_cleanval \
  --modes baseline_score_topk \
  --seeds 1088 2195 4960 \
  --meta-clean-val \
  --clean-val-source train_holdout \
  --clean-val-size 100 \
  --clean-val-strategy stratified \
  --clean-val-seed 0 \
  --set-aware-balance-alpha 0.5 \
  --set-aware-score-floor 0.4 \
  --set-aware-per-class-k 400 \
  --baseline-conf-threshold 0.0

