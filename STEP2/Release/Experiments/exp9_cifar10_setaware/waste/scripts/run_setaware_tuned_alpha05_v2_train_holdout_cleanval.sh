#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Control: clean-val comes from TRAIN holdout (unlabeled pool), NOT from test.
# This avoids using the test split for meta-training and evaluates on the full test set.
# - writes to a NEW results directory (won't overwrite previous runs)

python run_exp9_cifar10_setaware.py \
  --results-dir results_meta_balance_alpha05_setaware_tuned_v2_train_holdout_cleanval \
  --modes set_aware \
  --seeds 1088 2195 4960 \
  --meta-clean-val \
  --clean-val-source train_holdout \
  --clean-val-size 100 \
  --clean-val-strategy stratified \
  --clean-val-seed 0 \
  --set-aware-balance-alpha 0.5 \
  --set-aware-score-mode weight_conf \
  --lambda-conf 1.0 \
  --lambda-balance 0.0 \
  --meta-every 5 \
  --meta-lambda 2.0 \
  --meta-inner-lr 0.1 \
  --meta-set-size 256 \
  --delta-phi-scale 1.0

