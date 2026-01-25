#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Control: clean-val comes from TRAIN holdout (unlabeled pool), NOT from test.
# This avoids using the test split for meta-training and evaluates on the full test set.
# - baseline_balanced itself doesn't use clean-val, but we still pass --meta-clean-val so
#   the evaluation split matches the set_aware meta runs.
# - writes to a NEW results directory (won't overwrite previous runs)

python STEP2/exp9_cifar10_setaware/run_exp9_cifar10_setaware.py \
  --results-dir STEP2/exp9_cifar10_setaware/results_meta_balance_alpha05_baseline_balanced_train_holdout_cleanval \
  --modes baseline_balanced \
  --seeds 1088 2195 4960 \
  --meta-clean-val \
  --clean-val-source train_holdout \
  --clean-val-size 100 \
  --clean-val-strategy stratified \
  --clean-val-seed 0 \
  --set-aware-balance-alpha 0.5

