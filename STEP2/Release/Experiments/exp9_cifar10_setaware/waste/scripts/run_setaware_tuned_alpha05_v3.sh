#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Tuned set_aware (v3g): per-class score top-k + global safety floor.
# Key tweaks:
# - score = weight * confidence (unchanged)
# - score_topk keep-mask: per-class top-k after score floor
# - keep alpha=0.5 balance to protect worst-class
# - stronger confidence anchor (lambda_conf=1.2)
# - keep train-holdout clean-val (no test leakage)
#
# Safety:
# - writes to a NEW results directory
# - re-running skips seeds that are already complete

python run_exp9_cifar10_setaware.py \
  --results-dir results_meta_balance_alpha05_setaware_tuned_v3g \
  --modes set_aware \
  --seeds 1088 2195 4960 \
  --meta-clean-val \
  --clean-val-source train_holdout \
  --clean-val-size 100 \
  --clean-val-strategy stratified \
  --clean-val-seed 0 \
  --set-aware-balance-alpha 0.5 \
  --set-aware-score-mode weight_conf \
  --set-aware-conf-threshold 0.0 \
  --set-aware-threshold-mode score_topk \
  --set-aware-score-floor 0.4 \
  --set-aware-per-class-k 400 \
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
  --delta-phi-scale 1.0
