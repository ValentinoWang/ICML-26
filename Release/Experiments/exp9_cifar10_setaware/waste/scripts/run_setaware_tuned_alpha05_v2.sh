#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Tuned set_aware (v2): keep w*conf scoring, keep confidence target,
# remove balance loss (handled by α), and use a moderate meta-clean-val signal.
#
# Safety:
# - writes to a NEW results directory
# - re-running skips seeds that are already complete

python run_exp9_cifar10_setaware.py \
  --results-dir results_meta_balance_alpha05_setaware_tuned_v2 \
  --modes set_aware \
  --seeds 1088 2195 4960 \
  --meta-clean-val \
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
