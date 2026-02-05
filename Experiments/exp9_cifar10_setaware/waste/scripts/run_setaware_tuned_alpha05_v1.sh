#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Tuned set_aware aiming to beat baseline_balanced (Top-conf + α quotas) by learning *within-class*
# cleanliness weights via stronger meta-clean-val signal and reduced reliance on confidence targets.
#
# Safety:
# - writes to a NEW results directory
# - re-running skips seeds that are already complete

python run_exp9_cifar10_setaware.py \
  --results-dir results_meta_balance_alpha05_setaware_tuned_v1 \
  --modes set_aware \
  --seeds 2195 \
  --meta-clean-val \
  --clean-val-size 100 \
  --clean-val-strategy stratified \
  --clean-val-seed 0 \
  --set-aware-balance-alpha 0.5 \
  --set-aware-score-mode weight \
  --lambda-conf 0.1 \
  --lambda-balance 0.0 \
  --meta-every 1 \
  --meta-lambda 10.0 \
  --meta-inner-lr 0.2 \
  --meta-set-size 512 \
  --delta-phi-scale 1.0

