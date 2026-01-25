#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Control: NO clean-val (no meta). Evaluate on the full CIFAR-10 test split.
# Keep the tuned v2 set_aware configuration otherwise identical.
# - writes to a NEW results directory (won't overwrite previous runs)

python STEP2/exp9_cifar10_setaware/run_exp9_cifar10_setaware.py \
  --results-dir STEP2/exp9_cifar10_setaware/results_balance_alpha05_setaware_tuned_v2_no_cleanval \
  --modes set_aware \
  --seeds 1088 2195 4960 \
  --set-aware-balance-alpha 0.5 \
  --set-aware-score-mode weight_conf \
  --lambda-conf 1.0 \
  --lambda-balance 0.0 \
  --meta-every 5 \
  --meta-lambda 2.0 \
  --meta-inner-lr 0.1 \
  --meta-set-size 256 \
  --delta-phi-scale 1.0

