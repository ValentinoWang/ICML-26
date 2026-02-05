#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Run only the stronger fair baseline:
# - confidence ranking
# - same candidate pool size as set-aware
# - same class-balance quotas (α) used by set-aware selection
# - same clean-val split options so the evaluated test set matches the meta-clean-val runs
#
# This does NOT overwrite any previous results; it writes to a new directory.
# Re-running will skip seeds that are already complete.

python run_exp9_cifar10_setaware.py \
  --results-dir results_meta_balance_alpha05_baseline_balanced \
  --modes baseline_balanced \
  --seeds 1088 2195 4960 \
  --meta-clean-val \
  --clean-val-size 100 \
  --clean-val-strategy stratified \
  --clean-val-seed 0 \
  --set-aware-balance-alpha 0.5

