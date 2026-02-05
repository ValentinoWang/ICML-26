#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Control: NO clean-val (no meta). Evaluate on the full CIFAR-10 test split.
# - baseline_balanced uses confidence ranking + the same α class quotas
# - writes to a NEW results directory (won't overwrite previous runs)

python run_exp9_cifar10_setaware.py \
  --results-dir results_balance_alpha05_baseline_balanced_no_cleanval \
  --modes baseline_balanced \
  --seeds 1088 2195 4960 \
  --set-aware-balance-alpha 0.5

