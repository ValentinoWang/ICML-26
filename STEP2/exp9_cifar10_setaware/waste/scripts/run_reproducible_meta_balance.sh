#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Main ICML-facing run for Exp9:
# - meta clean-val (Scheme A) to give the filter a "correctness" signal
# - stratified clean-val split to cover all classes
# - selection-time class-balance constraint to prevent confirmation-bias collapse
#
# Note: This writes into a NEW results directory (does not overwrite the old Exp9 results).
# If re-run and the per-seed merged CSV is already complete, the Python runner will skip that seed.

python STEP2/exp9_cifar10_setaware/run_exp9_cifar10_setaware.py \
  --results-dir STEP2/exp9_cifar10_setaware/results_meta_balance_alpha05 \
  --modes baseline set_aware \
  --seeds 1088 2195 4960 \
  --meta-clean-val \
  --clean-val-size 100 \
  --clean-val-strategy stratified \
  --clean-val-seed 0 \
  --set-aware-balance-alpha 0.5 \
  --delta-phi-scale 1.0
