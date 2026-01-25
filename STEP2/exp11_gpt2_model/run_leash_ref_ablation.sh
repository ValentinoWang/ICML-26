#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Use the system HF cache so we reuse pre-downloaded models/datasets.
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SEEDS="${SEEDS:-1088}"

COMMON_ARGS=(
  --device cuda
  --seeds "${SEEDS}"
  --methods set_aware
  --wikitext-train-size 50000
  --wikitext-val-size 5000
  --prompt-pool-size 5000
  --candidate-pool 10000
  --train-samples 2000
  --generations 5
  --max-new-tokens 128
  --generation-batch 640
  --train-batch-size 80
  --eval-batch-size 80
  --embed-batch-size 640
  --embed-max-length 128
  --epochs-per-gen 3
  --lr 5e-5
  --warmup-steps 50
  --initial-epochs 1
  --initial-max-steps 500
  --temperature 0.8
  --top-p 0.9
  --eval-sample-size 512
  --val-eval-size 512
  --filter-set-size 1024
  --filter-steps 200
  --filter-lr 1e-3
  --filter-hidden 128
  --filter-heads 4
  --filter-layers 2
  --filter-dropout 0.1
  --filter-knn 32
  --filter-temperature 1.0
  --filter-ess-tau 256.0
  --filter-ess-weight 0.01
  --delta-phi-scale 1.0
  --ppl-leash-mode upper
  --ppl-leash-strength 1.0
  --ppl-leash-tau 0.7
)

SLIDING_OUT="${REPO_ROOT}/STEP2/Total_results/Tables/exp11_gpt2_model/Results/leash_ref_sliding/metrics_diversity_ppl.json"
FIXED_OUT="${REPO_ROOT}/STEP2/Total_results/Tables/exp11_gpt2_model/Results/leash_ref_fixed/metrics_diversity_ppl.json"

python STEP2/exp11_gpt2_model/run_exp11_gpt2_model.py \
  "${COMMON_ARGS[@]}" \
  --ppl-leash-ref-mode sliding \
  --results-path "${SLIDING_OUT}"

python STEP2/exp11_gpt2_model/run_exp11_gpt2_model.py \
  "${COMMON_ARGS[@]}" \
  --ppl-leash-ref-mode fixed \
  --results-path "${FIXED_OUT}"
