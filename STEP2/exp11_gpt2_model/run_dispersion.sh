#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Optional: avoid permission issues with HuggingFace caches by using a writable cache dir.
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python STEP2/exp11_gpt2_model/run_exp11_gpt2_model.py \
  --device cuda \
  --seeds 1088,2195,4960 \
  --methods dispersion \
  --wikitext-train-size 50000 \
  --wikitext-val-size 5000 \
  --prompt-pool-size 5000 \
  --candidate-pool 10000 \
  --train-samples 2000 \
  --generations 5 \
  --max-new-tokens 128 \
  --generation-batch 64 \
  --train-batch-size 8 \
  --eval-batch-size 8 \
  --embed-batch-size 64 \
  --embed-max-length 128 \
  --epochs-per-gen 3 \
  --lr 5e-5 \
  --warmup-steps 50 \
  --initial-epochs 1 \
  --initial-max-steps 500 \
  --temperature 0.8 \
  --top-p 0.9 \
  --eval-sample-size 512 \
  --val-eval-size 512 \
  --filter-knn 32 \
  --filter-temperature 1.0 \
  --ppl-leash-strength 0.0 \
  --delta-phi-scale 0.0 \
  --results-path STEP2/Total_results/Tables/exp11_gpt2_model/Results/dispersion/metrics_diversity_ppl.json
