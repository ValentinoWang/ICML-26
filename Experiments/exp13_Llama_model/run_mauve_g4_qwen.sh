#!/usr/bin/env bash
set -euo pipefail

# Compute MAUVE for Qwen2-7B recursive runs at Generation 4.
# - Generates missing candidates_g4_{method}.jsonl from adapter_g4_{method}
# - Runs MAUVE against the fixed validation distribution (val.jsonl)
#
# Usage:
#   bash exp13_Llama_model/run_mauve_g4_qwen.sh 2195 1
#     (seed=2195, run on physical CUDA:1)
#
# Notes:
# - We set CUDA_VISIBLE_DEVICES to a single GPU, so MAUVE's --device-id must be 0.
# - Set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=0 if you intentionally want network access.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}"
STEP2_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="/root/autodl-tmp/Model/unsloth_env_sys/bin/python"

SEED="${1:-2195}"
CUDA_PHYSICAL_ID="${2:-1}"

# Local model path (avoids HF network checks).
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen2-7B}"

RUN_TAG="${RUN_TAG:-qwen_g0_g4_b1p5}"
RUN_DIR="${BASE_DIR}/outputs/rec_seed${SEED}_${RUN_TAG}"

SEED_JSONL="${BASE_DIR}/data/seed.jsonl"
VAL_JSONL="${BASE_DIR}/data/val.jsonl"

export PYTHONPATH="${STEP2_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_PHYSICAL_ID}"

export HF_HOME="/root/.cache/huggingface"
export HF_DATASETS_CACHE="/root/.cache/huggingface/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export TORCHDYNAMO_DISABLE=1
export UNSLOTH_DISABLE_COMPILE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_CANDIDATES="${NUM_CANDIDATES:-3000}"
PROMPT_LEN="${PROMPT_LEN:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMP="${TEMP:-0.7}"
TOP_P="${TOP_P:-0.9}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-12}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"

MAUVE_OUT="${MAUVE_OUT:-${RUN_DIR}/mauve_g4_b25.csv}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Missing RUN_DIR: ${RUN_DIR}" >&2
  exit 1
fi

for method in base pointwise dispersion set_aware; do
  adapter="${RUN_DIR}/adapter_g4_${method}"
  if [[ ! -d "${adapter}" ]]; then
    echo "Missing adapter: ${adapter}" >&2
    exit 1
  fi
done

for method in base pointwise dispersion set_aware; do
  out="${RUN_DIR}/candidates_g4_${method}.jsonl"
  if [[ -f "${out}" ]]; then
    echo "[skip] ${out}"
    continue
  fi
  echo "[gen] G4 ${method} -> ${out}"
  "${PYTHON}" "${BASE_DIR}/run_exp12_oneshot.py" generate \
    --model-name "${MODEL}" \
    --adapter-dir "${RUN_DIR}/adapter_g4_${method}" \
    --seed-jsonl "${SEED_JSONL}" \
    --out-jsonl "${out}" \
    --num-candidates "${NUM_CANDIDATES}" \
    --prompt-len "${PROMPT_LEN}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMP}" \
    --top-p "${TOP_P}" \
    --repetition-penalty "${REPETITION_PENALTY}" \
    --gen-batch-size "${GEN_BATCH_SIZE}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --seed "${SEED}"
done

echo "[mauve] writing ${MAUVE_OUT}"
"${PYTHON}" "${BASE_DIR}/mauve_eval_qwen.py" \
  --run-dir "${RUN_DIR}" \
  --val-jsonl "${VAL_JSONL}" \
  --gens 4 \
  --methods base,pointwise,dispersion,set_aware \
  --out-csv "${MAUVE_OUT}" \
  --num-buckets 25 \
  --device-id 0 \
  --max-text-length 1024 \
  --batch-size 1 \
  --kmeans-num-redo 5 \
  --kmeans-max-iter 500 \
  --seed 25

echo "done: ${MAUVE_OUT}"
