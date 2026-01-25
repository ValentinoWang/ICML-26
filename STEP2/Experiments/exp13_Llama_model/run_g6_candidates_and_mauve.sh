#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}"
STEP2_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_DIR="$BASE_DIR/outputs/rec_seed1088_qwen_g0_g4_b1p5"
PYTHON="/root/autodl-tmp/Model/unsloth_env_sys/bin/python"
MODEL="/root/autodl-tmp/models/Qwen2-7B"

SEED_JSONL="$BASE_DIR/data/seed.jsonl"
VAL_JSONL="$BASE_DIR/data/val.jsonl"

export HF_HOME="/root/.cache/huggingface"
export HF_DATASETS_CACHE="/root/.cache/huggingface/datasets"
export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="${STEP2_ROOT}:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
export UNSLOTH_DISABLE_COMPILE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_CANDIDATES=3000
PROMPT_LEN=200
MAX_NEW_TOKENS=128
TEMP=0.7
TOP_P=0.9
REPETITION_PENALTY=1.1
GEN_BATCH_SIZE=1
MAX_SEQ_LEN=1024
SEED=1088

for method in base pointwise dispersion set_aware; do
  out="$RUN_DIR/candidates_g6_${method}.jsonl"
  if [[ ! -f "$out" ]]; then
    "$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" generate \
      --model-name "$MODEL" \
      --adapter-dir "$RUN_DIR/adapter_g6_${method}" \
      --seed-jsonl "$SEED_JSONL" \
      --out-jsonl "$out" \
      --num-candidates "$NUM_CANDIDATES" \
      --prompt-len "$PROMPT_LEN" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMP" \
      --top-p "$TOP_P" \
      --repetition-penalty "$REPETITION_PENALTY" \
      --gen-batch-size "$GEN_BATCH_SIZE" \
      --max-seq-len "$MAX_SEQ_LEN" \
      --seed "$SEED"
  fi
done

"$PYTHON" "$BASE_DIR/mauve_eval_qwen.py" \
  --run-dir "$RUN_DIR" \
  --val-jsonl "$VAL_JSONL" \
  --gens 5,6 \
  --methods base,pointwise,dispersion,set_aware \
  --out-csv "$RUN_DIR/mauve_g5_g6_b25.csv" \
  --num-buckets 25 \
  --device-id 0 \
  --max-text-length 1024 \
  --batch-size 1 \
  --kmeans-num-redo 5 \
  --kmeans-max-iter 500 \
  --seed 25
