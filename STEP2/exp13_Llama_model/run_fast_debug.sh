#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/root/autodl-tmp/ICML/STEP2/exp13_Llama_model"
PYTHON="/root/autodl-tmp/Model/unsloth_env_sys/bin/python"
MODEL="${MODEL:-unsloth/qwen2-7b-bnb-4bit}"

SEED_JSONL="$BASE_DIR/data/seed.jsonl"
G0_ADAPTER="$BASE_DIR/outputs/g0_adapter"

CANDIDATES="$BASE_DIR/outputs/candidates_cons.jsonl"
FILTER_CKPT="$BASE_DIR/outputs/filter_ckpt_cons.pt"
TRAIN_BASE="$BASE_DIR/outputs/train_base_cons.jsonl"
TRAIN_OURS="$BASE_DIR/outputs/train_ours_cons.jsonl"

export HF_HOME="/root/autodl-tmp/Model/hf_cache"
export HF_DATASETS_CACHE="/root/autodl-tmp/Model/hf_cache/datasets"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/root/autodl-tmp/ICML/STEP2"
export TORCHDYNAMO_DISABLE=1
export UNSLOTH_DISABLE_COMPILE=1

DO_GENERATE="${DO_GENERATE:-1}"
RESET_CANDIDATES="${RESET_CANDIDATES:-0}"
NUM_CANDIDATES="${NUM_CANDIDATES:-3000}"
PROMPT_LEN="${PROMPT_LEN:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMP="${TEMP:-0.7}"
TOP_P="${TOP_P:-0.9}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"

FILTER_STEPS="${FILTER_STEPS:-200}"
FILTER_SET_SIZE="${FILTER_SET_SIZE:-128}"
FILTER_BATCH_SIZE="${FILTER_BATCH_SIZE:-4}"

SELECT_K="${SELECT_K:-1000}"
PPL_KEEP_FRAC="${PPL_KEEP_FRAC:-0.8}"
SELECT_BATCH_SIZE="${SELECT_BATCH_SIZE:-4}"

QUALITY_BATCH_SIZE="${QUALITY_BATCH_SIZE:-4}"
QUALITY_MAX_SAMPLES="${QUALITY_MAX_SAMPLES:-500}"
QUALITY_TOLERANCE="${QUALITY_TOLERANCE:-0.0}"
FAST_LOG="${FAST_LOG:-$BASE_DIR/outputs/fast_debug.log}"

if [[ "$DO_GENERATE" == "1" ]]; then
  if [[ "$RESET_CANDIDATES" == "1" ]]; then
    rm -f "$CANDIDATES"
  fi
  HF_HOME="$HF_HOME" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PYTHONPATH="$PYTHONPATH" TORCHDYNAMO_DISABLE="$TORCHDYNAMO_DISABLE" UNSLOTH_DISABLE_COMPILE="$UNSLOTH_DISABLE_COMPILE" \
  "$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" generate \
    --model-name "$MODEL" \
    --adapter-dir "$G0_ADAPTER" \
    --seed-jsonl "$SEED_JSONL" \
    --out-jsonl "$CANDIDATES" \
    --num-candidates "$NUM_CANDIDATES" \
    --prompt-len "$PROMPT_LEN" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMP" \
    --top-p "$TOP_P" \
    --repetition-penalty "$REPETITION_PENALTY" \
    --gen-batch-size "$GEN_BATCH_SIZE" \
    --max-seq-len "$MAX_SEQ_LEN"
else
  if [[ ! -f "$CANDIDATES" ]]; then
    echo "Missing candidates file: $CANDIDATES" >&2
    exit 1
  fi
fi

HF_HOME="$HF_HOME" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PYTHONPATH="$PYTHONPATH" TORCHDYNAMO_DISABLE="$TORCHDYNAMO_DISABLE" UNSLOTH_DISABLE_COMPILE="$UNSLOTH_DISABLE_COMPILE" \
"$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" train_filter \
  --model-name "$MODEL" \
  --seed-jsonl "$SEED_JSONL" \
  --candidates-jsonl "$CANDIDATES" \
  --out-ckpt "$FILTER_CKPT" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --batch-size "$FILTER_BATCH_SIZE" \
  --set-size "$FILTER_SET_SIZE" \
  --steps "$FILTER_STEPS"

PYTHONPATH="$PYTHONPATH" \
"$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" select \
  --mode random \
  --candidates-jsonl "$CANDIDATES" \
  --out-jsonl "$TRAIN_BASE" \
  --select-k "$SELECT_K"

PYTHONPATH="$PYTHONPATH" TORCHDYNAMO_DISABLE="$TORCHDYNAMO_DISABLE" UNSLOTH_DISABLE_COMPILE="$UNSLOTH_DISABLE_COMPILE" \
"$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" select \
  --mode set_aware \
  --model-name "$MODEL" \
  --filter-ckpt "$FILTER_CKPT" \
  --candidates-jsonl "$CANDIDATES" \
  --out-jsonl "$TRAIN_OURS" \
  --select-k "$SELECT_K" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --batch-size "$SELECT_BATCH_SIZE" \
  --ppl-keep-frac "$PPL_KEEP_FRAC"

{
  echo "=== FAST DEBUG $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
  echo "MODEL=$MODEL NUM_CANDIDATES=$NUM_CANDIDATES TEMP=$TEMP TOP_P=$TOP_P REPETITION_PENALTY=$REPETITION_PENALTY"
  echo "PPL_KEEP_FRAC=$PPL_KEEP_FRAC SELECT_K=$SELECT_K QUALITY_MAX_SAMPLES=$QUALITY_MAX_SAMPLES"
  HF_HOME="$HF_HOME" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PYTHONPATH="$PYTHONPATH" \
  "$PYTHON" "$BASE_DIR/inspect_data_quality.py" \
    --model-name "$MODEL" \
    --adapter-dir "$G0_ADAPTER" \
    --base-jsonl "$TRAIN_BASE" \
    --ours-jsonl "$TRAIN_OURS" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --batch-size "$QUALITY_BATCH_SIZE" \
    --max-samples "$QUALITY_MAX_SAMPLES" \
    --tolerance "$QUALITY_TOLERANCE"
} | tee -a "$FAST_LOG"
