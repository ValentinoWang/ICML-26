#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}"
STEP2_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="/root/autodl-tmp/Model/unsloth_env_sys/bin/python"
MODEL="${MODEL:-unsloth/qwen2-7b-bnb-4bit}"
SEED="${SEED:-1088}"
RUN_TAG="${RUN_TAG:-}"

SEED_JSONL="$BASE_DIR/data/seed.jsonl"
VAL_JSONL="$BASE_DIR/data/val.jsonl"
G0_ADAPTER="$BASE_DIR/outputs/g0_adapter"

SUFFIX=""
if [[ -n "$RUN_TAG" ]]; then
  SUFFIX="_${RUN_TAG}"
fi

CANDIDATES="$BASE_DIR/outputs/candidates_cons${SUFFIX}.jsonl"
FILTER_CKPT="$BASE_DIR/outputs/filter_ckpt_cons${SUFFIX}.pt"
TRAIN_BASE="$BASE_DIR/outputs/train_base_cons${SUFFIX}.jsonl"
TRAIN_POINTWISE="$BASE_DIR/outputs/train_pointwise_cons${SUFFIX}.jsonl"
TRAIN_DISPERSION="$BASE_DIR/outputs/train_dispersion_cons${SUFFIX}.jsonl"
TRAIN_OURS="$BASE_DIR/outputs/train_ours_cons${SUFFIX}.jsonl"
G1_BASE="$BASE_DIR/outputs/g1_base_adapter_cons${SUFFIX}"
G1_POINTWISE="$BASE_DIR/outputs/g1_pointwise_adapter_cons${SUFFIX}"
G1_DISPERSION="$BASE_DIR/outputs/g1_dispersion_adapter_cons${SUFFIX}"
G1_OURS="$BASE_DIR/outputs/g1_ours_adapter_cons${SUFFIX}"

export HF_HOME="/root/.cache/huggingface"
export HF_DATASETS_CACHE="/root/.cache/huggingface/datasets"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${STEP2_ROOT}:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
export UNSLOTH_DISABLE_COMPILE=1

RESET_CANDIDATES="${RESET_CANDIDATES:-1}"
NUM_CANDIDATES="${NUM_CANDIDATES:-3000}"
PROMPT_LEN="${PROMPT_LEN:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMP="${TEMP:-0.7}"
TOP_P="${TOP_P:-0.9}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"

FILTER_STEPS="${FILTER_STEPS:-200}"
FILTER_SET_SIZE="${FILTER_SET_SIZE:-128}"
FILTER_BATCH_SIZE="${FILTER_BATCH_SIZE:-8}"

SELECT_K="${SELECT_K:-1000}"
PPL_KEEP_FRAC="${PPL_KEEP_FRAC:-0.8}"
PPL_KEEP_FRAC_POINTWISE="${PPL_KEEP_FRAC_POINTWISE:-1.0}"
PPL_KEEP_FRAC_DISPERSION="${PPL_KEEP_FRAC_DISPERSION:-1.0}"
SELECT_BATCH_SIZE="${SELECT_BATCH_SIZE:-8}"

SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_STEPS="${MAX_STEPS:-100}"

QUALITY_BATCH_SIZE="${QUALITY_BATCH_SIZE:-8}"
QUALITY_MAX_SAMPLES="${QUALITY_MAX_SAMPLES:-500}"
QUALITY_TOLERANCE="${QUALITY_TOLERANCE:-0.0}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
STREAM_PPL_OUT="${STREAM_PPL_OUT:-$BASE_DIR/outputs/streaming_ppl${SUFFIX}.csv}"

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
  --max-seq-len "$MAX_SEQ_LEN" \
  --seed "$SEED"

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
  --steps "$FILTER_STEPS" \
  --seed "$SEED"

PYTHONPATH="$PYTHONPATH" \
"$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" select \
  --mode random \
  --candidates-jsonl "$CANDIDATES" \
  --out-jsonl "$TRAIN_BASE" \
  --select-k "$SELECT_K" \
  --seed "$SEED"

PYTHONPATH="$PYTHONPATH" TORCHDYNAMO_DISABLE="$TORCHDYNAMO_DISABLE" UNSLOTH_DISABLE_COMPILE="$UNSLOTH_DISABLE_COMPILE" \
"$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" select \
  --mode pointwise \
  --model-name "$MODEL" \
  --candidates-jsonl "$CANDIDATES" \
  --out-jsonl "$TRAIN_POINTWISE" \
  --select-k "$SELECT_K" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --batch-size "$SELECT_BATCH_SIZE" \
  --ppl-keep-frac "$PPL_KEEP_FRAC_POINTWISE" \
  --seed "$SEED"

PYTHONPATH="$PYTHONPATH" TORCHDYNAMO_DISABLE="$TORCHDYNAMO_DISABLE" UNSLOTH_DISABLE_COMPILE="$UNSLOTH_DISABLE_COMPILE" \
"$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" select \
  --mode dispersion \
  --model-name "$MODEL" \
  --candidates-jsonl "$CANDIDATES" \
  --out-jsonl "$TRAIN_DISPERSION" \
  --select-k "$SELECT_K" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --batch-size "$SELECT_BATCH_SIZE" \
  --ppl-keep-frac "$PPL_KEEP_FRAC_DISPERSION" \
  --seed "$SEED"

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
  --ppl-keep-frac "$PPL_KEEP_FRAC" \
  --seed "$SEED"

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

HF_HOME="$HF_HOME" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PYTHONPATH="$PYTHONPATH" TORCHDYNAMO_DISABLE="$TORCHDYNAMO_DISABLE" UNSLOTH_DISABLE_COMPILE="$UNSLOTH_DISABLE_COMPILE" \
"$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" g1_sft \
  --model-name "$MODEL" \
  --adapter-dir "$G0_ADAPTER" \
  --train-jsonl "$TRAIN_BASE" \
  --output-dir "$G1_BASE" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --batch-size "$SFT_BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --max-steps "$MAX_STEPS" \
  --seed "$SEED"

HF_HOME="$HF_HOME" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PYTHONPATH="$PYTHONPATH" TORCHDYNAMO_DISABLE="$TORCHDYNAMO_DISABLE" UNSLOTH_DISABLE_COMPILE="$UNSLOTH_DISABLE_COMPILE" \
"$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" g1_sft \
  --model-name "$MODEL" \
  --adapter-dir "$G0_ADAPTER" \
  --train-jsonl "$TRAIN_POINTWISE" \
  --output-dir "$G1_POINTWISE" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --batch-size "$SFT_BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --max-steps "$MAX_STEPS" \
  --seed "$SEED"

HF_HOME="$HF_HOME" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PYTHONPATH="$PYTHONPATH" TORCHDYNAMO_DISABLE="$TORCHDYNAMO_DISABLE" UNSLOTH_DISABLE_COMPILE="$UNSLOTH_DISABLE_COMPILE" \
"$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" g1_sft \
  --model-name "$MODEL" \
  --adapter-dir "$G0_ADAPTER" \
  --train-jsonl "$TRAIN_DISPERSION" \
  --output-dir "$G1_DISPERSION" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --batch-size "$SFT_BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --max-steps "$MAX_STEPS" \
  --seed "$SEED"

HF_HOME="$HF_HOME" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PYTHONPATH="$PYTHONPATH" TORCHDYNAMO_DISABLE="$TORCHDYNAMO_DISABLE" UNSLOTH_DISABLE_COMPILE="$UNSLOTH_DISABLE_COMPILE" \
"$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" g1_sft \
  --model-name "$MODEL" \
  --adapter-dir "$G0_ADAPTER" \
  --train-jsonl "$TRAIN_OURS" \
  --output-dir "$G1_OURS" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --batch-size "$SFT_BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --max-steps "$MAX_STEPS" \
  --seed "$SEED"

HF_HOME="$HF_HOME" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PYTHONPATH="$PYTHONPATH" \
"$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" eval \
  --model-name "$MODEL" \
  --adapter-dir "$G1_BASE" \
  --val-jsonl "$VAL_JSONL" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --batch-size "$EVAL_BATCH_SIZE"

HF_HOME="$HF_HOME" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PYTHONPATH="$PYTHONPATH" \
"$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" eval \
  --model-name "$MODEL" \
  --adapter-dir "$G1_OURS" \
  --val-jsonl "$VAL_JSONL" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --batch-size "$EVAL_BATCH_SIZE"

PYTHONPATH="$PYTHONPATH" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
"$PYTHON" "$BASE_DIR/eval_streaming_ppl.py" \
  --model-name "$MODEL" \
  --load-in-4bit \
  --val-jsonl "$VAL_JSONL" \
  --max-length "$MAX_SEQ_LEN" \
  --stride "$((MAX_SEQ_LEN / 2))" \
  --adapter "g0=$G0_ADAPTER" \
  --adapter "base=$G1_BASE" \
  --adapter "pointwise=$G1_POINTWISE" \
  --adapter "dispersion=$G1_DISPERSION" \
  --adapter "ours=$G1_OURS" \
  --out-csv "$STREAM_PPL_OUT"
