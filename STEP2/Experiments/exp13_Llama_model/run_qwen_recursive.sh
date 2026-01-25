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

RUN_DIR="$BASE_DIR/outputs/rec_seed${SEED}${SUFFIX}"
STREAM_PPL_OUT="${STREAM_PPL_OUT:-$RUN_DIR/streaming_ppl.csv}"
TMP_PPL="$RUN_DIR/streaming_ppl_tmp.csv"

export HF_HOME="/root/.cache/huggingface"
export HF_DATASETS_CACHE="/root/.cache/huggingface/datasets"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${STEP2_ROOT}:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
export UNSLOTH_DISABLE_COMPILE=1

NUM_CANDIDATES="${NUM_CANDIDATES:-3000}"
PROMPT_LEN="${PROMPT_LEN:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMP="${TEMP:-0.7}"
TOP_P="${TOP_P:-0.9}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
DEFAULT_GEN_BATCH_SIZE=12
DEFAULT_FILTER_BATCH_SIZE=12
DEFAULT_SELECT_BATCH_SIZE=12
DEFAULT_SFT_BATCH_SIZE=6
DEFAULT_GRAD_ACCUM=4
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-$DEFAULT_GEN_BATCH_SIZE}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"

FILTER_STEPS="${FILTER_STEPS:-200}"
FILTER_SET_SIZE="${FILTER_SET_SIZE:-128}"
FILTER_BATCH_SIZE="${FILTER_BATCH_SIZE:-$DEFAULT_FILTER_BATCH_SIZE}"

SELECT_K="${SELECT_K:-1000}"
PPL_KEEP_FRAC_SET_AWARE="${PPL_KEEP_FRAC_SET_AWARE:-0.8}"
PPL_KEEP_FRAC_POINTWISE="${PPL_KEEP_FRAC_POINTWISE:-1.0}"
PPL_KEEP_FRAC_DISPERSION="${PPL_KEEP_FRAC_DISPERSION:-1.0}"
SELECT_BATCH_SIZE="${SELECT_BATCH_SIZE:-$DEFAULT_SELECT_BATCH_SIZE}"

SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-$DEFAULT_SFT_BATCH_SIZE}"
GRAD_ACCUM="${GRAD_ACCUM:-$DEFAULT_GRAD_ACCUM}"
MAX_STEPS="${MAX_STEPS:-100}"

GEN_START="${GEN_START:-0}"
GEN_END="${GEN_END:-4}"

mkdir -p "$RUN_DIR"

declare -A ADAPTERS
ADAPTERS[base]="$G0_ADAPTER"
ADAPTERS[pointwise]="$G0_ADAPTER"
ADAPTERS[dispersion]="$G0_ADAPTER"
ADAPTERS[set_aware]="$G0_ADAPTER"

if [[ "$GEN_START" -gt 0 ]]; then
  for method in base pointwise dispersion set_aware; do
    resume_adapter="$RUN_DIR/adapter_g${GEN_START}_${method}"
    if [[ -d "$resume_adapter" ]]; then
      ADAPTERS[$method]="$resume_adapter"
    else
      echo "Missing resume adapter: $resume_adapter" >&2
      exit 1
    fi
  done
fi

function eval_streaming() {
  local gen="$1"
  local adapters=("$@")
  shift

  if [[ ! -f "$STREAM_PPL_OUT" ]]; then
    echo "seed,generation,method,ppl_stream,adapter_dir" > "$STREAM_PPL_OUT"
  fi

  "$PYTHON" "$BASE_DIR/eval_streaming_ppl.py" \
    --model-name "$MODEL" \
    --load-in-4bit \
    --val-jsonl "$VAL_JSONL" \
    --max-length "$MAX_SEQ_LEN" \
    --stride "$((MAX_SEQ_LEN / 2))" \
    "$@" \
    --out-csv "$TMP_PPL"

  "$PYTHON" - <<PY
import csv
from pathlib import Path

tmp = Path("$TMP_PPL")
out = Path("$STREAM_PPL_OUT")
seed = int("$SEED")
gen = int("$gen")

existing = set()
if out.exists():
    with out.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing.add((int(row["seed"]), int(row["generation"]), row["method"]))

with tmp.open() as f:
    reader = csv.DictReader(f)
    rows = list(reader)

with out.open("a") as f:
    writer = csv.writer(f)
    for row in rows:
        key = (seed, gen, row["name"])
        if key in existing:
            continue
        writer.writerow([seed, gen, row["name"], row["ppl_stream"], row["adapter_dir"]])
PY
}

for ((gen=$GEN_START; gen<$GEN_END; gen++)); do
  for method in base pointwise dispersion set_aware; do
    candidates="$RUN_DIR/candidates_g${gen}_${method}.jsonl"
    if [[ ! -f "$candidates" ]]; then
      "$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" generate \
        --model-name "$MODEL" \
        --adapter-dir "${ADAPTERS[$method]}" \
        --seed-jsonl "$SEED_JSONL" \
        --out-jsonl "$candidates" \
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

    if [[ "$method" == "set_aware" ]]; then
      filter_ckpt="$RUN_DIR/filter_g${gen}_${method}.pt"
      if [[ ! -f "$filter_ckpt" ]]; then
        "$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" train_filter \
          --model-name "$MODEL" \
          --seed-jsonl "$SEED_JSONL" \
          --candidates-jsonl "$candidates" \
          --out-ckpt "$filter_ckpt" \
          --max-seq-len "$MAX_SEQ_LEN" \
          --batch-size "$FILTER_BATCH_SIZE" \
          --set-size "$FILTER_SET_SIZE" \
          --steps "$FILTER_STEPS" \
          --seed "$SEED"
      fi
    fi

    train_jsonl="$RUN_DIR/train_g$((gen+1))_${method}.jsonl"
    if [[ "$method" == "base" ]]; then
      if [[ ! -f "$train_jsonl" ]]; then
        "$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" select \
          --mode random \
          --candidates-jsonl "$candidates" \
          --out-jsonl "$train_jsonl" \
          --select-k "$SELECT_K" \
          --seed "$SEED"
      fi
    elif [[ "$method" == "pointwise" ]]; then
      if [[ ! -f "$train_jsonl" ]]; then
        "$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" select \
          --mode pointwise \
          --model-name "$MODEL" \
          --candidates-jsonl "$candidates" \
          --out-jsonl "$train_jsonl" \
          --select-k "$SELECT_K" \
          --max-seq-len "$MAX_SEQ_LEN" \
          --batch-size "$SELECT_BATCH_SIZE" \
          --ppl-keep-frac "$PPL_KEEP_FRAC_POINTWISE" \
          --seed "$SEED"
      fi
    elif [[ "$method" == "dispersion" ]]; then
      if [[ ! -f "$train_jsonl" ]]; then
        "$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" select \
          --mode dispersion \
          --model-name "$MODEL" \
          --candidates-jsonl "$candidates" \
          --out-jsonl "$train_jsonl" \
          --select-k "$SELECT_K" \
          --max-seq-len "$MAX_SEQ_LEN" \
          --batch-size "$SELECT_BATCH_SIZE" \
          --ppl-keep-frac "$PPL_KEEP_FRAC_DISPERSION" \
          --seed "$SEED"
      fi
    else
      if [[ ! -f "$train_jsonl" ]]; then
        "$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" select \
          --mode set_aware \
          --model-name "$MODEL" \
          --filter-ckpt "$filter_ckpt" \
          --candidates-jsonl "$candidates" \
          --out-jsonl "$train_jsonl" \
          --select-k "$SELECT_K" \
          --max-seq-len "$MAX_SEQ_LEN" \
          --batch-size "$SELECT_BATCH_SIZE" \
          --ppl-keep-frac "$PPL_KEEP_FRAC_SET_AWARE" \
          --seed "$SEED"
      fi
    fi

    out_adapter="$RUN_DIR/adapter_g$((gen+1))_${method}"
    if [[ ! -d "$out_adapter" ]]; then
      "$PYTHON" "$BASE_DIR/run_exp12_oneshot.py" g1_sft \
        --model-name "$MODEL" \
        --adapter-dir "${ADAPTERS[$method]}" \
        --train-jsonl "$train_jsonl" \
        --output-dir "$out_adapter" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --batch-size "$SFT_BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --max-steps "$MAX_STEPS" \
        --seed "$SEED"
    fi

    ADAPTERS[$method]="$out_adapter"
  done

  if [[ "$gen" -eq "$GEN_START" && "$GEN_START" -eq 0 ]]; then
    eval_streaming "$gen" \
      --adapter "g0=$G0_ADAPTER"
  fi
  eval_streaming "$((gen+1))" \
    --adapter "base=${ADAPTERS[base]}" \
    --adapter "pointwise=${ADAPTERS[pointwise]}" \
    --adapter "dispersion=${ADAPTERS[dispersion]}" \
    --adapter "set_aware=${ADAPTERS[set_aware]}"
done
