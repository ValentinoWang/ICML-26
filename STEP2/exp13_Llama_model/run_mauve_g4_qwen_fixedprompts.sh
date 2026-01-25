#!/usr/bin/env bash
set -euo pipefail

# Fixed-prompt MAUVE evaluation for Qwen2-7B recursive runs (G4).
# - Builds a fixed prompt set once (shared across seeds).
# - Generates G4 candidates into a dedicated folder (no overwrite).
# - Runs MAUVE and appends to a unified summary CSV.
#
# Usage:
#   bash exp13_Llama_model/run_mauve_g4_qwen_fixedprompts.sh 1
#     (runs on physical CUDA:1)

BASE_DIR="/root/autodl-tmp/ICML/STEP2/exp13_Llama_model"
PYTHON="/root/autodl-tmp/Model/unsloth_env_sys/bin/python"

CUDA_PHYSICAL_ID="${1:-0}"
RUN_TAG="${RUN_TAG:-qwen_g0_g4_b1p5}"

MODEL="${MODEL:-/root/autodl-tmp/models/Qwen2-7B}"

SEED_JSONL="${SEED_JSONL:-${BASE_DIR}/data/seed.jsonl}"
VAL_JSONL="${VAL_JSONL:-${BASE_DIR}/data/val.jsonl}"

PROMPT_SEED="${PROMPT_SEED:-25}"
PROMPT_LEN="${PROMPT_LEN:-200}"
NUM_CANDIDATES="${NUM_CANDIDATES:-3000}"
PROMPT_JSONL="${PROMPT_JSONL:-${BASE_DIR}/data/prompts_fixed_len${PROMPT_LEN}_n${NUM_CANDIDATES}_seed${PROMPT_SEED}.jsonl}"

SUMMARY_CSV="${SUMMARY_CSV:-${BASE_DIR}/outputs/mauve_g4_fixedprompts_summary.csv}"

export PYTHONPATH="/root/autodl-tmp/ICML/STEP2"
export CUDA_VISIBLE_DEVICES="${CUDA_PHYSICAL_ID}"

export HF_HOME="/root/autodl-tmp/Model/hf_cache"
export HF_DATASETS_CACHE="/root/autodl-tmp/Model/hf_cache/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export TORCHDYNAMO_DISABLE=1
export UNSLOTH_DISABLE_COMPILE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -f "${PROMPT_JSONL}" ]]; then
  echo "[prompts] building ${PROMPT_JSONL}"
  PROMPT_SEED="${PROMPT_SEED}" PROMPT_LEN="${PROMPT_LEN}" NUM_CANDIDATES="${NUM_CANDIDATES}" \
    SEED_JSONL="${SEED_JSONL}" PROMPT_JSONL="${PROMPT_JSONL}" \
    "${PYTHON}" - <<'PY'
import json
import os
import pathlib
import random

seed = int(os.environ["PROMPT_SEED"])
prompt_len = int(os.environ["PROMPT_LEN"])
num_candidates = int(os.environ["NUM_CANDIDATES"])
seed_jsonl = pathlib.Path(os.environ["SEED_JSONL"])
prompt_jsonl = pathlib.Path(os.environ["PROMPT_JSONL"])

texts = []
with seed_jsonl.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        text = obj.get("text", "")
        if text:
            texts.append(text)

prompts = [t[:prompt_len] for t in texts if len(t) > 10]
random.seed(seed)
random.shuffle(prompts)
prompts = prompts[:num_candidates]

prompt_jsonl.parent.mkdir(parents=True, exist_ok=True)
with prompt_jsonl.open("w", encoding="utf-8") as f:
    for text in prompts:
        f.write(json.dumps({"text": text}, ensure_ascii=True) + "\n")
print(f"[prompts] wrote {len(prompts)} prompts to {prompt_jsonl}")
PY
fi

if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "seed,generation,method,mauve,mauve_std,mauve_star,divergence,p_samples,q_samples,num_buckets,p_file" > "${SUMMARY_CSV}"
fi

for SEED in 1088 2195 4960; do
  RUN_DIR="${BASE_DIR}/outputs/rec_seed${SEED}_${RUN_TAG}"
  FIXED_DIR="${RUN_DIR}/fixedprompts_g4"
  MAUVE_OUT="${FIXED_DIR}/mauve_g4_b25_fixedprompts.csv"

  if [[ ! -d "${RUN_DIR}" ]]; then
    echo "Missing RUN_DIR: ${RUN_DIR}" >&2
    exit 1
  fi

  mkdir -p "${FIXED_DIR}"

  for method in base pointwise dispersion set_aware; do
    adapter="${RUN_DIR}/adapter_g4_${method}"
    if [[ ! -d "${adapter}" ]]; then
      echo "Missing adapter: ${adapter}" >&2
      exit 1
    fi
  done

  for method in base pointwise dispersion set_aware; do
    out="${FIXED_DIR}/candidates_g4_${method}.jsonl"
    if [[ -f "${out}" ]]; then
      echo "[skip] ${out}"
      continue
    fi
    echo "[gen] seed=${SEED} method=${method} -> ${out}"
    "${PYTHON}" "${BASE_DIR}/run_exp12_oneshot.py" generate \
      --model-name "${MODEL}" \
      --adapter-dir "${RUN_DIR}/adapter_g4_${method}" \
      --seed-jsonl "${PROMPT_JSONL}" \
      --out-jsonl "${out}" \
      --num-candidates "${NUM_CANDIDATES}" \
      --prompt-len "${PROMPT_LEN}" \
      --max-new-tokens "${MAX_NEW_TOKENS:-128}" \
      --temperature "${TEMP:-0.7}" \
      --top-p "${TOP_P:-0.9}" \
      --repetition-penalty "${REPETITION_PENALTY:-1.1}" \
      --gen-batch-size "${GEN_BATCH_SIZE:-12}" \
      --max-seq-len "${MAX_SEQ_LEN:-1024}" \
      --seed "${SEED}"
  done

  echo "[mauve] seed=${SEED} -> ${MAUVE_OUT}"
  "${PYTHON}" "${BASE_DIR}/mauve_eval_qwen.py" \
    --run-dir "${FIXED_DIR}" \
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

  # Append to unified summary if this seed is not already present.
  SEED="${SEED}" MAUVE_OUT="${MAUVE_OUT}" SUMMARY_CSV="${SUMMARY_CSV}" \
    "${PYTHON}" - <<'PY'
import csv
import os
import pathlib

seed = os.environ["SEED"]
mauve_out = pathlib.Path(os.environ["MAUVE_OUT"])
summary = pathlib.Path(os.environ["SUMMARY_CSV"])

existing_seeds = set()
with summary.open("r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        existing_seeds.add(row.get("seed"))

if seed in existing_seeds:
    print(f"[summary] seed {seed} already present, skip append")
else:
    with mauve_out.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = ["seed"] + reader.fieldnames
    with summary.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            row["seed"] = seed
            writer.writerow(row)
    print(f"[summary] appended seed {seed} -> {summary}")
PY
done

echo "done: ${SUMMARY_CSV}"
