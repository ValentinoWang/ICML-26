#!/usr/bin/env bash
set -euo pipefail

# Fixed-prompt MAUVE evaluation for GPT-2 G4.
# - Generates fixed prompts & fixed reference (once).
# - Generates G4 candidates from saved checkpoints into a separate directory.
# - Runs MAUVE into a separate output directory (no overwrite).

ROOT="/root/autodl-tmp/ICML/STEP2"
BASE_DIR="${ROOT}/exp11_gpt2_model"
PYTHON="${PYTHON:-python}"

CUDA_PHYSICAL_ID="${1:-0}"

export PYTHONPATH="${ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_PHYSICAL_ID}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/Model/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PROMPT_SEED="${PROMPT_SEED:-25}"
PROMPT_SIZE="${PROMPT_SIZE:-5000}"
REF_SEED="${REF_SEED:-25}"
REF_SIZE="${REF_SIZE:-1000}"

PROMPT_FILE="${PROMPT_FILE:-${BASE_DIR}/data/fixedprompts_train_seed${PROMPT_SEED}_n${PROMPT_SIZE}.txt}"
REF_FILE="${REF_FILE:-${BASE_DIR}/data/fixedref_val_seed${REF_SEED}_n${REF_SIZE}.txt}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "[prompts] building ${PROMPT_FILE}"
  PROMPT_SEED="${PROMPT_SEED}" PROMPT_SIZE="${PROMPT_SIZE}" PROMPT_FILE="${PROMPT_FILE}" \
    "${PYTHON}" - <<'PY'
import os
import pathlib
from exp11_gpt2_model.data import load_wikitext_subset

seed = int(os.environ["PROMPT_SEED"])
count = int(os.environ["PROMPT_SIZE"])
out_path = pathlib.Path(os.environ["PROMPT_FILE"])

texts = load_wikitext_subset("train", n_samples=count, seed=seed)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(texts), encoding="utf-8")
print(f"[prompts] wrote {len(texts)} -> {out_path}")
PY
fi

if [[ ! -f "${REF_FILE}" ]]; then
  echo "[ref] building ${REF_FILE}"
  REF_SEED="${REF_SEED}" REF_SIZE="${REF_SIZE}" REF_FILE="${REF_FILE}" \
    "${PYTHON}" - <<'PY'
import os
import pathlib
from exp11_gpt2_model.data import load_wikitext_subset

seed = int(os.environ["REF_SEED"])
count = int(os.environ["REF_SIZE"])
out_path = pathlib.Path(os.environ["REF_FILE"])

texts = load_wikitext_subset("validation", n_samples=count, seed=seed)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(texts), encoding="utf-8")
print(f"[ref] wrote {len(texts)} -> {out_path}")
PY
fi

GEN_DIR="${GEN_DIR:-${ROOT}/Total_results/Tables/exp11_gpt2_model/generations_fixedprompts}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT}/Total_results/Tables/exp11_gpt2_model/Results_streaming}"
CKPT_ROOT_DISP="${CKPT_ROOT_DISP:-${ROOT}/Total_results/Tables/exp11_gpt2_model/Results_streaming_ablate}"

METHODS="${METHODS:-no_filter,pointwise,set_aware,dispersion}"
SEEDS="${SEEDS:-1088 2195 4960}"

NUM_CANDIDATES="${NUM_CANDIDATES:-10000}"
GEN_BATCH="${GEN_BATCH:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMP="${TEMP:-0.8}"
TOP_P="${TOP_P:-0.9}"
RNG_SEED="${RNG_SEED:-25}"

for seed in ${SEEDS}; do
  out_dir="${GEN_DIR}/${seed}"
  "${PYTHON}" "${BASE_DIR}/generate_fixedprompt_candidates.py" \
    --seed "${seed}" \
    --gen 4 \
    --methods "${METHODS}" \
    --prompt-file "${PROMPT_FILE}" \
    --out-dir "${out_dir}" \
    --ckpt-root "${CKPT_ROOT}" \
    --ckpt-root-dispersion "${CKPT_ROOT_DISP}" \
    --num-candidates "${NUM_CANDIDATES}" \
    --batch-size "${GEN_BATCH}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMP}" \
    --top-p "${TOP_P}" \
    --device cuda \
    --random-seed "${RNG_SEED}"
done

MAUVE_OUT_DIR="${MAUVE_OUT_DIR:-${BASE_DIR}/MAUVE/fixedprompts}"
mkdir -p "${MAUVE_OUT_DIR}"

"${PYTHON}" "${BASE_DIR}/mauve_eval.py" \
  --generations-dir "${GEN_DIR}" \
  --output-dir "${MAUVE_OUT_DIR}" \
  --seeds "$(echo ${SEEDS} | tr ' ' ',')" \
  --methods "${METHODS}" \
  --min-gen 4 \
  --max-gen 4 \
  --sample-size 1000 \
  --reference-file "${REF_FILE}" \
  --device-id 0 \
  --model-name gpt2-large \
  --batch-size 32 \
  --kmeans-num-redo 1 \
  --kmeans-max-iter 100 \
  --num-buckets auto

if [[ -f "${MAUVE_OUT_DIR}/mauve_g0_g4.csv" ]]; then
  mv "${MAUVE_OUT_DIR}/mauve_g0_g4.csv" "${MAUVE_OUT_DIR}/mauve_g4_fixedprompts.csv"
fi
if [[ -f "${MAUVE_OUT_DIR}/mauve_g0_g4.json" ]]; then
  mv "${MAUVE_OUT_DIR}/mauve_g0_g4.json" "${MAUVE_OUT_DIR}/mauve_g4_fixedprompts.json"
fi

echo "done: ${MAUVE_OUT_DIR}/mauve_g4_fixedprompts.csv"
