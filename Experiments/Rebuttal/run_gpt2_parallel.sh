#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${EXPERIMENTS_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [ -z "${HF_HOME:-}" ]; then
  if [ -d /autodl-fs/data/Model/huggingface ]; then
    export HF_HOME=/autodl-fs/data/Model/huggingface
  else
    export HF_HOME=/root/.cache/huggingface
  fi
fi
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${SCRIPT_DIR}/logs"

python - <<'PY'
from pathlib import Path
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, AutoTokenizer, AutoModel
from datasets import load_dataset

hf_home = Path("/autodl-fs/data/Model/huggingface") if Path("/autodl-fs/data/Model/huggingface").exists() else Path("/root/.cache/huggingface")
manual_gpt2 = hf_home / "manual_gpt2"
manual_minilm = hf_home / "manual_all-MiniLM-L6-v2"

gpt2_ready = manual_gpt2.exists() and (manual_gpt2 / "config.json").exists() and (manual_gpt2 / "merges.txt").exists() and (manual_gpt2 / "vocab.json").exists() and ((manual_gpt2 / "model.safetensors").exists() or (manual_gpt2 / "pytorch_model.bin").exists())
minilm_ready = manual_minilm.exists() and (manual_minilm / "config.json").exists() and (((manual_minilm / "model.safetensors").exists()) or ((manual_minilm / "pytorch_model.bin").exists()))

if not gpt2_ready:
    print("prefetch_gpt2_tokenizer", flush=True)
    GPT2TokenizerFast.from_pretrained("gpt2")
    print("prefetch_gpt2_model", flush=True)
    GPT2LMHeadModel.from_pretrained("gpt2")
else:
    print(f"reuse_manual_gpt2={manual_gpt2}", flush=True)

if not minilm_ready:
    print("prefetch_minilm", flush=True)
    AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
else:
    print(f"reuse_manual_minilm={manual_minilm}", flush=True)

print("prefetch_wikitext_train", flush=True)
load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:1%]")
print("prefetch_wikitext_val", flush=True)
load_dataset("wikitext", "wikitext-103-raw-v1", split="validation[:1%]")
print("prefetch_done", flush=True)
PY

python Rebuttal/run_gpt2_rebuttal.py \
  --device "${DEVICE:-cuda}" \
  --seeds 1088,2195,4960 \
  --methods set_aware,clean_mix,batch_mean \
  --clean-ref-size 500 \
  --save-generations

python /root/autodl-tmp/ICML/Experiments/exp11_gpt2_model/mauve_eval.py \
  --generations-dir /root/autodl-tmp/ICML/Experiments/Rebuttal/results/gpt2/generations \
  --seeds 1088,2195,4960 \
  --methods set_aware,clean_mix,batch_mean
