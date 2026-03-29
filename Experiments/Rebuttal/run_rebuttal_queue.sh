#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENTS_DIR}/.." && pwd)"
cd "${EXPERIMENTS_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
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

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"

CIFAR_ROOT="${REPO_ROOT}/data"
CIFAR_TAR="${CIFAR_ROOT}/cifar-10-python.tar.gz"
CIFAR_URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DEVICE="${DEVICE:-cuda}"

timestamp() {
  date '+%F %T'
}

cifar_ready() {
  python - <<'PY' >/dev/null 2>&1
from torchvision.datasets import CIFAR10
CIFAR10(root='/root/autodl-tmp/ICML/data', train=True, download=False)
CIFAR10(root='/root/autodl-tmp/ICML/data', train=False, download=False)
PY
}

extract_cifar() {
  [ -f "${CIFAR_TAR}" ] || return 1
  tar -xzf "${CIFAR_TAR}" -C "${CIFAR_ROOT}" >/dev/null 2>&1
}

download_cifar_fresh() {
  mkdir -p "${CIFAR_ROOT}"
  local bad_path=""
  if [ -f "${CIFAR_TAR}" ]; then
    bad_path="${CIFAR_TAR}.bad.$(date +%s)"
    mv "${CIFAR_TAR}" "${bad_path}" || true
  fi
  echo "[$(timestamp)] Downloading CIFAR-10 archive..." | tee -a "${LOG_DIR}/queue.log"
  aria2c -x8 -s8 --allow-overwrite=true -o "$(basename "${CIFAR_TAR}")" -d "${CIFAR_ROOT}" "${CIFAR_URL}"
}

prepare_cifar() {
  echo "[$(timestamp)] Checking CIFAR-10 availability..." | tee -a "${LOG_DIR}/queue.log"
  if cifar_ready; then
    echo "[$(timestamp)] CIFAR-10 already ready." | tee -a "${LOG_DIR}/queue.log"
    return 0
  fi
  while true; do
    if extract_cifar && cifar_ready; then
      echo "[$(timestamp)] CIFAR-10 extracted and verified." | tee -a "${LOG_DIR}/queue.log"
      return 0
    fi
    download_cifar_fresh
    if extract_cifar && cifar_ready; then
      echo "[$(timestamp)] CIFAR-10 downloaded, extracted, and verified." | tee -a "${LOG_DIR}/queue.log"
      return 0
    fi
    echo "[$(timestamp)] CIFAR-10 verification failed after download; retrying..." | tee -a "${LOG_DIR}/queue.log"
    sleep 5
  done
}

prefetch_gpt2_assets() {
  echo "[$(timestamp)] Prefetching GPT-2 / MiniLM / Wikitext assets..." | tee -a "${LOG_DIR}/queue.log"
  python - <<'PY'
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, AutoTokenizer, AutoModel
from datasets import load_dataset

hub = '/autodl-fs/data/Model/huggingface/hub'
ds_cache = '/autodl-fs/data/Model/huggingface/datasets'

GPT2TokenizerFast.from_pretrained('gpt2', cache_dir=hub)
GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=hub)
AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir=hub)
AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir=hub)
load_dataset('wikitext', 'wikitext-103-raw-v1', split='train[:1%]', cache_dir=ds_cache)
load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation[:1%]', cache_dir=ds_cache)
print('gpt2_prefetch_done')
PY
}

run_cifar() {
  local seeds=("$@")
  echo "[$(timestamp)] Running CIFAR rebuttal for seeds: ${seeds[*]}" | tee -a "${LOG_DIR}/queue.log"
  python Rebuttal/run_cifar_rebuttal.py \
    --device "${DEVICE}" \
    --seeds "${seeds[@]}" \
    --modes set_aware clean_ft clean_mix batch_mean \
    --clean-set-size 100
}

run_gpt2() {
  local seeds_csv="$1"
  echo "[$(timestamp)] Running GPT-2 rebuttal for seeds: ${seeds_csv}" | tee -a "${LOG_DIR}/queue.log"
  python Rebuttal/run_gpt2_rebuttal.py \
    --device "${DEVICE}" \
    --seeds "${seeds_csv}" \
    --methods set_aware,clean_mix,batch_mean \
    --clean-ref-size 500
}

main() {
  echo "[$(timestamp)] Rebuttal queue started." | tee -a "${LOG_DIR}/queue.log"
  prepare_cifar
  run_cifar 1088 2>&1 | tee "${LOG_DIR}/cifar_seed1088.log"
  run_cifar 2195 4960 2>&1 | tee "${LOG_DIR}/cifar_seed2195_4960.log"
  prefetch_gpt2_assets 2>&1 | tee "${LOG_DIR}/gpt2_prefetch.log"
  run_gpt2 "1088,2195,4960" 2>&1 | tee "${LOG_DIR}/gpt2_all_seeds.log"
  echo "[$(timestamp)] Rebuttal queue finished." | tee -a "${LOG_DIR}/queue.log"
}

main "$@"
