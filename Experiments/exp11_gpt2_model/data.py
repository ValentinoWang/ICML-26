import os
import pathlib
from typing import Dict, List, Sequence, Tuple


def _normalize_omp_num_threads() -> None:
    value = os.environ.get("OMP_NUM_THREADS")
    if value is None:
        return
    try:
        if int(value) >= 1:
            return
    except ValueError:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"


_normalize_omp_num_threads()

import torch
from datasets import Dataset as HFDataset
from datasets import DownloadConfig, concatenate_datasets, load_dataset
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def _resolve_default_hf_home() -> pathlib.Path:
    env_home = os.environ.get("HF_HOME")
    if env_home:
        return pathlib.Path(env_home)
    preferred = pathlib.Path("./.hf_cache")
    if preferred.exists():
        return preferred
    return pathlib.Path("./.hf_cache")


DEFAULT_HF_HOME = _resolve_default_hf_home()
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_HF_HOME / "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_HF_HOME / "datasets"))


def _manual_model_dir(name: str) -> pathlib.Path | None:
    candidates = [
        DEFAULT_HF_HOME / f"manual_{name}",
        pathlib.Path("./.hf_cache") / f"manual_{name}",
        pathlib.Path("./.hf_cache") / f"manual_{name}",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _is_complete_gpt2_dir(path: pathlib.Path) -> bool:
    required = ["config.json", "merges.txt", "vocab.json"]
    if not all((path / name).exists() for name in required):
        return False
    safetensors = path / "model.safetensors"
    pytorch_bin = path / "pytorch_model.bin"
    if safetensors.exists() and safetensors.stat().st_size > 100_000_000:
        return True
    if pytorch_bin.exists() and pytorch_bin.stat().st_size > 100_000_000:
        return True
    return False


def _hf_hub_cache() -> str | None:
    cache = os.environ.get("HF_HUB_CACHE")
    if cache:
        return cache
    return str(DEFAULT_HF_HOME / "hub")


def _hf_datasets_cache() -> str | None:
    cache = os.environ.get("HF_DATASETS_CACHE")
    if cache:
        return cache
    return str(DEFAULT_HF_HOME / "datasets")


def _wikitext_cache_root() -> pathlib.Path:
    return pathlib.Path(_hf_datasets_cache()) / "wikitext" / "wikitext-103-raw-v1" / "0.0.0"


def _has_local_wikitext_cache() -> bool:
    cache_root = _wikitext_cache_root()
    if not cache_root.exists():
        return False
    for candidate in cache_root.iterdir():
        if not candidate.is_dir():
            continue
        required = [
            candidate / "dataset_info.json",
            candidate / "wikitext-validation.arrow",
            candidate / "wikitext-test.arrow",
            candidate / "wikitext-train-00000-of-00002.arrow",
            candidate / "wikitext-train-00001-of-00002.arrow",
        ]
        if all(path.exists() for path in required):
            return True
    return False


def _find_local_wikitext_cache_dir() -> pathlib.Path | None:
    cache_root = _wikitext_cache_root()
    if not cache_root.exists():
        return None
    for candidate in cache_root.iterdir():
        if not candidate.is_dir():
            continue
        required = [
            candidate / "dataset_info.json",
            candidate / "wikitext-validation.arrow",
            candidate / "wikitext-test.arrow",
            candidate / "wikitext-train-00000-of-00002.arrow",
            candidate / "wikitext-train-00001-of-00002.arrow",
        ]
        if all(path.exists() for path in required):
            return candidate
    return None


def _load_local_wikitext_split(split: str) -> HFDataset:
    cache_dir = _find_local_wikitext_cache_dir()
    if cache_dir is None:
        raise FileNotFoundError("Local WikiText cache is not complete.")
    if split == "train":
        parts = [
            HFDataset.from_file(str(cache_dir / "wikitext-train-00000-of-00002.arrow")),
            HFDataset.from_file(str(cache_dir / "wikitext-train-00001-of-00002.arrow")),
        ]
        return concatenate_datasets(parts)
    if split == "validation":
        return HFDataset.from_file(str(cache_dir / "wikitext-validation.arrow"))
    if split == "test":
        return HFDataset.from_file(str(cache_dir / "wikitext-test.arrow"))
    raise ValueError(f"Unsupported split={split!r}")

class LMDataset(Dataset):
    """Simple LM dataset wrapper for tokenized sequences."""

    def __init__(self, encodings: Dict[str, torch.Tensor]) -> None:
        self.encodings = encodings

    def __len__(self) -> int:
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.encodings.items()}


def prepare_tokenizer_model() -> Tuple[GPT2TokenizerFast, GPT2LMHeadModel]:
    cache_dir = _hf_hub_cache()
    manual_gpt2 = _manual_model_dir("gpt2")
    model_source = str(manual_gpt2) if manual_gpt2 is not None and _is_complete_gpt2_dir(manual_gpt2) else "gpt2"
    local_only = model_source != "gpt2"
    tokenizer = GPT2TokenizerFast.from_pretrained(
        model_source,
        local_files_only=local_only,
        cache_dir=cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # avoid decoder-only right-padding warnings
    model = GPT2LMHeadModel.from_pretrained(
        model_source,
        local_files_only=local_only,
        cache_dir=cache_dir,
    )
    model.resize_token_embeddings(len(tokenizer))
    # Silence loss_type=None warning by setting a recognized loss name.
    model.config.loss_type = "ForCausalLMLoss"
    return tokenizer, model


def load_wikitext_subset(split: str, n_samples: int, seed: int) -> List[str]:
    cache_dir = _hf_datasets_cache()
    if _has_local_wikitext_cache():
        ds = _load_local_wikitext_split(split)
    else:
        ds = load_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            split=split,
            cache_dir=cache_dir,
            download_config=DownloadConfig(local_files_only=False),
        )
    ds = ds.shuffle(seed=seed)
    texts: List[str] = []
    for item in ds:
        text = item["text"].strip()
        if not text:
            continue
        texts.append(text)
        if len(texts) >= n_samples:
            break
    return texts


def tokenize_texts(tokenizer: GPT2TokenizerFast, texts: Sequence[str], max_length: int) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}
