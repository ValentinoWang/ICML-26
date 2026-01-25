from typing import Dict, List, Sequence, Tuple

import os
import pathlib

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DEFAULT_HF_HOME = pathlib.Path(os.environ.get("HF_HOME", "/root/autodl-tmp/Model/hf_cache"))
if DEFAULT_HF_HOME.exists():
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_HOME / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_HF_HOME / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_HF_HOME / "datasets"))


def _hf_hub_cache() -> str | None:
    cache = os.environ.get("HF_HUB_CACHE")
    if cache:
        return cache
    if DEFAULT_HF_HOME.exists():
        return str(DEFAULT_HF_HOME / "hub")
    return None


def _hf_datasets_cache() -> str | None:
    cache = os.environ.get("HF_DATASETS_CACHE")
    if cache:
        return cache
    if DEFAULT_HF_HOME.exists():
        return str(DEFAULT_HF_HOME / "datasets")
    return None

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
    tokenizer = GPT2TokenizerFast.from_pretrained(
        "gpt2",
        local_files_only=True,
        cache_dir=cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # avoid decoder-only right-padding warnings
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        local_files_only=True,
        cache_dir=cache_dir,
    )
    model.resize_token_embeddings(len(tokenizer))
    # Silence loss_type=None warning by setting a recognized loss name.
    model.config.loss_type = "ForCausalLMLoss"
    return tokenizer, model


def load_wikitext_subset(split: str, n_samples: int, seed: int) -> List[str]:
    cache_dir = _hf_datasets_cache()
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, cache_dir=cache_dir)
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
