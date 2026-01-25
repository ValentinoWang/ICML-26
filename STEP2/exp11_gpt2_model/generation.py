import random
from typing import List, Sequence

import torch
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast
from tqdm.auto import tqdm


def generate_texts(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    prompt_pool: Sequence[str],
    n_samples: int,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    temperature: float,
    top_p: float,
    progress: tqdm | None = None,
) -> List[str]:
    # Ensure left padding to silence decoder-only padding warnings.
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    out_texts: List[str] = []
    with torch.no_grad():
        while len(out_texts) < n_samples:
            k = min(len(prompt_pool), batch_size, n_samples - len(out_texts))
            prompts = random.sample(prompt_pool, k=k)
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            ).to(device)
            gen = model.generate(
                **enc,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
            out_texts.extend(texts)
            if progress is not None:
                progress.update(len(texts))
    return out_texts[:n_samples]


def embed_texts(
    encoder: AutoModel,
    encoder_tokenizer: AutoTokenizer,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int,
    max_length: int = 128,
) -> torch.Tensor:
    encoder.eval()
    all_embs: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            enc = encoder_tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            out = encoder(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            all_embs.append(pooled.cpu())
    return torch.cat(all_embs, dim=0)
