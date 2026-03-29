import math
from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from .data import LMDataset, tokenize_texts


def fine_tune(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    texts: Sequence[str],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    warmup_steps: int,
    max_length: int,
    max_steps: int | None = None,
    progress_desc: str | None = None,
    ref_model: GPT2LMHeadModel | None = None,
    kl_coef: float = 0.0,
) -> GPT2LMHeadModel:
    # Ensure recognized loss setting to avoid warnings.
    if hasattr(model, "config"):
        model.config.loss_type = "ForCausalLMLoss"
    model.train()
    enc = tokenize_texts(tokenizer, texts, max_length=max_length)
    ds = LMDataset(enc)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = AdamW(model.parameters(), lr=lr)
    total_steps = epochs * len(loader) if max_steps is None else max_steps
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    step = 0
    pbar = None
    if progress_desc is not None:
        pbar = tqdm(total=total_steps, desc=progress_desc, position=2, leave=False)
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            use_kl = ref_model is not None and float(kl_coef) > 0.0
            micro_batch_size = min(int(batch["input_ids"].size(0)), 16 if use_kl else int(batch["input_ids"].size(0)))
            total_batch = int(batch["input_ids"].size(0))
            opt.zero_grad()
            for start in range(0, total_batch, micro_batch_size):
                end = min(start + micro_batch_size, total_batch)
                micro = {k: v[start:end] for k, v in batch.items()}
                out = model(**micro, labels=micro["input_ids"])
                loss = out.loss
                if use_kl:
                    with torch.no_grad():
                        ref_logits = ref_model(
                            input_ids=micro["input_ids"],
                            attention_mask=micro.get("attention_mask"),
                        ).logits
                    train_logits = out.logits[:, :-1, :].contiguous()
                    ref_logits = ref_logits[:, :-1, :].contiguous()
                    shift_mask = micro["attention_mask"][:, 1:].contiguous().to(train_logits.dtype)
                    token_kl = F.kl_div(
                        F.log_softmax(train_logits, dim=-1),
                        F.softmax(ref_logits, dim=-1),
                        reduction="none",
                    ).sum(dim=-1)
                    loss_kl = (token_kl * shift_mask).sum() / shift_mask.sum().clamp_min(1.0)
                    loss = loss + float(kl_coef) * loss_kl
                # Preserve the effective batch size while keeping the KL path memory-safe.
                scale = float(end - start) / float(total_batch)
                (loss * scale).backward()
            opt.step()
            sched.step()
            step += 1
            if pbar is not None:
                pbar.update(1)
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break
    if pbar is not None:
        pbar.close()
    return model


def compute_perplexities(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> list[float]:
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    ppl_scores: list[float] = []
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size(0), -1)
        seq_loss = (loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp_min(1)
        batch_ppl = torch.exp(seq_loss)
        ppl_scores.extend(batch_ppl.detach().cpu().tolist())
    return ppl_scores


def eval_validation_ppl(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    val_texts: Sequence[str],
    device: torch.device,
    batch_size: int,
    max_eval: int,
    progress_desc: str | None = None,
) -> float:
    if hasattr(model, "config") and not getattr(model.config, "loss_type", None):
        model.config.loss_type = "ForCausalLMLoss"
    model.eval()
    sample = list(val_texts[:max_eval])
    if not sample:
        return float("inf")
    text = "\n\n".join(sample)
    if hasattr(tokenizer, "model_max_length"):
        tokenizer.model_max_length = int(1e9)
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0].to(device)
    seq_len = input_ids.size(0)
    block_size = min(1024, getattr(model.config, "n_positions", 1024))
    stride = 512
    nlls: list[torch.Tensor] = []
    total_tokens = 0
    pbar = None
    if progress_desc is not None:
        total_steps = math.ceil(seq_len / stride)
        pbar = tqdm(total=total_steps, desc=progress_desc, position=2, leave=False)
    for i in range(0, seq_len, stride):
        begin_loc = max(i + stride - block_size, 0)
        end_loc = min(i + stride, seq_len)
        trg_len = end_loc - i
        if trg_len <= 0:
            continue
        input_chunk = input_ids[begin_loc:end_loc].unsqueeze(0)
        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_chunk, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)
        total_tokens += int(trg_len)
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()
    if total_tokens == 0:
        return float("inf")
    return float(math.exp(torch.stack(nlls).sum().item() / total_tokens))
