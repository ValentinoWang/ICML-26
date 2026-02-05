import argparse
import hashlib
import json
import math
import pathlib
from typing import List, Tuple

import numpy as np
import torch


def load_texts(path: pathlib.Path, max_samples: int | None) -> List[str]:
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if text:
                texts.append(text)
            if max_samples and len(texts) >= max_samples:
                break
    return texts


def compute_overlap(base_texts: List[str], ours_texts: List[str]) -> Tuple[int, int, float]:
    base_ids = {hashlib.sha1(t.encode("utf-8")).hexdigest() for t in base_texts}
    ours_ids = {hashlib.sha1(t.encode("utf-8")).hexdigest() for t in ours_texts}
    overlap = len(base_ids & ours_ids)
    denom = max(1, min(len(base_ids), len(ours_ids)))
    return overlap, denom, overlap / denom


def load_model_and_tokenizer(model_name: str, adapter_dir: pathlib.Path | None, max_seq_len: int):
    from unsloth import FastLanguageModel

    local_only = pathlib.Path(model_name).expanduser().exists()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
        local_files_only=local_only,
    )
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()
    return model, tokenizer


def compute_nlls(
    model,
    tokenizer,
    texts: List[str],
    max_seq_len: int,
    batch_size: int,
) -> np.ndarray:
    nlls: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            labels = enc["input_ids"]
            outputs = model(**enc, return_dict=True)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_mask = enc["attention_mask"][:, 1:].contiguous().float()
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            token_log_probs = token_log_probs * shift_mask
            token_counts = shift_mask.sum(dim=1).clamp_min(1.0)
            batch_nll = -(token_log_probs.sum(dim=1) / token_counts)
            nlls.append(batch_nll.cpu().numpy())
    return np.concatenate(nlls, axis=0)


def summarize(nlls: np.ndarray) -> dict:
    avg_nll = float(np.mean(nlls))
    avg_ppl = float(math.exp(avg_nll))
    med_ppl = float(np.median(np.exp(nlls)))
    return {"avg_nll": avg_nll, "avg_ppl": avg_ppl, "median_ppl": med_ppl}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality gate for selected data.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-dir", type=pathlib.Path)
    parser.add_argument("--base-jsonl", required=True, type=pathlib.Path)
    parser.add_argument("--ours-jsonl", required=True, type=pathlib.Path)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--tolerance", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.adapter_dir, args.max_seq_len)

    base_texts = load_texts(args.base_jsonl, args.max_samples)
    ours_texts = load_texts(args.ours_jsonl, args.max_samples)

    base_nlls = compute_nlls(model, tokenizer, base_texts, args.max_seq_len, args.batch_size)
    ours_nlls = compute_nlls(model, tokenizer, ours_texts, args.max_seq_len, args.batch_size)

    base_stats = summarize(base_nlls)
    ours_stats = summarize(ours_nlls)
    overlap, overlap_denom, overlap_rate = compute_overlap(base_texts, ours_texts)

    print(
        "BASE  avg_nll={avg_nll:.4f} avg_ppl={avg_ppl:.2f} median_ppl={median_ppl:.2f}".format(**base_stats)
    )
    print(
        "OURS  avg_nll={avg_nll:.4f} avg_ppl={avg_ppl:.2f} median_ppl={median_ppl:.2f}".format(**ours_stats)
    )
    print(f"OVERLAP {overlap}/{overlap_denom} = {overlap_rate:.2%}")

    improvement = 100.0 * (base_stats["avg_ppl"] - ours_stats["avg_ppl"]) / max(base_stats["avg_ppl"], 1e-8)

    if ours_stats["avg_ppl"] > base_stats["avg_ppl"] + args.tolerance:
        raise SystemExit(
            f"[FAIL] Ours Avg PPL {ours_stats['avg_ppl']:.2f} > Base Avg PPL "
            f"{base_stats['avg_ppl']:.2f} (+{args.tolerance:.2f}). Overlap: {overlap_rate:.2%}"
        )

    print(
        f"[PASS] Ours Avg PPL ({ours_stats['avg_ppl']:.2f}) < Base Avg PPL "
        f"({base_stats['avg_ppl']:.2f}). Improvement: {improvement:.2f}%. Overlap: {overlap_rate:.2%}"
    )


if __name__ == "__main__":
    main()
