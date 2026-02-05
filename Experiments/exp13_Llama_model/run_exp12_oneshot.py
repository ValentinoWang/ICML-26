import argparse
import json
import math
import pathlib
import random
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import torch
from datasets import Dataset

from filter.set_aware.model import SetAwareBiasRobustFilter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_texts(path: pathlib.Path, max_samples: int | None = None) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    texts: List[str] = []
    if path.suffix.lower() == ".jsonl":
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
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    texts.append(line.strip())
                if max_samples and len(texts) >= max_samples:
                    break
    if not texts:
        raise ValueError(f"No texts found in {path}")
    return texts


def save_jsonl(texts: Iterable[str], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}, ensure_ascii=True) + "\n")


def build_prompts(seed_texts: List[str], prompt_len: int, num_prompts: int) -> List[str]:
    prompts = [t[:prompt_len] for t in seed_texts if len(t) > 10]
    random.shuffle(prompts)
    return prompts[:num_prompts]


def _require_unlsloth() -> None:
    try:
        import unsloth  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Unsloth is required for this stage. Install with: "
            "pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\""
        ) from exc


def load_model_and_tokenizer(
    model_name: str,
    adapter_dir: pathlib.Path | None,
    max_seq_len: int,
    load_in_4bit: bool,
    adapter_trainable: bool = False,
):
    _require_unlsloth()
    from unsloth import FastLanguageModel

    local_only = pathlib.Path(model_name).expanduser().exists()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=load_in_4bit,
        local_files_only=local_only,
    )
    if adapter_dir is not None:
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=adapter_trainable)
        except Exception:
            # Fallback: try to load adapter directly if supported
            try:
                model.load_adapter(str(adapter_dir))
            except Exception as exc:
                raise RuntimeError(f"Failed to load adapter from {adapter_dir}") from exc
    return model, tokenizer


def setup_lora(model, r: int, lora_alpha: int, lora_dropout: float):
    _require_unlsloth()
    from unsloth import FastLanguageModel

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,
    )
    return model


def train_sft(
    model_name: str,
    adapter_dir: pathlib.Path | None,
    train_jsonl: pathlib.Path,
    output_dir: pathlib.Path,
    max_seq_len: int,
    max_steps: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    seed: int,
) -> None:
    _require_unlsloth()
    from transformers import TrainingArguments
    from trl import SFTTrainer

    set_seed(seed)
    texts = load_texts(train_jsonl)
    dataset = Dataset.from_dict({"text": texts})

    model, tokenizer = load_model_and_tokenizer(
        model_name,
        adapter_dir,
        max_seq_len,
        load_in_4bit=True,
        adapter_trainable=adapter_dir is not None,
    )
    if adapter_dir is None:
        model = setup_lora(model, r=16, lora_alpha=16, lora_dropout=0.0)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        packing=True,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            max_steps=max_steps,
            learning_rate=lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            output_dir=str(output_dir),
            logging_steps=10,
            save_steps=max_steps,
            report_to=[],
        ),
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def generate_candidates(
    model_name: str,
    adapter_dir: pathlib.Path,
    seed_jsonl: pathlib.Path,
    out_jsonl: pathlib.Path,
    num_candidates: int,
    append: bool,
    resume: bool,
    skip: int,
    prompt_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_seq_len: int,
    gen_batch_size: int,
    seed: int,
) -> None:
    _require_unlsloth()
    from unsloth import FastLanguageModel

    set_seed(seed)
    model, tokenizer = load_model_and_tokenizer(model_name, adapter_dir, max_seq_len, load_in_4bit=True)
    model = FastLanguageModel.for_inference(model)
    model.eval()

    seed_texts = load_texts(seed_jsonl)
    prompts = build_prompts(seed_texts, prompt_len=prompt_len, num_prompts=num_candidates)

    if resume and out_jsonl.exists():
        with out_jsonl.open("r", encoding="utf-8") as f:
            existing = sum(1 for _ in f)
        skip = max(skip, existing)
        append = True

    if skip > 0:
        prompts = prompts[skip:]
        if not prompts:
            return

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and out_jsonl.exists() else "w"
    with out_jsonl.open(mode, encoding="utf-8") as f:
        for i in range(0, len(prompts), gen_batch_size):
            batch = prompts[i : i + gen_batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=tokenizer.eos_token_id,
                )
            texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            for text in texts:
                f.write(json.dumps({"text": text}, ensure_ascii=True) + "\n")


def compute_embeddings(
    model,
    tokenizer,
    texts: List[str],
    max_seq_len: int,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    embeddings: List[np.ndarray] = []
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
            out = model(**enc, output_hidden_states=True, return_dict=True)
            hidden = out.hidden_states[-1]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def compute_nlls(
    model,
    tokenizer,
    texts: List[str],
    max_seq_len: int,
    batch_size: int,
) -> np.ndarray:
    model.eval()
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


def train_set_aware_filter(
    seed_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    out_ckpt: pathlib.Path,
    set_size: int,
    steps: int,
    lr: float,
    seed: int,
) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dim = candidate_embeddings.shape[1]
    model = SetAwareBiasRobustFilter(dim=dim, hidden=128, n_heads=4, n_layers=2, dropout=0.0)
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    mu_clean = torch.tensor(seed_embeddings, device=device).mean(dim=0)
    candidates = torch.tensor(candidate_embeddings, device=device)

    for _ in range(steps):
        idx = torch.randint(0, candidates.shape[0], (set_size,), device=device)
        batch = candidates[idx].unsqueeze(0)  # [1, N, D]
        weights, delta_phi = model(batch)
        weighted = model.weighted_estimate(batch, weights)
        pred = weighted + delta_phi
        loss = torch.mean((pred.squeeze(0) - mu_clean) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_ckpt)


def score_candidates(
    candidate_embeddings: np.ndarray,
    filter_ckpt: pathlib.Path,
    set_size: int,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = candidate_embeddings.shape[1]
    model = SetAwareBiasRobustFilter(dim=dim, hidden=128, n_heads=4, n_layers=2, dropout=0.0)
    model.load_state_dict(torch.load(filter_ckpt, map_location=device))
    model.to(device)
    model.eval()

    scores = np.zeros(candidate_embeddings.shape[0], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, candidate_embeddings.shape[0], set_size):
            chunk = candidate_embeddings[start : start + set_size]
            batch = torch.tensor(chunk, device=device).unsqueeze(0)
            weights, _ = model(batch)
            weights = weights.squeeze(0).cpu().numpy()
            scores[start : start + len(chunk)] = weights[: len(chunk)]
    return scores


def _select_pointwise(
    texts: List[str],
    model,
    tokenizer,
    max_seq_len: int,
    batch_size: int,
    select_k: int,
) -> List[str]:
    nlls = compute_nlls(model, tokenizer, texts, max_seq_len=max_seq_len, batch_size=batch_size)
    keep_k = min(select_k, len(texts))
    keep_idx = np.argsort(nlls)[:keep_k]
    return [texts[i] for i in keep_idx]


def _select_dispersion(
    texts: List[str],
    model,
    tokenizer,
    max_seq_len: int,
    batch_size: int,
    select_k: int,
    seed: int,
) -> List[str]:
    embeddings = compute_embeddings(model, tokenizer, texts, max_seq_len=max_seq_len, batch_size=batch_size)
    n_samples = embeddings.shape[0]
    n_clusters = min(select_k, n_samples)
    if n_clusters == 0:
        return []
    from sklearn.cluster import MiniBatchKMeans

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=min(1024, n_samples),
        n_init=1,
        max_iter=50,
    )
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_

    selected = []
    for c in range(n_clusters):
        idxs = np.where(labels == c)[0]
        if idxs.size == 0:
            continue
        cluster_emb = embeddings[idxs]
        dists = np.linalg.norm(cluster_emb - centers[c], axis=1)
        selected.append(int(idxs[np.argmin(dists)]))

    selected = list(dict.fromkeys(selected))
    if len(selected) < select_k:
        remaining = [i for i in range(n_samples) if i not in selected]
        rng = np.random.default_rng(seed)
        rng.shuffle(remaining)
        selected.extend(remaining[: max(0, select_k - len(selected))])

    return [texts[i] for i in selected[:select_k]]


def select_candidates(
    candidates_jsonl: pathlib.Path,
    out_jsonl: pathlib.Path,
    mode: str,
    select_k: int,
    model_name: str | None,
    filter_ckpt: pathlib.Path | None,
    max_seq_len: int,
    batch_size: int,
    ppl_keep_frac: float,
    seed: int,
) -> None:
    set_seed(seed)
    texts = load_texts(candidates_jsonl)

    model = None
    tokenizer = None
    if mode in {"set_aware", "pointwise", "dispersion"} or ppl_keep_frac < 1.0:
        if model_name is None:
            raise ValueError("Selection mode requires --model-name")
        model, tokenizer = load_model_and_tokenizer(model_name, adapter_dir=None, max_seq_len=max_seq_len, load_in_4bit=True)

    nlls = None
    if ppl_keep_frac < 1.0:
        nlls = compute_nlls(model, tokenizer, texts, max_seq_len=max_seq_len, batch_size=batch_size)
        keep_k = max(1, int(len(texts) * ppl_keep_frac))
        keep_idx = np.argsort(nlls)[:keep_k]
        texts = [texts[i] for i in keep_idx]

    if mode == "random":
        random.shuffle(texts)
        save_jsonl(texts[:select_k], out_jsonl)
        return

    if mode == "pointwise":
        selected = _select_pointwise(
            texts,
            model,
            tokenizer,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            select_k=select_k,
        )
        save_jsonl(selected, out_jsonl)
        return

    if mode == "dispersion":
        selected = _select_dispersion(
            texts,
            model,
            tokenizer,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            select_k=select_k,
            seed=seed,
        )
        save_jsonl(selected, out_jsonl)
        return

    if filter_ckpt is None:
        raise ValueError("set_aware mode requires --filter-ckpt")

    embeddings = compute_embeddings(model, tokenizer, texts, max_seq_len=max_seq_len, batch_size=batch_size)
    scores = score_candidates(embeddings, filter_ckpt, set_size=128)
    top_idx = np.argsort(-scores)[:select_k]
    selected = [texts[i] for i in top_idx]
    save_jsonl(selected, out_jsonl)


def evaluate_ppl(
    model_name: str,
    adapter_dir: pathlib.Path,
    val_jsonl: pathlib.Path,
    max_seq_len: int,
    batch_size: int,
) -> float:
    model, tokenizer = load_model_and_tokenizer(model_name, adapter_dir, max_seq_len, load_in_4bit=True)
    model.eval()

    texts = load_texts(val_jsonl)
    total_loss = 0.0
    total_tokens = 0

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
            outputs = model(**enc, labels=enc["input_ids"], return_dict=True)
            loss = outputs.loss
            tokens = enc["input_ids"].numel()
            total_loss += loss.item() * tokens
            total_tokens += tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp12 One-Shot LLM validation.")
    sub = parser.add_subparsers(dest="stage", required=True)

    g0 = sub.add_parser("g0_sft")
    g0.add_argument("--model-name", required=True)
    g0.add_argument("--train-jsonl", required=True, type=pathlib.Path)
    g0.add_argument("--output-dir", required=True, type=pathlib.Path)
    g0.add_argument("--max-seq-len", type=int, default=512)
    g0.add_argument("--max-steps", type=int, default=100)
    g0.add_argument("--batch-size", type=int, default=4)
    g0.add_argument("--grad-accum", type=int, default=4)
    g0.add_argument("--lr", type=float, default=2e-4)
    g0.add_argument("--seed", type=int, default=1088)

    gen = sub.add_parser("generate")
    gen.add_argument("--model-name", required=True)
    gen.add_argument("--adapter-dir", required=True, type=pathlib.Path)
    gen.add_argument("--seed-jsonl", required=True, type=pathlib.Path)
    gen.add_argument("--out-jsonl", required=True, type=pathlib.Path)
    gen.add_argument("--num-candidates", type=int, default=3000)
    gen.add_argument("--append", action="store_true")
    gen.add_argument("--resume", action="store_true")
    gen.add_argument("--skip", type=int, default=0)
    gen.add_argument("--prompt-len", type=int, default=200)
    gen.add_argument("--max-new-tokens", type=int, default=200)
    gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--top-p", type=float, default=0.9)
    gen.add_argument("--repetition-penalty", type=float, default=1.0)
    gen.add_argument("--gen-batch-size", type=int, default=4)
    gen.add_argument("--max-seq-len", type=int, default=512)
    gen.add_argument("--seed", type=int, default=1088)

    filt = sub.add_parser("train_filter")
    filt.add_argument("--model-name", required=True)
    filt.add_argument("--seed-jsonl", required=True, type=pathlib.Path)
    filt.add_argument("--candidates-jsonl", required=True, type=pathlib.Path)
    filt.add_argument("--out-ckpt", required=True, type=pathlib.Path)
    filt.add_argument("--max-seq-len", type=int, default=512)
    filt.add_argument("--batch-size", type=int, default=8)
    filt.add_argument("--set-size", type=int, default=128)
    filt.add_argument("--steps", type=int, default=200)
    filt.add_argument("--lr", type=float, default=1e-3)
    filt.add_argument("--seed", type=int, default=1088)

    sel = sub.add_parser("select")
    sel.add_argument("--mode", choices=["random", "pointwise", "dispersion", "set_aware"], required=True)
    sel.add_argument("--candidates-jsonl", required=True, type=pathlib.Path)
    sel.add_argument("--out-jsonl", required=True, type=pathlib.Path)
    sel.add_argument("--select-k", type=int, default=1000)
    sel.add_argument("--model-name")
    sel.add_argument("--filter-ckpt", type=pathlib.Path)
    sel.add_argument("--max-seq-len", type=int, default=512)
    sel.add_argument("--batch-size", type=int, default=8)
    sel.add_argument("--ppl-keep-frac", type=float, default=1.0)
    sel.add_argument("--seed", type=int, default=1088)

    g1 = sub.add_parser("g1_sft")
    g1.add_argument("--model-name", required=True)
    g1.add_argument("--adapter-dir", required=True, type=pathlib.Path)
    g1.add_argument("--train-jsonl", required=True, type=pathlib.Path)
    g1.add_argument("--output-dir", required=True, type=pathlib.Path)
    g1.add_argument("--max-seq-len", type=int, default=512)
    g1.add_argument("--max-steps", type=int, default=100)
    g1.add_argument("--batch-size", type=int, default=4)
    g1.add_argument("--grad-accum", type=int, default=4)
    g1.add_argument("--lr", type=float, default=2e-4)
    g1.add_argument("--seed", type=int, default=1088)

    ev = sub.add_parser("eval")
    ev.add_argument("--model-name", required=True)
    ev.add_argument("--adapter-dir", required=True, type=pathlib.Path)
    ev.add_argument("--val-jsonl", required=True, type=pathlib.Path)
    ev.add_argument("--max-seq-len", type=int, default=512)
    ev.add_argument("--batch-size", type=int, default=4)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.stage == "g0_sft":
        train_sft(
            model_name=args.model_name,
            adapter_dir=None,
            train_jsonl=args.train_jsonl,
            output_dir=args.output_dir,
            max_seq_len=args.max_seq_len,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            seed=args.seed,
        )
        return

    if args.stage == "generate":
        generate_candidates(
            model_name=args.model_name,
            adapter_dir=args.adapter_dir,
            seed_jsonl=args.seed_jsonl,
            out_jsonl=args.out_jsonl,
            num_candidates=args.num_candidates,
            append=args.append,
            resume=args.resume,
            skip=args.skip,
            prompt_len=args.prompt_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_seq_len=args.max_seq_len,
            gen_batch_size=args.gen_batch_size,
            seed=args.seed,
        )
        return

    if args.stage == "train_filter":
        model, tokenizer = load_model_and_tokenizer(args.model_name, adapter_dir=None, max_seq_len=args.max_seq_len, load_in_4bit=True)
        seed_texts = load_texts(args.seed_jsonl)
        candidate_texts = load_texts(args.candidates_jsonl)

        seed_emb = compute_embeddings(model, tokenizer, seed_texts, max_seq_len=args.max_seq_len, batch_size=args.batch_size)
        cand_emb = compute_embeddings(model, tokenizer, candidate_texts, max_seq_len=args.max_seq_len, batch_size=args.batch_size)
        train_set_aware_filter(
            seed_embeddings=seed_emb,
            candidate_embeddings=cand_emb,
            out_ckpt=args.out_ckpt,
            set_size=args.set_size,
            steps=args.steps,
            lr=args.lr,
            seed=args.seed,
        )
        return

    if args.stage == "select":
        select_candidates(
            candidates_jsonl=args.candidates_jsonl,
            out_jsonl=args.out_jsonl,
            mode=args.mode,
            select_k=args.select_k,
            model_name=args.model_name,
            filter_ckpt=args.filter_ckpt,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            ppl_keep_frac=args.ppl_keep_frac,
            seed=args.seed,
        )
        return

    if args.stage == "g1_sft":
        train_sft(
            model_name=args.model_name,
            adapter_dir=args.adapter_dir,
            train_jsonl=args.train_jsonl,
            output_dir=args.output_dir,
            max_seq_len=args.max_seq_len,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            seed=args.seed,
        )
        return

    if args.stage == "eval":
        ppl = evaluate_ppl(
            model_name=args.model_name,
            adapter_dir=args.adapter_dir,
            val_jsonl=args.val_jsonl,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
        )
        print(f"PPL: {ppl:.4f}")
        return

    raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
