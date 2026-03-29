import argparse
import json
import math
import pathlib
from typing import Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import PeftModel
except Exception as exc:  # pragma: no cover - import guard for runtime environment
    raise RuntimeError(
        "peft is required for loading LoRA adapters. Install with: pip install peft"
    ) from exc


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


def maybe_quant_config(load_in_4bit: bool) -> BitsAndBytesConfig | None:
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_model_and_tokenizer(model_name: str, load_in_4bit: bool) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    quant = maybe_quant_config(load_in_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=quant,
    )
    return model, tokenizer


def streaming_ppl(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: Iterable[str],
    max_length: int,
    stride: int,
) -> float:
    text = "\n\n".join(texts)
    tokenizer.model_max_length = int(1e9)
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0].to(model.device)
    seq_len = input_ids.size(0)
    block_size = min(max_length, getattr(model.config, "max_position_embeddings", max_length))
    stride = min(stride, block_size)

    nlls: List[torch.Tensor] = []
    total_tokens = 0
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

    if total_tokens == 0:
        return float("inf")
    return float(math.exp(torch.stack(nlls).sum().item() / total_tokens))


def parse_adapter(arg: str) -> Tuple[str, pathlib.Path]:
    if "=" not in arg:
        raise ValueError(f"Adapter must be NAME=PATH, got: {arg}")
    name, path = arg.split("=", 1)
    return name.strip(), pathlib.Path(path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming PPL evaluation for Qwen/Llama adapters.")
    parser.add_argument("--model-name", required=True, help="Base model name (HF hub or local path).")
    parser.add_argument("--val-jsonl", required=True, type=pathlib.Path, help="Validation JSONL with text field.")
    parser.add_argument("--adapter", action="append", required=True, help="NAME=PATH for adapter directory.")
    parser.add_argument("--max-length", type=int, default=2048, help="Streaming block size.")
    parser.add_argument("--stride", type=int, default=1024, help="Streaming stride.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of validation samples.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit base model loading.")
    parser.add_argument("--out-csv", type=pathlib.Path, default=None, help="Optional CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts = load_texts(args.val_jsonl, max_samples=args.max_samples)
    model, tokenizer = load_model_and_tokenizer(args.model_name, load_in_4bit=args.load_in_4bit)

    rows = []
    peft_model = None
    for name, adapter_dir in map(parse_adapter, args.adapter):
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_dir}")
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, str(adapter_dir), adapter_name=name)
        else:
            peft_model.load_adapter(str(adapter_dir), adapter_name=name)
        peft_model.set_adapter(name)
        peft_model.eval()
        ppl = streaming_ppl(peft_model, tokenizer, texts, max_length=args.max_length, stride=args.stride)
        rows.append({"name": name, "adapter_dir": str(adapter_dir), "ppl_stream": ppl})
        print(f"{name}: {ppl:.4f}")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", encoding="utf-8") as f:
            f.write("name,adapter_dir,ppl_stream\n")
            for row in rows:
                f.write(f"{row['name']},{row['adapter_dir']},{row['ppl_stream']}\n")
        print(f"Saved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
