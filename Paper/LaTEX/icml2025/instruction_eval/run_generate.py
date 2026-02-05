import argparse
import json
import pathlib
import random
from typing import List, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_prompts(path: pathlib.Path, limit: int | None) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "prompt" not in obj:
                continue
            items.append(obj)
            if limit and len(items) >= limit:
                break
    if not items:
        raise ValueError(f"No prompts found in {path}")
    return items


def format_prompt(prompt: str, template: str) -> str:
    if template == "instruct":
        return f"Instruction:\n{prompt}\n\nResponse:\n"
    if template == "plain":
        return prompt
    raise ValueError(f"Unknown template: {template}")


def trim_response(text: str) -> str:
    markers = [
        "\n\nInstruction:",
        "\nInstruction:",
        "\n\n### Instruction",
        "\n### Instruction",
    ]
    for marker in markers:
        idx = text.find(marker)
        if idx != -1:
            return text[:idx].strip()
    return text.strip()


def load_model_and_tokenizer(
    model_name: str,
    adapter_dir: pathlib.Path | None,
    max_seq_len: int,
    load_in_4bit: bool,
) -> Tuple[object, object, bool]:
    local_only = pathlib.Path(model_name).expanduser().exists()
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_len,
            dtype=None,
            load_in_4bit=load_in_4bit,
            local_files_only=local_only,
        )
        if adapter_dir is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        model = FastLanguageModel.for_inference(model)
        return model, tokenizer, True
    except Exception:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=local_only,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if adapter_dir is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        return model, tokenizer, False


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> List[str]:
    device = next(model.parameters()).device
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    with torch.no_grad():
        gen_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=pad_id,
        )
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate instruction responses.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-dir", type=pathlib.Path)
    parser.add_argument("--prompts-jsonl", required=True, type=pathlib.Path)
    parser.add_argument("--out-jsonl", required=True, type=pathlib.Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--template", choices=["plain", "instruct"], default="instruct")
    parser.add_argument("--seed", type=int, default=1088)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    items = load_prompts(args.prompts_jsonl, args.limit)
    prompts = [format_prompt(item["prompt"], args.template) for item in items]

    print(f"Loaded {len(prompts)} prompts from {args.prompts_jsonl}")
    model, tokenizer, _ = load_model_and_tokenizer(
        args.model_name,
        args.adapter_dir,
        max_seq_len=args.max_seq_len,
        load_in_4bit=args.load_in_4bit,
    )
    model.eval()
    print("Model loaded, starting generation...")

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        total = len(prompts)
        for start in range(0, total, args.batch_size):
            batch_prompts = prompts[start : start + args.batch_size]
            outputs = generate_batch(
                model,
                tokenizer,
                batch_prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            for i, full_text in enumerate(outputs):
                item = items[start + i]
                prompt_text = batch_prompts[i]
                response = full_text
                if full_text.startswith(prompt_text):
                    response = full_text[len(prompt_text) :].strip()
                response = trim_response(response)
                record = {
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "response": response,
                }
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
            done = min(start + len(batch_prompts), total)
            print(f"Generated {done}/{total} prompts")


if __name__ == "__main__":
    main()
