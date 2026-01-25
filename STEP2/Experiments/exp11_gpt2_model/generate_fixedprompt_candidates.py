import argparse
import pathlib
import random
import sys
from typing import List

import torch

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp11_gpt2_model.data import prepare_tokenizer_model  # noqa: E402
from exp11_gpt2_model.generation import generate_texts  # noqa: E402


def load_prompts(path: pathlib.Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    prompts: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = line.strip()
            if text:
                prompts.append(text)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def load_checkpoint(model, ckpt_path: pathlib.Path) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] {ckpt_path} missing={len(missing)} unexpected={len(unexpected)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GPT-2 candidates with fixed prompts (G4).")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--methods", type=str, default="no_filter,pointwise,set_aware,dispersion")
    parser.add_argument("--prompt-file", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--ckpt-root", type=pathlib.Path, required=True)
    parser.add_argument("--ckpt-root-dispersion", type=pathlib.Path, default=None)
    parser.add_argument("--num-candidates", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--random-seed", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    prompts = load_prompts(args.prompt_file)

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    tokenizer, model = prepare_tokenizer_model()
    model.to(device)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for method in methods:
        out_path = args.out_dir / f"g{args.gen}_{method}_candidates.txt"
        if out_path.exists():
            print(f"[skip] {out_path}")
            continue
        ckpt_root = args.ckpt_root
        if method == "dispersion" and args.ckpt_root_dispersion is not None:
            ckpt_root = args.ckpt_root_dispersion
        ckpt_path = ckpt_root / str(args.seed) / f"{method}_gen{args.gen}.pt"
        load_checkpoint(model, ckpt_path)
        texts = generate_texts(
            model=model,
            tokenizer=tokenizer,
            prompt_pool=prompts,
            n_samples=args.num_candidates,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
            temperature=args.temperature,
            top_p=args.top_p,
            progress=None,
        )
        out_path.write_text("\n".join(texts), encoding="utf-8")
        print(f"[gen] seed={args.seed} {method} -> {out_path}")


if __name__ == "__main__":
    main()
