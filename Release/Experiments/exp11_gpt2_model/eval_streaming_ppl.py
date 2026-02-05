import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp11_gpt2_model.data import load_wikitext_subset, prepare_tokenizer_model  # noqa: E402
from exp11_gpt2_model.training import eval_validation_ppl  # noqa: E402


def iter_checkpoints(ckpt_root: Path) -> Iterable[Tuple[Path, str, int, int]]:
    for path in ckpt_root.rglob("*_gen*.pt"):
        name = path.stem
        if "_gen" not in name:
            continue
        method, gen_str = name.rsplit("_gen", 1)
        if not gen_str.isdigit():
            continue
        gen = int(gen_str)
        seed = -1
        try:
            rel = path.relative_to(ckpt_root)
            for part in rel.parts:
                if part.isdigit():
                    seed = int(part)
                    break
        except ValueError:
            pass
        yield path, method, gen, seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate streaming (sliding-window) PPL from saved checkpoints.")
    parser.add_argument("--ckpt-root", type=Path, required=True, help="Root directory containing *_gen*.pt checkpoints.")
    parser.add_argument("--val-size", type=int, default=512, help="Validation subset size from Wikitext-103.")
    parser.add_argument("--max-eval", type=int, default=512, help="Max validation samples to use.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "Total_results" / "Tables" / "exp11_gpt2_streaming_ppl.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_root = args.ckpt_root
    if not ckpt_root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {ckpt_root}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    val_texts = load_wikitext_subset("validation", n_samples=args.val_size, seed=1)
    tokenizer, base_model = prepare_tokenizer_model()
    base_model.to(device)

    records = []
    for ckpt_path, method, gen, seed in sorted(iter_checkpoints(ckpt_root)):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = base_model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"Warning: {ckpt_path} missing={len(missing)} unexpected={len(unexpected)}")
        ppl = eval_validation_ppl(
            model=base_model,
            tokenizer=tokenizer,
            val_texts=val_texts,
            device=device,
            batch_size=8,
            max_eval=args.max_eval,
        )
        records.append(
            {
                "seed": seed,
                "method": method,
                "generation": gen,
                "val_ppl_stream": ppl,
                "ckpt_path": str(ckpt_path),
            }
        )
        print(f"{method} G{gen} seed={seed} PPL={ppl:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seed", "method", "generation", "val_ppl_stream", "ckpt_path"],
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
