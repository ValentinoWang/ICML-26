from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from typing import Sequence

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "Total_results" / "Tables" / SCRIPT_DIR.name
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp11_gpt2_model.text_quality import compute_text_quality, read_lines  # noqa: E402


def parse_int_list(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def parse_str_list(s: str) -> list[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def find_seed_dirs(root: pathlib.Path, seeds: Sequence[int] | None) -> list[pathlib.Path]:
    if seeds is None:
        out = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
        return sorted(out, key=lambda p: int(p.name))
    out: list[pathlib.Path] = []
    for s in seeds:
        p = root / str(s)
        if not p.exists():
            raise FileNotFoundError(f"Missing seed directory: {p}")
        out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize text-level quality metrics from saved generation .txt files.")
    ap.add_argument(
        "--generations-root",
        type=str,
        default=str(TABLES_DIR / "generations"),
        help="Root directory that contains <seed>/g<gen>_<method>_{train,candidates}.txt",
    )
    ap.add_argument("--seeds", type=str, default="", help="Comma-separated seeds. Empty => auto-detect.")
    ap.add_argument("--gen", type=int, default=4, help="Generation id to summarize (e.g., 4 for last when --generations 5).")
    ap.add_argument("--methods", type=str, default="no_filter,pointwise,set_aware", help="Comma-separated methods.")
    ap.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "candidates"],
        help="Which saved text file to analyze per method.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional CSV output path. Default writes to Total_results/Tables/exp11_gpt2_model/analysis_text_quality/<split>_g<gen>.csv",
    )
    args = ap.parse_args()

    generations_root = pathlib.Path(args.generations_root)
    seeds = parse_int_list(args.seeds) if args.seeds.strip() else None
    methods = parse_str_list(args.methods)

    seed_dirs = find_seed_dirs(generations_root, seeds)
    rows: list[dict[str, str | int | float]] = []
    for sd in seed_dirs:
        seed = int(sd.name)
        for method in methods:
            path = sd / f"g{args.gen}_{method}_{args.split}.txt"
            if not path.exists():
                continue
            lines = read_lines(str(path))
            raw_total = len(lines)
            raw_nonempty = sum(1 for l in lines if l.strip())
            m = compute_text_quality(lines)
            rows.append(
                {
                    "seed": seed,
                    "gen": int(args.gen),
                    "method": method,
                    "split": args.split,
                    "raw_lines_total": raw_total,
                    "raw_lines_nonempty": raw_nonempty,
                    "n_lines": m.n_lines,
                    "unique_line_ratio": m.unique_line_ratio,
                    "avg_words": m.avg_words,
                    "rep4_intra": m.rep4_intra,
                    "ascii_ratio": m.ascii_ratio,
                    "gzip_ratio": m.gzip_ratio,
                    "path": str(path),
                }
            )

    if not rows:
        raise SystemExit("No matching generation files found; check --generations-root/--gen/--methods/--split.")

    out_path: pathlib.Path
    if args.out.strip():
        out_path = pathlib.Path(args.out)
    else:
        out_path = (
            TABLES_DIR / "analysis_text_quality" / f"{args.split}_g{args.gen}.csv"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    for r in rows:
        print(
            f"seed={r['seed']} gen={r['gen']} {r['method']} {r['split']} | "
            f"uniq={r['unique_line_ratio']:.3f} rep4={r['rep4_intra']:.3f} "
            f"avg_words={r['avg_words']:.1f} gzip={r['gzip_ratio']:.3f} "
            f"lines={r['raw_lines_nonempty']}/{r['raw_lines_total']}"
        )


if __name__ == "__main__":
    main()
