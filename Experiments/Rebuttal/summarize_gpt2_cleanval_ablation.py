import argparse
import csv
import math
import pathlib
import statistics
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "gpt2_cleanval_size"
DEFAULT_OUT_DIR = SCRIPT_DIR / "results" / "summary_cleanval_size"


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out):
        return None
    return out


def summarize(values: Sequence[float]) -> Tuple[float | None, float | None]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.fmean(vals), statistics.stdev(vals)


def format_pm(mean: float | None, std: float | None, digits: int = 4) -> str:
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def parse_clean_val_size(path: pathlib.Path) -> int:
    name = path.name
    if not name.startswith("n"):
        raise ValueError(f"Unexpected clean-val run directory: {path}")
    return int(name[1:])


def iter_run_dirs(results_root: pathlib.Path) -> List[pathlib.Path]:
    return sorted(
        [path for path in results_root.iterdir() if path.is_dir() and path.name.startswith("n")],
        key=parse_clean_val_size,
    )


def read_metric_rows(results_root: pathlib.Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run_dir in iter_run_dirs(results_root):
        clean_val_size = parse_clean_val_size(run_dir)
        for seed_dir in sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.isdigit()):
            seed = int(seed_dir.name)
            csv_path = seed_dir / "metrics_diversity_ppl.csv"
            if not csv_path.exists():
                continue
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if str(row.get("method", "")).strip() != "set_aware":
                        continue
                    gen = safe_float(row.get("generation"))
                    if gen is None:
                        continue
                    rows.append(
                        {
                            "clean_val_size": clean_val_size,
                            "seed": seed,
                            "generation": int(gen),
                            "distinct4": safe_float(row.get("distinct4")),
                            "val_ppl": safe_float(row.get("val_ppl")),
                            "train_rep4_intra": safe_float(row.get("train_rep4_intra")),
                        }
                    )
    return rows


def read_mauve_rows(results_root: pathlib.Path) -> Dict[Tuple[int, int, int], Dict[str, object]]:
    rows: Dict[Tuple[int, int, int], Dict[str, object]] = {}
    for run_dir in iter_run_dirs(results_root):
        clean_val_size = parse_clean_val_size(run_dir)
        csv_path = run_dir / "mauve" / "mauve_g0_g4.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("method", "")).strip() != "set_aware":
                    continue
                seed = int(float(row["seed"]))
                generation = int(float(row["generation"]))
                rows[(clean_val_size, seed, generation)] = {
                    "mauve": safe_float(row.get("mauve")),
                    "mauve_star": safe_float(row.get("mauve_star")),
                }
    return rows


def determine_target_generation(rows: Iterable[Dict[str, object]], requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    by_run_seed: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for row in rows:
        by_run_seed[(int(row["clean_val_size"]), int(row["seed"]))].append(int(row["generation"]))
    max_gens = [max(gens) for gens in by_run_seed.values() if gens]
    return min(max_gens) if max_gens else 0


def build_final_summary(
    metric_rows: List[Dict[str, object]],
    mauve_rows: Dict[Tuple[int, int, int], Dict[str, object]],
    generation: int | None,
) -> Tuple[List[Dict[str, object]], int]:
    target_gen = determine_target_generation(metric_rows, generation)
    by_size: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        if int(row["generation"]) != target_gen:
            continue
        clean_val_size = int(row["clean_val_size"])
        seed = int(row["seed"])
        merged = dict(row)
        merged.update(mauve_rows.get((clean_val_size, seed, target_gen), {}))
        by_size[clean_val_size].append(merged)

    summary_rows: List[Dict[str, object]] = []
    for clean_val_size in sorted(by_size):
        seed_rows = by_size[clean_val_size]
        seeds = sorted(int(row["seed"]) for row in seed_rows)
        out: Dict[str, object] = {
            "clean_val_size": clean_val_size,
            "generation": target_gen,
            "n_seeds": len(seed_rows),
            "seeds": ",".join(str(seed) for seed in seeds),
        }
        for metric in ["distinct4", "val_ppl", "mauve", "train_rep4_intra"]:
            values = [row.get(metric) for row in seed_rows if row.get(metric) is not None]
            mean, std = summarize(values)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_pm"] = format_pm(mean, std)
        summary_rows.append(out)
    return summary_rows, target_gen


def build_trajectory_summary(
    metric_rows: List[Dict[str, object]],
    mauve_rows: Dict[Tuple[int, int, int], Dict[str, object]],
) -> List[Dict[str, object]]:
    by_size_gen: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        clean_val_size = int(row["clean_val_size"])
        seed = int(row["seed"])
        generation = int(row["generation"])
        merged = dict(row)
        merged.update(mauve_rows.get((clean_val_size, seed, generation), {}))
        by_size_gen[(clean_val_size, generation)].append(merged)

    summary_rows: List[Dict[str, object]] = []
    for clean_val_size, generation in sorted(by_size_gen):
        seed_rows = by_size_gen[(clean_val_size, generation)]
        out: Dict[str, object] = {
            "clean_val_size": clean_val_size,
            "generation": generation,
            "n_seeds": len(seed_rows),
        }
        for metric in ["distinct4", "val_ppl", "mauve", "train_rep4_intra"]:
            values = [row.get(metric) for row in seed_rows if row.get(metric) is not None]
            mean, std = summarize(values)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_pm"] = format_pm(mean, std)
        summary_rows.append(out)
    return summary_rows


def write_csv(path: pathlib.Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_markdown(final_rows: List[Dict[str, object]]) -> str:
    lines = ["# GPT-2 Clean-Val Size Ablation", ""]
    headers = ["N_val", "Gen", "Distinct-4", "Val PPL", "MAUVE", "Rep-4 Intra", "Seeds"]
    rows = [
        [
            str(row["clean_val_size"]),
            str(row["generation"]),
            str(row["distinct4_pm"]),
            str(row["val_ppl_pm"]),
            str(row["mauve_pm"]),
            str(row["train_rep4_intra_pm"]),
            str(row["seeds"]),
        ]
        for row in final_rows
    ]
    if rows:
        lines.append(markdown_table(headers, rows))
    return "\n".join(lines).rstrip() + "\n"


def print_console_summary(final_rows: List[Dict[str, object]]) -> None:
    if not final_rows:
        print("gpt2-cleanval: no complete rows found")
        return
    print("\n== GPT2 CLEAN-VAL SIZE ==")
    for row in final_rows:
        print(
            f"- N={row['clean_val_size']}: gen={row['generation']} "
            f"distinct4={row['distinct4_pm']} val_ppl={row['val_ppl_pm']} "
            f"mauve={row['mauve_pm']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GPT-2 clean validation size ablations.")
    parser.add_argument("--results-root", type=pathlib.Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gen", type=int, default=4)
    args = parser.parse_args()

    results_root = pathlib.Path(args.results_root)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = read_metric_rows(results_root) if results_root.exists() else []
    mauve_rows = read_mauve_rows(results_root) if results_root.exists() else {}
    final_rows, target_gen = build_final_summary(metric_rows, mauve_rows, generation=args.gen)
    traj_rows = build_trajectory_summary(metric_rows, mauve_rows)

    write_csv(out_dir / "gpt2_cleanval_size_final.csv", final_rows)
    write_csv(out_dir / "gpt2_cleanval_size_trajectory.csv", traj_rows)
    md = build_markdown(final_rows)
    md_path = out_dir / "gpt2_cleanval_size_table.md"
    md_path.write_text(md, encoding="utf-8")

    print_console_summary(final_rows)
    print(f"\nWrote final table to {out_dir / 'gpt2_cleanval_size_final.csv'} (gen={target_gen})")
    print(f"Wrote trajectory table to {out_dir / 'gpt2_cleanval_size_trajectory.csv'}")
    print(f"Wrote Markdown table to {md_path}")


if __name__ == "__main__":
    main()
