import argparse
import csv
import math
import pathlib
import statistics
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "gpt2_unsup_surrogate"
DEFAULT_OUT_DIR = SCRIPT_DIR / "results" / "summary_unsup_surrogate"
METHOD_ORDER = ["unsup_set_aware", "no_filter", "kl_reg", "pointwise"]
METHOD_LABELS = {
    "unsup_set_aware": "Unsup. SAGE",
    "no_filter": "No Filter",
    "kl_reg": "KL Regularization",
    "pointwise": "Pointwise",
}


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


def method_sort_key(method: str) -> Tuple[int, str]:
    if method in METHOD_ORDER:
        return METHOD_ORDER.index(method), method
    return len(METHOD_ORDER), method


def parse_path_list(raw: str) -> List[pathlib.Path]:
    return [pathlib.Path(part.strip()) for part in str(raw).split(",") if part.strip()]


def read_metric_rows(results_roots: Sequence[pathlib.Path]) -> List[Dict[str, object]]:
    rows_by_key: Dict[Tuple[int, str, int], Dict[str, object]] = {}
    for results_root in results_roots:
        if not results_root.exists():
            continue
        for seed_dir in sorted(path for path in results_root.iterdir() if path.is_dir() and path.name.isdigit()):
            seed = int(seed_dir.name)
            csv_path = seed_dir / "metrics_diversity_ppl.csv"
            if not csv_path.exists():
                continue
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    method = str(row.get("method", "")).strip()
                    if method not in METHOD_LABELS:
                        continue
                    gen = safe_float(row.get("generation"))
                    if gen is None:
                        continue
                    rows_by_key[(seed, method, int(gen))] = {
                        "seed": seed,
                        "method": method,
                        "generation": int(gen),
                        "distinct4": safe_float(row.get("distinct4")),
                        "val_ppl": safe_float(row.get("val_ppl")),
                        "train_rep4_intra": safe_float(row.get("train_rep4_intra")),
                        "train_unique_line_ratio": safe_float(row.get("train_unique_line_ratio")),
                    }
    return [rows_by_key[key] for key in sorted(rows_by_key.keys())]


def read_mauve_rows(results_roots: Sequence[pathlib.Path]) -> Dict[Tuple[str, int, int], Dict[str, object]]:
    out: Dict[Tuple[str, int, int], Dict[str, object]] = {}
    for results_root in results_roots:
        csv_path = results_root / "mauve" / "mauve_g0_g4.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                method = str(row.get("method", "")).strip()
                if method not in METHOD_LABELS:
                    continue
                seed = int(float(row["seed"]))
                generation = int(float(row["generation"]))
                out[(method, seed, generation)] = {
                    "mauve": safe_float(row.get("mauve")),
                }
    return out


def group_rows(rows: Iterable[Dict[str, object]]) -> Dict[Tuple[str, int], Dict[int, Dict[str, object]]]:
    grouped: Dict[Tuple[str, int], Dict[int, Dict[str, object]]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["method"]), int(row["seed"]))][int(row["generation"])] = row
    return grouped


def determine_target_generation(grouped: Dict[Tuple[str, int], Dict[int, Dict[str, object]]], requested_gen: int | None) -> int:
    if requested_gen is not None:
        return int(requested_gen)
    max_gens = [max(gen_map.keys()) for gen_map in grouped.values() if gen_map]
    if not max_gens:
        return 0
    return min(max_gens)


def build_final_summary(
    rows: List[Dict[str, object]],
    mauve_rows: Dict[Tuple[str, int, int], Dict[str, object]],
    gen: int | None,
) -> Tuple[List[Dict[str, object]], int]:
    grouped = group_rows(rows)
    target_gen = determine_target_generation(grouped, gen)
    per_method: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for (method, seed), gen_map in grouped.items():
        if 0 not in gen_map or target_gen not in gen_map:
            continue
        g0 = gen_map[0]
        gt = dict(gen_map[target_gen])
        gt["seed"] = seed
        gt["distinct4_delta_from_g0"] = (
            None if gt["distinct4"] is None or g0["distinct4"] is None else float(gt["distinct4"]) - float(g0["distinct4"])
        )
        gt["val_ppl_delta_from_g0"] = (
            None if gt["val_ppl"] is None or g0["val_ppl"] is None else float(gt["val_ppl"]) - float(g0["val_ppl"])
        )
        gt.update(mauve_rows.get((method, seed, target_gen), {}))
        per_method[method].append(gt)

    summary_rows: List[Dict[str, object]] = []
    for method in sorted(per_method.keys(), key=method_sort_key):
        seed_rows = per_method[method]
        seeds = sorted(int(r["seed"]) for r in seed_rows)
        out: Dict[str, object] = {
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "generation": target_gen,
            "n_seeds": len(seed_rows),
            "seeds": ",".join(str(seed) for seed in seeds),
        }
        for metric in ["distinct4", "val_ppl", "distinct4_delta_from_g0", "val_ppl_delta_from_g0", "mauve", "train_rep4_intra"]:
            vals = [row.get(metric) for row in seed_rows if row.get(metric) is not None]
            mean, std = summarize(vals)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_pm"] = format_pm(mean, std)
        summary_rows.append(out)
    return summary_rows, target_gen


def build_trajectory_summary(
    rows: List[Dict[str, object]],
    mauve_rows: Dict[Tuple[str, int, int], Dict[str, object]],
) -> List[Dict[str, object]]:
    by_method_gen: Dict[Tuple[str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        method = str(row["method"])
        seed = int(row["seed"])
        generation = int(row["generation"])
        merged = dict(row)
        merged.update(mauve_rows.get((method, seed, generation), {}))
        by_method_gen[(method, generation)].append(merged)

    summary_rows: List[Dict[str, object]] = []
    for method, generation in sorted(by_method_gen.keys(), key=lambda x: (method_sort_key(x[0]), x[1])):
        seed_rows = by_method_gen[(method, generation)]
        out: Dict[str, object] = {
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "generation": generation,
            "n_seeds": len(seed_rows),
        }
        for metric in ["distinct4", "val_ppl", "mauve", "train_rep4_intra"]:
            vals = [row.get(metric) for row in seed_rows if row.get(metric) is not None]
            mean, std = summarize(vals)
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
    lines: List[str] = ["# GPT-2 Unsupervised Surrogate", ""]
    headers = ["Method", "Gen", "Distinct-4", "Held-out PPL", "ΔDistinct-4 vs G0", "ΔPPL vs G0", "MAUVE", "Rep-4 Intra"]
    rows = [
        [
            str(row["method_label"]),
            str(row["generation"]),
            str(row["distinct4_pm"]),
            str(row["val_ppl_pm"]),
            str(row["distinct4_delta_from_g0_pm"]),
            str(row["val_ppl_delta_from_g0_pm"]),
            str(row["mauve_pm"]),
            str(row["train_rep4_intra_pm"]),
        ]
        for row in final_rows
    ]
    if rows:
        lines.append(markdown_table(headers, rows))
    return "\n".join(lines).rstrip() + "\n"


def print_console_summary(final_rows: List[Dict[str, object]]) -> None:
    if not final_rows:
        print("gpt2-unsup: no complete rows found")
        return
    print("\n== GPT2 UNSUP SURROGATE ==")
    for row in final_rows:
        print(
            f"- {row['method_label']}: gen={row['generation']} "
            f"distinct4={row['distinct4_pm']} ppl={row['val_ppl_pm']} "
            f"mauve={row['mauve_pm']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GPT-2 unsupervised surrogate rebuttal outputs.")
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(DEFAULT_RESULTS_ROOT),
        help="Results root, or a comma-separated list of roots to merge.",
    )
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gen", type=int, default=None)
    args = parser.parse_args()

    results_roots = parse_path_list(args.results_root)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = read_metric_rows(results_roots)
    mauve_rows = read_mauve_rows(results_roots)
    final_rows, target_gen = build_final_summary(metric_rows, mauve_rows, gen=args.gen)
    traj_rows = build_trajectory_summary(metric_rows, mauve_rows)

    write_csv(out_dir / "gpt2_unsup_surrogate_final.csv", final_rows)
    write_csv(out_dir / "gpt2_unsup_surrogate_trajectory.csv", traj_rows)
    md = build_markdown(final_rows)
    md_path = out_dir / "gpt2_unsup_surrogate_table.md"
    md_path.write_text(md, encoding="utf-8")

    print_console_summary(final_rows)
    print(f"\nWrote final table to {out_dir / 'gpt2_unsup_surrogate_final.csv'} (gen={target_gen})")
    print(f"Wrote trajectory table to {out_dir / 'gpt2_unsup_surrogate_trajectory.csv'}")
    print(f"Wrote Markdown table to {md_path}")


if __name__ == "__main__":
    main()
