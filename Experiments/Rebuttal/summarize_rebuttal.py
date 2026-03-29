import argparse
import csv
import math
import pathlib
import statistics
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_CIFAR_DIR = SCRIPT_DIR / "results" / "cifar10"
DEFAULT_GPT2_DIR = SCRIPT_DIR / "results" / "gpt2"
DEFAULT_OUT_DIR = SCRIPT_DIR / "results" / "summary"

CIFAR_METHOD_ORDER = ["set_aware", "clean_only", "clean_ft", "clean_mix", "batch_mean", "baseline"]
GPT2_METHOD_ORDER = ["set_aware", "unsup_set_aware", "kl_reg", "clean_only", "clean_ft", "clean_mix", "batch_mean", "pointwise", "no_filter"]
METHOD_LABELS = {
    "set_aware": "SAGE-Filter",
    "unsup_set_aware": "Unsup. SAGE",
    "kl_reg": "KL Regularization",
    "clean_only": "Clean-Only",
    "clean_ft": "Direct Fine-tuning",
    "clean_mix": "Direct Mixing",
    "batch_mean": "Batch-Mean",
    "baseline": "Pseudo-only",
    "pointwise": "Pointwise",
    "no_filter": "No Filter",
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


def method_sort_key(method: str, domain: str) -> Tuple[int, str]:
    order = CIFAR_METHOD_ORDER if domain == "cifar" else GPT2_METHOD_ORDER
    if method in order:
        return order.index(method), method
    return len(order), method


def read_cifar_rows(results_dir: pathlib.Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(results_dir.glob("rebuttal_cifar_seed*_merged.csv")):
        seed = int(path.stem.split("seed")[-1].split("_")[0])
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gen = safe_float(row.get("generation"))
                if gen is None:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "method": str(row.get("method", "")).strip(),
                        "generation": int(gen),
                        "acc": safe_float(row.get("acc")),
                        "worst_class_acc": safe_float(row.get("worst_class_acc")),
                        "sel_pseudo_acc": safe_float(row.get("sel_pseudo_acc")),
                        "sel_mean_conf": safe_float(row.get("sel_mean_conf")),
                        "sel_mean_weight": safe_float(row.get("sel_mean_weight")),
                        "sel_mean_score": safe_float(row.get("sel_mean_score")),
                    }
                )
    return rows


def read_gpt2_rows(results_dirs: Sequence[pathlib.Path]) -> List[Dict[str, object]]:
    rows_by_key: Dict[Tuple[int, str, int], Dict[str, object]] = {}
    for results_dir in results_dirs:
        if not results_dir.exists():
            continue
        for seed_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
            if not seed_dir.name.isdigit():
                continue
            seed = int(seed_dir.name)
            csv_path = seed_dir / "metrics_diversity_ppl.csv"
            if not csv_path.exists():
                continue
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    gen = safe_float(row.get("generation"))
                    if gen is None:
                        continue
                    method = str(row.get("method", "")).strip()
                    payload = {
                        "seed": seed,
                        "method": method,
                        "generation": int(gen),
                        "distinct2": safe_float(row.get("distinct2")),
                        "distinct3": safe_float(row.get("distinct3")),
                        "distinct4": safe_float(row.get("distinct4")),
                        "val_ppl": safe_float(row.get("val_ppl")),
                        "train_unique_line_ratio": safe_float(row.get("train_unique_line_ratio")),
                        "train_rep4_intra": safe_float(row.get("train_rep4_intra")),
                        "train_gzip_ratio": safe_float(row.get("train_gzip_ratio")),
                        "train_avg_words": safe_float(row.get("train_avg_words")),
                        "train_clean_refs": safe_float(row.get("train_clean_refs")),
                        "batch_mean_avg_align": safe_float(row.get("batch_mean_avg_align")),
                        "batch_mean_avg_weight": safe_float(row.get("batch_mean_avg_weight")),
                    }
                    rows_by_key[(seed, method, int(gen))] = payload
    return [rows_by_key[key] for key in sorted(rows_by_key.keys())]


def parse_path_list(raw: str | pathlib.Path) -> List[pathlib.Path]:
    if isinstance(raw, pathlib.Path):
        return [raw]
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    return [pathlib.Path(part) for part in parts]


def group_rows(rows: Iterable[Dict[str, object]]) -> Dict[Tuple[str, int], Dict[int, Dict[str, object]]]:
    grouped: Dict[Tuple[str, int], Dict[int, Dict[str, object]]] = defaultdict(dict)
    for row in rows:
        method = str(row["method"])
        seed = int(row["seed"])
        gen = int(row["generation"])
        grouped[(method, seed)][gen] = row
    return grouped


def determine_target_generation(
    grouped: Dict[Tuple[str, int], Dict[int, Dict[str, object]]],
    requested_gen: int | None,
) -> int:
    if requested_gen is not None:
        return int(requested_gen)
    max_gens = [max(gen_map.keys()) for gen_map in grouped.values() if gen_map]
    if not max_gens:
        return 0
    return min(max_gens)


def build_final_summary(
    rows: List[Dict[str, object]],
    domain: str,
    gen: int | None,
) -> Tuple[List[Dict[str, object]], int]:
    grouped = group_rows(rows)
    target_gen = determine_target_generation(grouped, gen)
    per_method: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for (method, seed), gen_map in grouped.items():
        if 0 not in gen_map or target_gen not in gen_map:
            continue
        g0 = gen_map[0]
        gt = gen_map[target_gen]
        merged = dict(gt)
        merged["seed"] = seed
        if domain == "cifar":
            merged["acc_delta_from_g0"] = (
                None if gt["acc"] is None or g0["acc"] is None else float(gt["acc"]) - float(g0["acc"])
            )
            merged["worst_delta_from_g0"] = (
                None
                if gt["worst_class_acc"] is None or g0["worst_class_acc"] is None
                else float(gt["worst_class_acc"]) - float(g0["worst_class_acc"])
            )
        else:
            merged["distinct4_delta_from_g0"] = (
                None
                if gt["distinct4"] is None or g0["distinct4"] is None
                else float(gt["distinct4"]) - float(g0["distinct4"])
            )
            merged["val_ppl_delta_from_g0"] = (
                None if gt["val_ppl"] is None or g0["val_ppl"] is None else float(gt["val_ppl"]) - float(g0["val_ppl"])
            )
        per_method[method].append(merged)

    summary_rows: List[Dict[str, object]] = []
    for method in sorted(per_method.keys(), key=lambda m: method_sort_key(m, domain)):
        seed_rows = per_method[method]
        seeds = sorted(int(r["seed"]) for r in seed_rows)
        out: Dict[str, object] = {
            "domain": domain,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "generation": target_gen,
            "n_seeds": len(seed_rows),
            "seeds": ",".join(str(s) for s in seeds),
        }
        if domain == "cifar":
            metric_names = [
                "acc",
                "worst_class_acc",
                "acc_delta_from_g0",
                "worst_delta_from_g0",
                "sel_pseudo_acc",
                "sel_mean_conf",
            ]
        else:
            metric_names = [
                "distinct4",
                "val_ppl",
                "distinct4_delta_from_g0",
                "val_ppl_delta_from_g0",
                "train_rep4_intra",
                "train_unique_line_ratio",
            ]
        for metric in metric_names:
            vals = [r.get(metric) for r in seed_rows]
            mean, std = summarize([v for v in vals if v is not None])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_pm"] = format_pm(mean, std)
        summary_rows.append(out)
    return summary_rows, target_gen


def build_trajectory_summary(
    rows: List[Dict[str, object]],
    domain: str,
) -> List[Dict[str, object]]:
    by_method_gen: Dict[Tuple[str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_method_gen[(str(row["method"]), int(row["generation"]))].append(row)

    summary_rows: List[Dict[str, object]] = []
    for method, gen in sorted(by_method_gen.keys(), key=lambda x: (method_sort_key(x[0], domain), x[1])):
        seed_rows = by_method_gen[(method, gen)]
        out: Dict[str, object] = {
            "domain": domain,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "generation": gen,
            "n_seeds": len(seed_rows),
        }
        if domain == "cifar":
            metrics = ["acc", "worst_class_acc", "sel_pseudo_acc"]
        else:
            metrics = ["distinct4", "val_ppl", "train_rep4_intra"]
        for metric in metrics:
            vals = [r.get(metric) for r in seed_rows if r.get(metric) is not None]
            mean, std = summarize(vals)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_pm"] = format_pm(mean, std)
        summary_rows.append(out)
    return summary_rows


def write_csv(path: pathlib.Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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


def build_markdown(final_cifar: List[Dict[str, object]], final_gpt2: List[Dict[str, object]]) -> str:
    lines: List[str] = ["# Rebuttal Tables", ""]
    if final_cifar:
        lines.append("## CIFAR-10")
        headers = ["Method", "Gen", "Acc", "Worst-class Acc", "ΔWorst vs G0", "Sel. Pseudo Acc"]
        rows = [
            [
                str(r["method_label"]),
                str(r["generation"]),
                str(r["acc_pm"]),
                str(r["worst_class_acc_pm"]),
                str(r["worst_delta_from_g0_pm"]),
                str(r["sel_pseudo_acc_pm"]),
            ]
            for r in final_cifar
        ]
        lines.append(markdown_table(headers, rows))
        lines.append("")
    if final_gpt2:
        lines.append("## GPT-2")
        headers = ["Method", "Gen", "Distinct-4", "Val PPL", "ΔDistinct-4 vs G0", "ΔVal PPL vs G0", "Rep-4 Intra"]
        rows = [
            [
                str(r["method_label"]),
                str(r["generation"]),
                str(r["distinct4_pm"]),
                str(r["val_ppl_pm"]),
                str(r["distinct4_delta_from_g0_pm"]),
                str(r["val_ppl_delta_from_g0_pm"]),
                str(r["train_rep4_intra_pm"]),
            ]
            for r in final_gpt2
        ]
        lines.append(markdown_table(headers, rows))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def print_console_summary(final_rows: List[Dict[str, object]], domain: str) -> None:
    if not final_rows:
        print(f"{domain}: no complete rows found")
        return
    print(f"\n== {domain.upper()} ==")
    for row in final_rows:
        if domain == "cifar":
            print(
                f"- {row['method_label']}: gen={row['generation']} "
                f"acc={row['acc_pm']} worst={row['worst_class_acc_pm']} "
                f"delta_worst={row['worst_delta_from_g0_pm']}"
            )
        else:
            print(
                f"- {row['method_label']}: gen={row['generation']} "
                f"distinct4={row['distinct4_pm']} val_ppl={row['val_ppl_pm']} "
                f"delta_d4={row['distinct4_delta_from_g0_pm']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize rebuttal experiment outputs into CSV and Markdown tables.")
    parser.add_argument("--cifar-dir", type=pathlib.Path, default=DEFAULT_CIFAR_DIR)
    parser.add_argument(
        "--gpt2-dir",
        type=str,
        default=str(DEFAULT_GPT2_DIR),
        help="GPT-2 results dir, or a comma-separated list of result dirs to merge.",
    )
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--gen",
        type=int,
        default=None,
        help="Generation to summarize. Default: auto-pick the largest generation shared by all available seed/method runs per domain.",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cifar_rows = read_cifar_rows(pathlib.Path(args.cifar_dir)) if pathlib.Path(args.cifar_dir).exists() else []
    gpt2_rows = read_gpt2_rows(parse_path_list(args.gpt2_dir))

    final_cifar, cifar_gen = build_final_summary(cifar_rows, domain="cifar", gen=args.gen)
    final_gpt2, gpt2_gen = build_final_summary(gpt2_rows, domain="gpt2", gen=args.gen)
    traj_cifar = build_trajectory_summary(cifar_rows, domain="cifar")
    traj_gpt2 = build_trajectory_summary(gpt2_rows, domain="gpt2")

    write_csv(out_dir / "cifar_final.csv", final_cifar)
    write_csv(out_dir / "gpt2_final.csv", final_gpt2)
    write_csv(out_dir / "cifar_trajectory.csv", traj_cifar)
    write_csv(out_dir / "gpt2_trajectory.csv", traj_gpt2)

    md = build_markdown(final_cifar, final_gpt2)
    md_path = out_dir / "rebuttal_tables.md"
    md_path.write_text(md, encoding="utf-8")

    print_console_summary(final_cifar, "cifar")
    print_console_summary(final_gpt2, "gpt2")
    print(f"\nWrote CIFAR final table to {out_dir / 'cifar_final.csv'} (gen={cifar_gen})")
    print(f"Wrote GPT-2 final table to {out_dir / 'gpt2_final.csv'} (gen={gpt2_gen})")
    print(f"Wrote Markdown tables to {md_path}")


if __name__ == "__main__":
    main()
