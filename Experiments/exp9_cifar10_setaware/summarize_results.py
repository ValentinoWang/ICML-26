import argparse
import csv
import pathlib
import statistics
from typing import Dict, List, Tuple

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_OUT_CSV = ROOT_DIR / "Total_results" / "Tables" / "exp9_cifar10_g5_summary.csv"


def read_gen_metrics(path: pathlib.Path, gen: int) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """
    Returns: {method: {seed: (acc, worst)}}
    """
    out: Dict[str, Dict[int, Tuple[float, float]]] = {}
    seed = int(path.stem.split("seed")[-1].split("_")[0])
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row_gen = int(float(row.get("generation", -1)))
            except (TypeError, ValueError):
                continue
            if row_gen != gen:
                continue
            method = str(row.get("method", "")).strip() or "unknown"
            try:
                acc = float(row.get("acc", "nan"))
                worst = float(row.get("worst_class_acc", "nan"))
            except (TypeError, ValueError):
                continue
            out.setdefault(method, {})[seed] = (acc, worst)
    return out


def summarize(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results_dirs", nargs="+", help="One or more exp9 results directories (each contains exp9_seed*_merged.csv).")
    p.add_argument("--gen", type=int, default=5, help="Which generation to summarize.")
    p.add_argument("--out-csv", type=str, default=str(DEFAULT_OUT_CSV), help="Optional CSV output path (empty to skip).")
    args = p.parse_args()

    rows: List[Dict[str, object]] = []
    for d in args.results_dirs:
        results_dir = pathlib.Path(d)
        merged_paths = sorted(results_dir.glob("exp9_seed*_merged.csv"))
        if not merged_paths:
            print(f"{results_dir}: (no merged CSV found)")
            continue
        per_method: Dict[str, Dict[int, Tuple[float, float]]] = {}
        for mp in merged_paths:
            metrics = read_gen_metrics(mp, gen=int(args.gen))
            for method, seed_map in metrics.items():
                per_method.setdefault(method, {}).update(seed_map)

        print(f"\n== {results_dir} (gen={args.gen}) ==")
        for method in sorted(per_method.keys()):
            seeds = sorted(per_method[method].keys())
            accs = [per_method[method][s][0] for s in seeds]
            worsts = [per_method[method][s][1] for s in seeds]
            acc_mean, acc_std = summarize(accs)
            worst_mean, worst_std = summarize(worsts)
            seeds_str = ", ".join(f"{s}: acc={per_method[method][s][0]:.4f}, worst={per_method[method][s][1]:.4f}" for s in seeds)
            print(f"- {method}: n={len(seeds)} acc={acc_mean:.4f}±{acc_std:.4f} worst={worst_mean:.4f}±{worst_std:.4f}")
            print(f"  {seeds_str}")
            rows.append(
                {
                    "results_dir": str(results_dir),
                    "method": method,
                    "gen": int(args.gen),
                    "n_seeds": len(seeds),
                    "seeds": ",".join(str(s) for s in seeds),
                    "acc_mean": acc_mean,
                    "acc_std": acc_std,
                    "worst_mean": worst_mean,
                    "worst_std": worst_std,
                }
            )

    if args.out_csv.strip() and rows:
        out_path = pathlib.Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote summary CSV to {out_path}")


if __name__ == "__main__":
    main()
