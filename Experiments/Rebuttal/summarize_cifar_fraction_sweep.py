import argparse
import csv
import pathlib
import statistics
from collections import defaultdict
from typing import Dict, List, Tuple


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results" / "cifar10_fraction_sweep"
DEFAULT_OUT_DIR = SCRIPT_DIR / "results" / "summary_fraction_sweep"


def method_to_alpha(method: str) -> float | None:
    if not method.startswith("frac_"):
        return None
    text = method[len("frac_") :].replace("p", ".")
    try:
        pct = float(text)
    except ValueError:
        return None
    return pct / 100.0


def fraction_counts(alpha: float, total_size: int) -> Tuple[int, int]:
    pseudo_count = int(round(alpha * total_size))
    pseudo_count = max(0, min(total_size, pseudo_count))
    clean_count = total_size - pseudo_count
    return pseudo_count, clean_count


def summarize(vals: List[float]) -> Tuple[float | None, float | None]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CIFAR-10 fraction sweep at the final generation.")
    parser.add_argument("--results-dir", type=pathlib.Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--generation", type=int, default=5)
    parser.add_argument("--total-size", type=int, default=4000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_method: Dict[str, List[Dict[str, float | int]]] = defaultdict(list)

    for path in sorted(args.results_dir.glob("rebuttal_cifar_seed*_merged.csv")):
        seed = int(path.stem.split("seed")[-1].split("_")[0])
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                method = str(row.get("method", "")).strip()
                alpha = method_to_alpha(method)
                if alpha is None:
                    continue
                if int(float(row.get("generation", -1))) != int(args.generation):
                    continue
                worst = float(row["worst_class_acc"])
                by_method[method].append({"seed": seed, "worst_class_acc": worst})

    summary_rows: List[Dict[str, object]] = []
    for method in sorted(by_method.keys(), key=lambda m: method_to_alpha(m) or 0.0):
        alpha = float(method_to_alpha(method))
        pseudo_count, clean_count = fraction_counts(alpha, int(args.total_size))
        vals = [float(r["worst_class_acc"]) for r in by_method[method]]
        mean, std = summarize(vals)
        summary_rows.append(
            {
                "method": method,
                "alpha": alpha,
                "alpha_pct": alpha * 100.0,
                "generation": int(args.generation),
                "n_seeds": len(vals),
                "seeds": ",".join(str(int(r["seed"])) for r in sorted(by_method[method], key=lambda x: int(x["seed"]))),
                "pseudo_count": pseudo_count,
                "clean_count": clean_count,
                "worst_class_acc_mean": mean,
                "worst_class_acc_std": std,
                "worst_class_acc_pm": format_pm(mean, std),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"cifar_fraction_sweep_g{int(args.generation)}.csv"
    md_path = args.out_dir / f"cifar_fraction_sweep_g{int(args.generation)}.md"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "alpha",
                "alpha_pct",
                "generation",
                "n_seeds",
                "seeds",
                "pseudo_count",
                "clean_count",
                "worst_class_acc_mean",
                "worst_class_acc_std",
                "worst_class_acc_pm",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# CIFAR-10 Fraction Sweep\n\n")
        f.write("| Synthetic Fraction | Pseudo | Clean | Gen | Worst-class Acc |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for row in summary_rows:
            f.write(
                f"| {row['alpha_pct']:.1f}% | {row['pseudo_count']} | {row['clean_count']} | "
                f"{row['generation']} | {row['worst_class_acc_pm']} |\n"
            )

    print(f"Wrote CSV summary to {csv_path}")
    print(f"Wrote Markdown summary to {md_path}")


if __name__ == "__main__":
    main()
