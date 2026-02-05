import argparse
import json
from typing import Dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize A/B judge results.")
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--latex-out")
    args = parser.parse_args()

    counts: Dict[str, int] = {"A": 0, "B": 0, "tie": 0}
    total = 0
    with open(args.results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            winner = obj.get("winner", "tie")
            if winner not in counts:
                winner = "tie"
            counts[winner] += 1
            total += 1

    win_rate = counts["A"] / total if total else 0.0
    tie_rate = counts["tie"] / total if total else 0.0
    print(
        f"Total: {total} | {args.label_a} wins: {counts['A']} | {args.label_b} wins: {counts['B']} | "
        f"ties: {counts['tie']} | {args.label_a} win-rate: {win_rate:.3f} | tie-rate: {tie_rate:.3f}"
    )

    if args.latex_out:
        row = (
            f"{args.label_a} vs {args.label_b} & {counts['A']} & {counts['B']} & {counts['tie']} & "
            f"{win_rate:.3f} \\\\" 
        )
        with open(args.latex_out, "w", encoding="utf-8") as f:
            f.write(row + "\n")


if __name__ == "__main__":
    main()
