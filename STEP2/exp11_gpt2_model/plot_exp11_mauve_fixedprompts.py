import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.2,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "figure.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "exp11_gpt2_model" / "MAUVE" / "fixedprompts" / "mauve_g4_fixedprompts.csv"
OUT_PDF = ROOT / "Paper" / "LaTEX" / "icml2025" / "Figures" / "exp11_mauve_g4_fixedprompts.pdf"
OUT_PNG = ROOT / "Paper" / "LaTEX" / "icml2025" / "Figures" / "exp11_mauve_g4_fixedprompts.png"

STYLES = {
    "no_filter": {"label": "No Filter", "c": "#949494"},
    "pointwise": {"label": "Pointwise", "c": "#C44E52"},
    "dispersion": {"label": "Dispersion", "c": "#55A868"},
    "set_aware": {"label": "Set-Aware", "c": "#4C72B0"},
}


def summarize_ci(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, grp in df.groupby("method"):
        vals = grp["mauve"].astype(float).values
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        rows.append({"method": method, "mean": mean, "ci95": ci95})
    return pd.DataFrame(rows)


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing MAUVE CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    summary = summarize_ci(df)

    order = ["no_filter", "pointwise", "dispersion", "set_aware"]
    summary = summary.set_index("method").loc[order].reset_index()

    x = np.arange(len(order))
    means = summary["mean"].values
    ci = summary["ci95"].values
    colors = [STYLES[m]["c"] for m in order]
    labels = [STYLES[m]["label"] for m in order]

    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.bar(x, means, color=colors, edgecolor="none", alpha=0.9)
    ax.errorbar(x, means, yerr=ci, fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("MAUVE (G4)")
    ax.set_title("Fixed-Prompt MAUVE (GPT-2)")
    ax.set_ylim(0.0, max(means + ci) * 1.35)
    ax.grid(True, axis="y")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
