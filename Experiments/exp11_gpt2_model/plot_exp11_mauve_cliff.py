import argparse
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

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_PPL_ROOT = ROOT_DIR / "Total_results" / "Tables" / "exp11_gpt2_model" / "Results_streaming"
DEFAULT_PPL_ABLATE_ROOT = (
    ROOT_DIR / "Total_results" / "Tables" / "exp11_gpt2_model" / "Results_streaming_ablate"
)
DEFAULT_MAUVE_CSV = ROOT_DIR / "exp11_gpt2_model" / "MAUVE" / "dphi1_leash_final_streaming" / "mauve_g0_g4.csv"
DEFAULT_MAUVE_EXTRA_CSV = (
    ROOT_DIR
    / "exp11_gpt2_model"
    / "MAUVE"
    / "dphi1_leash_final_streaming_dispersion"
    / "mauve_g0_g4.csv"
)
DEFAULT_OUT_PDF = ROOT_DIR / "Paper" / "LaTEX" / "icml2025" / "Figures" / "exp11_mauve_cliff.pdf"
DEFAULT_OUT_SVG = ROOT_DIR / "Paper" / "LaTEX" / "icml2025" / "Figures" / "exp11_mauve_cliff.svg"
DEFAULT_OUT_PDF_APPENDIX = (
    ROOT_DIR / "Paper" / "LaTEX" / "icml2025" / "Figures" / "exp11_mauve_cliff_dispersion.pdf"
)
DEFAULT_OUT_SVG_APPENDIX = (
    ROOT_DIR / "Paper" / "LaTEX" / "icml2025" / "Figures" / "exp11_mauve_cliff_dispersion.svg"
)

STYLES = {
    "no_filter": {
        "label": "No Filter",
        "c": "#949494",
        "ls": "-",
        "mk": "o",
        "lw": 1.2,
        "alpha": 0.9,
        "fill_alpha": 0.0,
    },
    "pointwise": {
        "label": "Pointwise",
        "c": "#C44E52",
        "ls": (0, (4, 2)),
        "mk": "^",
        "lw": 1.2,
        "alpha": 0.9,
        "fill_alpha": 0.08,
    },
    "dispersion": {
        "label": "Dispersion",
        "c": "#55A868",
        "ls": "-",
        "mk": "s",
        "lw": 1.2,
        "alpha": 0.9,
        "fill_alpha": 0.08,
    },
    "set_aware": {
        "label": "Set-Aware",
        "c": "#4C72B0",
        "ls": "-",
        "mk": "D",
        "lw": 1.2,
        "alpha": 1.0,
        "fill_alpha": 0.12,
    },
}


def summarize_ci(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for (method, gen), grp in df.groupby(["method", "generation"]):
        vals = grp[value_col].astype(float).values
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        rows.append({"method": method, "generation": gen, "mean": mean, "ci95": ci95})
    return pd.DataFrame(rows)


def load_ppl_summary(results_root: pathlib.Path, extra_root: pathlib.Path | None = None) -> pd.DataFrame:
    frames = []
    for seed_dir in results_root.iterdir():
        if not seed_dir.is_dir():
            continue
        csv_path = seed_dir / "metrics_diversity_ppl.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        frames.append(df[["method", "generation", "val_ppl"]])
    if extra_root:
        for seed_dir in extra_root.iterdir():
            if not seed_dir.is_dir():
                continue
            csv_path = seed_dir / "metrics_diversity_ppl.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            frames.append(df[df["method"] == "dispersion"][["method", "generation", "val_ppl"]])
    if not frames:
        raise FileNotFoundError(f"No metrics_diversity_ppl.csv found under {results_root}")
    df = pd.concat(frames, ignore_index=True)
    return summarize_ci(df, "val_ppl")


def load_mauve_summary(mauve_csv: pathlib.Path, extra_csv: pathlib.Path | None = None) -> pd.DataFrame:
    if not mauve_csv.exists():
        raise FileNotFoundError(f"MAUVE CSV not found: {mauve_csv}")
    df = pd.read_csv(mauve_csv)
    if extra_csv:
        if not extra_csv.exists():
            raise FileNotFoundError(f"Extra MAUVE CSV not found: {extra_csv}")
        extra = pd.read_csv(extra_csv)
        df = pd.concat([df, extra], ignore_index=True)
    return summarize_ci(df, "mauve")


def _label_positions(values: dict[str, float], y_min: float, y_max: float, spacing: float) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    if not ordered:
        return {}
    max_val = ordered[0][1]
    top_target = min(y_max - spacing, max_val + spacing * (len(ordered) - 1) / 2.0)
    positions = [top_target - i * spacing for i in range(len(ordered))]
    bottom = positions[-1]
    if bottom < y_min + spacing:
        shift = (y_min + spacing) - bottom
        positions = [y + shift for y in positions]
        top = positions[0]
        if top > y_max - spacing:
            shift2 = top - (y_max - spacing)
            positions = [y - shift2 for y in positions]
    return {method: y for (method, _), y in zip(ordered, positions)}


def _annotate_end_labels(
    ax: plt.Axes,
    x: float,
    values: dict[str, float],
    formatter: str,
    spacing: float,
    x_text: float,
) -> None:
    y_min, y_max = ax.get_ylim()
    positions = _label_positions(values, y_min, y_max, spacing)
    for method, value in values.items():
        style = STYLES[method]
        label = f"{style['label']}\n{value:{formatter}}"
        ax.annotate(
            label,
            xy=(x, value),
            xytext=(x_text, positions[method]),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=8,
            color=style["c"],
            arrowprops=dict(arrowstyle="-", lw=0.6, color=style["c"]),
        )


def plot_cliff(
    ppl_df: pd.DataFrame,
    mauve_df: pd.DataFrame,
    out_pdf: pathlib.Path,
    out_svg: pathlib.Path,
    methods: list[str],
    legend_cols: int,
) -> None:
    gens = np.arange(0, 5)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), constrained_layout=True)

    # Left: MAUVE
    ax = axes[0]
    mauve_g4 = {}
    for method in methods:
        style = STYLES[method]
        sub = mauve_df[mauve_df["method"] == method].sort_values("generation")
        means = np.array([sub[sub["generation"] == g]["mean"].values[0] for g in gens])
        ci = np.array([sub[sub["generation"] == g]["ci95"].values[0] for g in gens])
        zorder = 3 if method == "set_aware" else 2
        line = ax.plot(
            gens,
            means,
            color=style["c"],
            linestyle=style["ls"],
            marker=style["mk"],
            linewidth=style["lw"],
            alpha=style["alpha"],
            markersize=5,
            label=style["label"],
            zorder=zorder,
            )
        if style["fill_alpha"] > 0:
            ax.fill_between(
                gens,
                means - ci,
                means + ci,
                color=style["c"],
                alpha=style["fill_alpha"],
                linewidth=0,
                zorder=zorder - 1,
            )
        mauve_g4[method] = float(means[-1])
        ax.set_title("MAUVE ↑", pad=6)
    ax.set_xlabel("Generation")
    ax.set_ylabel("MAUVE")
    ax.set_xticks(gens)
    ax.set_xticklabels([str(g) for g in gens])
    ax.set_ylim(0.0, 0.25)
    ax.set_xlim(-0.1, 4.6)
    ax.grid(True, which="major", axis="both")
    _annotate_end_labels(ax, gens[-1], mauve_g4, ".3f", spacing=0.03, x_text=4.25)

    # Right: PPL
    ax = axes[1]
    ppl_g4 = {}
    for method in methods:
        style = STYLES[method]
        sub = ppl_df[ppl_df["method"] == method].sort_values("generation")
        means = np.array([sub[sub["generation"] == g]["mean"].values[0] for g in gens])
        ci = np.array([sub[sub["generation"] == g]["ci95"].values[0] for g in gens])
        zorder = 3 if method == "set_aware" else 2
        ax.plot(
            gens,
            means,
            color=style["c"],
            linestyle=style["ls"],
            marker=style["mk"],
            linewidth=style["lw"],
            alpha=style["alpha"],
            markersize=5,
            label=style["label"],
            zorder=zorder,
            )
        if style["fill_alpha"] > 0:
            ax.fill_between(
                gens,
                means - ci,
                means + ci,
                color=style["c"],
                alpha=style["fill_alpha"],
                linewidth=0,
                zorder=zorder - 1,
            )
        ppl_g4[method] = float(means[-1])
        if method == "pointwise":
            ax.annotate(
                "Pointwise worst at G4",
                xy=(gens[-1], means[-1]),
                xytext=(1.0, means[-1] - 45),
                textcoords="data",
                arrowprops=dict(arrowstyle="->", lw=0.8, color=style["c"]),
                fontsize=8,
                color=style["c"],
            )
        ax.set_title("Streaming PPL ↓", pad=6)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Validation PPL")
    ax.set_xticks(gens)
    ax.set_xticklabels([str(g) for g in gens])
    ax.set_ylim(35, 240)
    ax.set_xlim(-0.1, 4.6)
    ax.grid(True, which="major", axis="both")
    _annotate_end_labels(ax, gens[-1], ppl_g4, ".1f", spacing=20.0, x_text=4.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=legend_cols, frameon=False, bbox_to_anchor=(0.5, 1.12))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppl-root", type=pathlib.Path, default=DEFAULT_PPL_ROOT)
    parser.add_argument("--ppl-ablate-root", type=pathlib.Path, default=DEFAULT_PPL_ABLATE_ROOT)
    parser.add_argument("--mauve-csv", type=pathlib.Path, default=DEFAULT_MAUVE_CSV)
    parser.add_argument("--extra-mauve-csv", type=pathlib.Path, default=DEFAULT_MAUVE_EXTRA_CSV)
    parser.add_argument("--out-pdf", type=pathlib.Path, default=DEFAULT_OUT_PDF)
    parser.add_argument("--out-svg", type=pathlib.Path, default=DEFAULT_OUT_SVG)
    parser.add_argument("--appendix-out-pdf", type=pathlib.Path, default=DEFAULT_OUT_PDF_APPENDIX)
    parser.add_argument("--appendix-out-svg", type=pathlib.Path, default=DEFAULT_OUT_SVG_APPENDIX)
    args = parser.parse_args()

    ppl_df = load_ppl_summary(args.ppl_root, extra_root=args.ppl_ablate_root)
    mauve_df = load_mauve_summary(args.mauve_csv, extra_csv=args.extra_mauve_csv)
    main_methods = ["no_filter", "pointwise", "set_aware"]
    appendix_methods = ["no_filter", "pointwise", "dispersion", "set_aware"]
    plot_cliff(ppl_df, mauve_df, args.out_pdf, args.out_svg, methods=main_methods, legend_cols=3)
    if args.appendix_out_pdf and args.appendix_out_svg:
        plot_cliff(
            ppl_df,
            mauve_df,
            args.appendix_out_pdf,
            args.appendix_out_svg,
            methods=appendix_methods,
            legend_cols=4,
        )
        print(f"Saved appendix cliff figure to {args.appendix_out_pdf}")
    print(f"Saved cliff figure to {args.out_pdf}")


if __name__ == "__main__":
    main()
