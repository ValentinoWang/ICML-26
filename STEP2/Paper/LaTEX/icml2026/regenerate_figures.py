"""
Regenerate paper figures by invoking each experiment's plotting helpers.
No experiments are rerun; we read existing CSV/NPZ outputs and write figures to the ICML Figures directory.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "Paper" / "LaTEX" / "icml2026" / "Figures"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import plotting helpers from each experiment
from exp1_bias_sources.plot_exp1 import load_series_from_csv, plot_from_series
from exp2_bias_sensitivity.plot_exp2 import plot_by_method, plot_param_curve, plot_trajectories_grid
from exp3_data_efficiency.make_figures_exp3 import (
    plot_breaking_floor,
    plot_data_efficiency,
    plot_data_efficiency_and_robustness,
    plot_robustness_cost,
)
from exp4_bias_correction_visualization.plot_exp4 import (
    plot_exp41,
    plot_exp41_enhance,
    plot_exp42,
    plot_exp42_phase,
    plot_exp42_43_combined,
    plot_exp43,
)
from exp5_high_dim_scalability.make_fig_bias_reduction import METHODS as EXP5_METHODS
from exp5_high_dim_scalability.make_fig_bias_reduction import plot_adaptive, plot_tail_vs_dim, plot_trajectories
from exp6_arch_ablation.plot_exp6 import METHODS as EXP6_METHODS
from exp6_arch_ablation.plot_exp6 import plot_set_size_trajs, plot_tail_bar as plot_exp6_tail_bar, plot_trajectories as plot_exp6_trajs
from exp7_recursive_regression.plot_exp7_recursive import plot_series as plot_exp7_series
from exp8_mnist_recursive.plot_exp8 import plot_series as plot_exp8_series
from exp8_mnist_recursive.make_exp8_enhance import compose as compose_exp8_enhence
from exp9_cifar10_setaware import plot_exp9_times
from exp11_gpt2_model.plot_exp11_phase import (
    aggregate_mean as aggregate_exp11_mean,
    load_results as load_exp11_results,
    plot_phase_portrait,
    plot_dual_axis,
    plot_recursive_lines,
)


def copy_if_exists(src: pathlib.Path, dest: pathlib.Path) -> None:
    try:
        if src.resolve() == dest.resolve():
            return
    except FileNotFoundError:
        pass
    if src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)


def read_csv(path: pathlib.Path) -> Dict[str, List[float]]:
    with path.open() as f:
        header = f.readline().strip().split(",")
        cols: Dict[str, List[float]] = {h: [] for h in header}
        for line in f:
            parts = line.strip().split(",")
            for h, v in zip(header, parts):
                cols[h].append(float(v))
    return cols


# ---- Exp1 ----
def regenerate_exp1() -> None:
    exp1_dir = ROOT / "exp1_bias_sources" / "results"
    tasks = [
        ("exp1_1.1_const", exp1_dir / "exp1_1.1_const.csv"),
        ("exp1_1.2_ridge", exp1_dir / "exp1_1.2_ridge.csv"),
        ("exp1_1.3_bayes", exp1_dir / "exp1_1.3_bayes.csv"),
    ]
    for name, csv_path in tasks:
        if csv_path.exists():
            try:
                gens, cfg_order, series = load_series_from_csv(csv_path)
                plot_from_series(gens, cfg_order, series, FIG_DIR / f"{name}.png")
                continue
            except Exception:
                # 回退到直接复制已有图像
                copy_if_exists(exp1_dir / f"{name}.png", FIG_DIR / f"{name}.png")
        else:
            copy_if_exists(exp1_dir / f"{name}.png", FIG_DIR / f"{name}.png")


# ---- Exp2 ----
def regenerate_exp2() -> None:
    res_dir = ROOT / "exp2_bias_sensitivity" / "results"
    # 直接使用已有绘图输出
    for name in [
        "exp2_2.1_bias_vs_error.png",
        "exp2_2.1_trajs_by_method.png",
        "exp2_2.1_trajs_by_bias.png",
        "exp2_2.2_alpha_vs_error.png",
        "exp2_2.2_trajs_by_alpha.png",
        "exp2_2.2_trajs_by_method.png",
        "exp2_2.3_prior_offset_curves.png",
        "exp2_2.3_trajs_n5.png",
        "exp2_2.3_trajs_n3.png",
        "exp2_2.3_trajs_n50.png",
        "exp2_2.3_delta_log.png",
    ]:
        copy_if_exists(res_dir / name, FIG_DIR / name)


# ---- Exp3 ----
def regenerate_exp3() -> None:
    const = read_csv(ROOT / "exp3_data_efficiency" / "results" / "exp3_data_eff_const.csv")
    ridge = read_csv(ROOT / "exp3_data_efficiency" / "results" / "exp3_data_eff_ridge.csv")
    bayes_n5 = read_csv(ROOT / "exp3_data_efficiency" / "results" / "exp3_data_eff_bayes_n5.csv")
    bayes_n100 = read_csv(ROOT / "exp3_data_efficiency" / "results" / "exp3_data_eff_bayes.csv")

    plot_breaking_floor(const, FIG_DIR / "exp3_fig_A_breaking_floor.png")
    plot_data_efficiency(ridge, FIG_DIR / "exp3_fig_B_data_efficiency.png")
    plot_robustness_cost(bayes_n5, bayes_n100, FIG_DIR / "exp3_fig_C_robustness_cost.png")
    plot_data_efficiency_and_robustness(
        ridge,
        bayes_n5,
        bayes_n100,
        FIG_DIR / "exp3_fig_BC_combined.png",
    )


# ---- Exp4 ----
def _load_exp41_metrics(path: pathlib.Path) -> Dict[str, Dict[str, List[float]]]:
    data = read_csv(path)
    return {
        "error": {"mean": data["error_mean"], "std": data["error_std"]},
        "delta_norm": {"mean": data["delta_norm_mean"], "std": data["delta_norm_std"]},
        "dist": {"mean": data["dist_to_minus_b_mean"], "std": data["dist_to_minus_b_std"]},
        "cos": {"mean": data["cos_to_minus_b_mean"], "std": data["cos_to_minus_b_std"]},
        "ess": {"mean": data["ess_mean"], "std": data["ess_std"]},
        "w_var": {"mean": data["w_variance_mean"], "std": data["w_variance_std"]},
    }


def _load_exp42_metrics(path: pathlib.Path) -> Dict[str, Dict[str, List[float]]]:
    data = read_csv(path)
    return {
        "theta_norm": {"mean": data["theta_norm_mean"], "std": data["theta_norm_std"]},
        "delta_norm": {"mean": data["delta_norm_mean"], "std": data["delta_norm_std"]},
        "bias_norm": {"mean": data["bias_norm_mean"], "std": data["bias_norm_std"]},
        "cos_to_theta": {"mean": data["cos_to_theta_mean"], "std": data["cos_to_theta_std"]},
        "cos_to_correction": {"mean": data["cos_to_correction_mean"], "std": data["cos_to_correction_std"]},
        "error": {"mean": data["error_mean"], "std": data["error_std"]},
        "ess": {"mean": data["ess_mean"], "std": data["ess_std"]},
        "w_var": {"mean": data["w_variance_mean"], "std": data["w_variance_std"]},
    }


def regenerate_exp4() -> None:
    metrics41 = _load_exp41_metrics(ROOT / "exp4_bias_correction_visualization" / "results" / "exp4_4.1_base.csv")
    plot_exp41(metrics41, target_norm=0.5, out_dir=FIG_DIR, n_seeds=8)
    plot_exp41_enhance(metrics41, target_norm=0.5, out_dir=FIG_DIR, n_seeds=8)

    metrics42 = _load_exp42_metrics(ROOT / "exp4_bias_correction_visualization" / "results" / "exp4_4.2_ridge.csv")
    plot_exp42(metrics42, out_dir=FIG_DIR, n_seeds=8)
    plot_exp42_phase(metrics42, out_dir=FIG_DIR, n_seeds=8)

    bayes_csv = ROOT / "exp4_bias_correction_visualization" / "results" / "exp4_4.3_bayes_scatter.csv"
    if bayes_csv.exists():
        data = read_csv(bayes_csv)
        x_eval = np.array(data["x"])
        w_eval = np.array(data["weight_mean"])
        args = argparse.Namespace(mu_prior=0.0, bayes_true_mean=5.0)  # args only for labels
        plot_exp43(x_eval, w_eval, args=args, out_dir=FIG_DIR)
        if "delta_norm" in metrics42 and "theta_norm" in metrics42:
            plot_exp42_43_combined(metrics42, x_eval, w_eval, args=args, out_dir=FIG_DIR, n_seeds=8)


# ---- Exp5 ----
def _load_exp5_tails(path: pathlib.Path) -> Dict[int, Dict[str, Dict[str, float]]]:
    data = read_csv(path)
    tails: Dict[int, Dict[str, Dict[str, float]]] = {}
    for i, dim in enumerate(data["dim"]):
        tails[int(dim)] = {
            m: {"mean": data[f"{m}_mean"][i], "std": data[f"{m}_std"][i]} for m in EXP5_METHODS
        }
    return tails


def _load_exp5_series(result_dir: pathlib.Path, dims: List[int]) -> Dict[int, Dict[str, Dict[str, List[float]]]]:
    all_series: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    for d in dims:
        path = result_dir / f"exp5_dim{d}_trajectories.csv"
        if not path.exists():
            continue
        data = read_csv(path)
        series: Dict[str, Dict[str, List[float]]] = {}
        for m in EXP5_METHODS:
            series[m] = {"mean": data[f"{m}_mean"], "std": data[f"{m}_std"]}
        all_series[d] = series
    return all_series


def regenerate_exp5() -> None:
    result_dir = ROOT / "exp5_high_dim_scalability" / "results"
    tail_stats = _load_exp5_tails(result_dir / "exp5_tail_summary.csv")
    dims = sorted(tail_stats.keys())
    plot_tail_vs_dim(dims, tail_stats, FIG_DIR / "exp5_tail_vs_dim.png", n_seeds=8, methods=EXP5_METHODS)

    all_series = _load_exp5_series(result_dir, dims)
    if all_series:
        plot_trajectories(all_series, FIG_DIR / "exp5_trajs_by_dim.png", n_seeds=8, methods=EXP5_METHODS)
    # Bias reduction bar
    tails_adapt = {
        d: {
            "no_filter": v["no_filter"]["mean"],
            "mlp_filter": v["mlp_filter"]["mean"],
            "mlp_correction": v["mlp_correction"]["mean"],
            "ours": v["ours"]["mean"],
        }
        for d, v in tail_stats.items()
    }
    plot_adaptive(tails_adapt, FIG_DIR / "exp5_bias_reduction_rate.png")


# ---- Exp6 (non-gated) ----
def _load_exp6_series(result_dir: pathlib.Path, scenarios: List[str]) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    all_series: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for name in scenarios:
        path = result_dir / f"exp6_{name}_trajectories.csv"
        if not path.exists():
            continue
        data = read_csv(path)
        series: Dict[str, Dict[str, List[float]]] = {}
        for m in EXP6_METHODS:
            series[m] = {"mean": data[f"{m}_mean"], "std": data[f"{m}_std"]}
        all_series[name] = series
    return all_series


def _load_exp6_tails(result_dir: pathlib.Path, scenarios: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    tails: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name in scenarios:
        path = result_dir / f"exp6_{name}_tail_summary.csv"
        if not path.exists():
            continue
        data = read_csv(path)
        tails[name] = {
            m: {"mean": data[f"{m}_mean"][0], "std": data[f"{m}_std"][0]} for m in EXP6_METHODS
        }
    return tails


def regenerate_exp6() -> None:
    result_dir = ROOT / "exp6_arch_ablation" / "results"
    scenarios = ["bayes", "ridge", "complex"]
    all_series = _load_exp6_series(result_dir, scenarios)
    tails = _load_exp6_tails(result_dir, scenarios)
    if all_series:
        plot_exp6_trajs(all_series, FIG_DIR / "exp6_trajs.png", n_seeds=8)
    if tails:
        plot_exp6_tail_bar(tails, FIG_DIR / "exp6_tail_bar.png", n_seeds=8)
    # Set-size ablation (complex scenario)
    setsize_traj = result_dir / "exp6_setsize_complex_trajectories.csv"
    if setsize_traj.exists():
        data = read_csv(setsize_traj)
        sizes = sorted({int(k.split("_")[0][1:]) for k in data if k.startswith("N") and k.endswith("_mean")})
        series: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
        for n in sizes:
            series[n] = {
                "ours_full": {
                    "mean": data[f"N{n}_mean"],
                    "std": data[f"N{n}_std"],
                }
            }
        plot_set_size_trajs(series, FIG_DIR / "exp6_setsize_complex.png", n_seeds=8, scenario="complex")


# ---- Exp7 ----
def _load_exp7_stats(path: pathlib.Path) -> Dict[str, Dict[str, List[float]]]:
    data = read_csv(path)
    return {
        "mse": {
            "no_filter": {"mean": data["no_filter_mse_mean"], "std": data["no_filter_mse_std"]},
            "batch_stats": {"mean": data["batch_stats_mse_mean"], "std": data["batch_stats_mse_std"]},
            "dst": {"mean": data["dst_mse_mean"], "std": data["dst_mse_std"]},
            "l2ac": {"mean": data["l2ac_mse_mean"], "std": data["l2ac_mse_std"]},
            "ours": {"mean": data["ours_mse_mean"], "std": data["ours_mse_std"]},
        },
        "norm": {
            "no_filter": {"mean": data["no_filter_norm_mean"], "std": data["no_filter_norm_std"]},
            "batch_stats": {"mean": data["batch_stats_norm_mean"], "std": data["batch_stats_norm_std"]},
            "dst": {"mean": data["dst_norm_mean"], "std": data["dst_norm_std"]},
            "l2ac": {"mean": data["l2ac_norm_mean"], "std": data["l2ac_norm_std"]},
            "ours": {"mean": data["ours_norm_mean"], "std": data["ours_norm_std"]},
        },
    }


def regenerate_exp7() -> None:
    stats = _load_exp7_stats(ROOT / "exp7_recursive_regression" / "results" / "exp7_trajectories.csv")
    plot_exp7_series(stats, FIG_DIR, n_seeds=3)
    # variance/attention visuals from exp7_variance_attention
    va_dir = ROOT / "exp7_variance_attention" / "results"
    copy_if_exists(va_dir / "exp7_attention_maps.png", FIG_DIR / "exp7_attention_maps.png")
    copy_if_exists(va_dir / "exp7_tsne_latent.png", FIG_DIR / "exp7_tsne_latent.png")
    copy_if_exists(va_dir / "exp7_response_curve.png", FIG_DIR / "exp7_response_curve.png")
    copy_if_exists(va_dir / "exp7_combined.png", FIG_DIR / "exp7_response_tsne.png")


# ---- Exp8 ----
def _load_exp8_stats(path: pathlib.Path) -> Dict[str, Dict[str, List[float]]]:
    data = read_csv(path)
    return {
        "mse": {
            "no_filter": {"mean": data["no_filter_mse_mean"], "std": data["no_filter_mse_std"]},
            "mlp_filter": {"mean": data["mlp_filter_mse_mean"], "std": data["mlp_filter_mse_std"]},
            "batch_stats": {"mean": data["batch_stats_mse_mean"], "std": data["batch_stats_mse_std"]},
            "dst": {"mean": data["dst_mse_mean"], "std": data["dst_mse_std"]},
            "l2ac": {"mean": data["l2ac_mse_mean"], "std": data["l2ac_mse_std"]},
            "tent": {"mean": data["tent_mse_mean"], "std": data["tent_mse_std"]},
            "ours": {"mean": data["ours_mse_mean"], "std": data["ours_mse_std"]},
        },
        "norm": {
            "no_filter": {"mean": data["no_filter_norm_mean"], "std": data["no_filter_norm_std"]},
            "mlp_filter": {"mean": data["mlp_filter_norm_mean"], "std": data["mlp_filter_norm_std"]},
            "batch_stats": {"mean": data["batch_stats_norm_mean"], "std": data["batch_stats_norm_std"]},
            "dst": {"mean": data["dst_norm_mean"], "std": data["dst_norm_std"]},
            "l2ac": {"mean": data["l2ac_norm_mean"], "std": data["l2ac_norm_std"]},
            "tent": {"mean": data["tent_norm_mean"], "std": data["tent_norm_std"]},
            "ours": {"mean": data["ours_norm_mean"], "std": data["ours_norm_std"]},
        },
    }


def regenerate_exp8() -> None:
    stats = _load_exp8_stats(ROOT / "exp8_mnist_recursive" / "results" / "exp8_trajectories.csv")
    res_dir = ROOT / "exp8_mnist_recursive" / "results"
    plot_exp8_series(stats, res_dir, n_seeds=3)
    copy_if_exists(res_dir / "exp8_curves.png", FIG_DIR / "exp8_curves.png")
    # Grids
    copy_if_exists(res_dir / "exp8_visual_grid.png", FIG_DIR / "exp8_visual_grid.png")
    copy_if_exists(res_dir / "exp8_visual_grid_digit3.png", FIG_DIR / "exp8_visual_grid_digit3.png")
    # Enhanced stacked figure (curves + grid)
    if (res_dir / "exp8_curves.png").exists() and (res_dir / "exp8_visual_grid.png").exists():
        out_stacked = res_dir / "exp8_enhence.png"
        compose_exp8_enhence(res_dir / "exp8_curves.png", res_dir / "exp8_visual_grid.png", out_stacked)
        copy_if_exists(out_stacked, FIG_DIR / "exp8_enhence.png")


# ---- Exp9 ----
def regenerate_exp9() -> None:
    # Use the latest reproducible Exp9 setting used in the paper text:
    # baseline_balanced vs set_aware (v3g), both under meta clean-val + alpha=0.5.
    results_dir = ROOT / "exp9_cifar10_setaware"
    baseline_dir = ROOT / "exp9_cifar10_setaware" / "results_meta_balance_alpha05_baseline_balanced_train_holdout_cleanval"
    setaware_dir = ROOT / "exp9_cifar10_setaware" / "results_meta_balance_alpha05_setaware_tuned_v3g"
    pointwise_dir = ROOT / "exp9_cifar10_setaware" / "results_meta_balance_alpha05_baseline_score_topk_train_holdout_cleanval"
    diversity_dir = ROOT / "exp9_cifar10_setaware" / "results_meta_balance_alpha05_diversity_gpu"
    baseline_method = "baseline_balanced"
    setaware_method = "set_aware"
    pointwise_method = "baseline_score_topk"
    # Fallback to Total_results meta-balance dirs if paper paths are missing.
    if not baseline_dir.exists() or not list(baseline_dir.glob("exp9_seed*_merged.csv")):
        fallback_base = ROOT / "Total_results" / "Tables" / "exp9_cifar10_setaware"
        fallback_baseline = fallback_base / "results_meta_balance_alpha05_baseline_balanced_train_holdout_cleanval"
        fallback_setaware = fallback_base / "results_meta_balance_alpha05_setaware_tuned_v3g"
        fallback_pointwise = fallback_base / "results_meta_balance_alpha05_baseline_score_topk_train_holdout_cleanval"
        fallback_diversity_gpu = fallback_base / "results_meta_balance_alpha05_diversity_gpu"
        fallback_diversity = fallback_base / "results_meta_balance_alpha05_diversity"
        if fallback_baseline.exists() and list(fallback_baseline.glob("exp9_seed*_merged.csv")):
            baseline_dir = fallback_baseline
            setaware_dir = fallback_setaware if fallback_setaware.exists() else fallback_baseline
            pointwise_dir = fallback_pointwise if fallback_pointwise.exists() else fallback_baseline
            if fallback_diversity_gpu.exists():
                diversity_dir = fallback_diversity_gpu
            else:
                diversity_dir = fallback_diversity if fallback_diversity.exists() else baseline_dir
        else:
            fallback = fallback_base / "results"
            if fallback.exists():
                baseline_dir = fallback
                setaware_dir = fallback
                baseline_method = "baseline"
                setaware_method = "set_aware"
                pointwise_dir = fallback
                pointwise_method = "baseline_score_topk"
                diversity_dir = fallback

    merged_baseline = plot_exp9_times.load_merged_results(baseline_dir, method=baseline_method)
    merged_setaware = plot_exp9_times.load_merged_results(setaware_dir, method=setaware_method)
    merged_pointwise = plot_exp9_times.load_merged_results(pointwise_dir, method=pointwise_method)
    merged_kcenter = plot_exp9_times.load_merged_results(diversity_dir, method="k_center")
    merged_dpp = plot_exp9_times.load_merged_results(diversity_dir, method="dpp")
    if merged_baseline and merged_setaware:
        agg_base = plot_exp9_times.aggregate_metrics(merged_baseline)
        agg_set = plot_exp9_times.aggregate_metrics(merged_setaware)
        plot_exp9_times.plot_curves(agg_base, agg_set, FIG_DIR / "exp9_cifar10_combo.png")
        extra_methods = []
        if merged_kcenter:
            extra_methods.append(("k-Center", merged_kcenter, "k_center"))
        if merged_dpp:
            extra_methods.append(("DPP", merged_dpp, "dpp"))
        # Pseudo-label histograms at Gen 1 and Gen 5
        plot_exp9_times.plot_pseudo_hist_generation(
            merged_baseline,
            merged_setaware,
            generation=1,
            out_path=FIG_DIR / "exp9_pseudo_gen1.png",
            extra_methods=extra_methods,
        )
        plot_exp9_times.plot_pseudo_hist_generation(
            merged_baseline,
            merged_setaware,
            generation=5,
            out_path=FIG_DIR / "exp9_pseudo_gen5_only.png",
            extra_methods=extra_methods,
        )
        use_kcenter = bool(merged_kcenter)
        diversity_source = merged_kcenter if use_kcenter else merged_dpp
        diversity_label = "k-Center (diversity-only)" if use_kcenter else "DPP (diversity-only)"
        diversity_color = plot_exp9_times.COLORS["k_center"] if use_kcenter else plot_exp9_times.COLORS["dpp"]
        pointwise_source = merged_baseline if merged_baseline else merged_pointwise
        if pointwise_source and diversity_source:
            plot_exp9_times.plot_worstclass_hist_small(
                pointwise_source,
                merged_setaware,
                diversity_source,
                generation=5,
                out_path=FIG_DIR / "exp9_cifar10_hist_small.png",
                diversity_label=diversity_label,
                diversity_color=diversity_color,
            )
        # Compose vertical stack: Gen1 (top) and Gen5 (bottom)
        img1 = FIG_DIR / "exp9_pseudo_gen1.png"
        img5 = FIG_DIR / "exp9_pseudo_gen5_only.png"
        if img1.exists() and img5.exists():
            top = Image.open(img1)
            bottom = Image.open(img5)
            target_width = max(top.width, bottom.width)

            def resize_width(img: Image.Image) -> Image.Image:
                if img.width == target_width:
                    return img
                ratio = target_width / img.width
                return img.resize((target_width, int(img.height * ratio)), Image.LANCZOS)

            top = resize_width(top)
            bottom = resize_width(bottom)
            sep = 30
            total_height = top.height + sep + bottom.height
            canvas = Image.new("RGB", (target_width, total_height), color=(255, 255, 255))
            canvas.paste(top, (0, 0))
            canvas.paste(bottom, (0, top.height + sep))
            canvas.save(FIG_DIR / "exp9_pseudo_gen1_final.png")
    else:
        copy_if_exists(results_dir / "exp9_plots_times.png", FIG_DIR / "exp9_cifar10_combo.png")


def regenerate_exp11() -> None:
    # Prefer streaming GPT-2 results used in the paper text; fall back to dphi1 if missing.
    results_root = ROOT / "Total_results" / "Tables" / "exp11_gpt2_model" / "Results_streaming"
    if not (results_root.exists() and list(results_root.glob("*/*metrics_diversity_ppl.json"))):
        results_root = ROOT / "exp11_gpt2_model" / "Results" / "dphi1"
        alt_root = ROOT / "Total_results" / "Tables" / "exp11_gpt2_model" / "Results" / "dphi1"
        if alt_root.exists() and list(alt_root.glob("*/*metrics_diversity_ppl.json")):
            results_root = alt_root
    df = load_exp11_results(results_root)
    df_mean = aggregate_exp11_mean(df)
    plot_phase_portrait(df_mean, FIG_DIR / "exp11_phase_portrait.png")
    # Dual-axis evolution plot
    plot_dual_axis(df_mean, FIG_DIR / "exp11_dual_axis.png")
    # Recursive lines
    plot_recursive_lines(df_mean, FIG_DIR / "exp11_gpt2_recursive.png")


def compose_exp9_exp11() -> None:
    """Compose exp9 stacked pseudo hists (Gen1+Gen5) and exp11 phase portrait side by side."""
    src_exp9 = FIG_DIR / "exp9_pseudo_gen1_final.png"
    src_exp11 = FIG_DIR / "exp11_phase_portrait.png"
    if not (src_exp9.exists() and src_exp11.exists()):
        return
    img9 = Image.open(src_exp9)
    img11 = Image.open(src_exp11)

    # Equal heights; preserve aspect ratios; simple left/right.
    target_height = max(img9.height, img11.height)

    def resize_height(img: Image.Image) -> Image.Image:
        if img.height == target_height:
            return img
        ratio = target_height / img.height
        return img.resize((int(img.width * ratio), target_height), Image.LANCZOS)

    img9 = resize_height(img9)
    img11 = resize_height(img11)
    sep = 40
    total_width = img9.width + sep + img11.width
    canvas = Image.new("RGB", (total_width, target_height), color=(255, 255, 255))
    canvas.paste(img9, (0, 0))
    canvas.paste(img11, (img9.width + sep, 0))
    out_path = FIG_DIR / "exp9_11_enhence.png"
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate paper figures.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[f"exp{i}" for i in range(1, 12)],
        help="Only regenerate the specified experiments (e.g., --only exp3 exp7).",
    )
    parser.add_argument(
        "--copy",
        nargs=2,
        metavar=("SRC", "DEST_NAME"),
        action="append",
        help="Copy a single figure from SRC (absolute or ROOT-relative) to Figures/DEST_NAME.",
    )
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tasks = {
        "exp1": regenerate_exp1,
        "exp2": regenerate_exp2,
        "exp3": regenerate_exp3,
        "exp4": regenerate_exp4,
        "exp5": regenerate_exp5,
        "exp6": regenerate_exp6,
        "exp7": regenerate_exp7,
        "exp8": regenerate_exp8,
        "exp9": regenerate_exp9,
        "exp10": lambda: None,  # no figures to regenerate here
        "exp11": regenerate_exp11,
    }

    selected = args.only if args.only else [k for k in tasks if k != "exp10"]
    for key in selected:
        tasks[key]()

    # Compose exp9+exp11 if possible
    compose_exp9_exp11()

    if args.copy:
        for src, dest_name in args.copy:
            src_path = pathlib.Path(src)
            if not src_path.is_absolute():
                src_path = ROOT / src_path
            copy_if_exists(src_path, FIG_DIR / dest_name)


if __name__ == "__main__":
    main()
