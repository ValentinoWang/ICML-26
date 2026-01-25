#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VOC2007 noisy 集上运行 YOLO + L_bias_only（无 g_phi）的批量实验脚本。
默认与 run_three_seeds.sh 一致的超参：epochs=10000, batch=128, imgsz=640, AdamW, lr0=0.001, patience=20, workers=4, device=0。
Seeds 默认使用 [1088, 2195, 4960]。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ICML.core.yolo_bias_finetune.anchor import AnchorModel
from ICML.core.yolo_bias_finetune.bias_trainer import BiasDetectionTrainer


# Anchor 固定放在公共 Anchor 路径下，不随 results_root 变化
DEFAULT_ANCHOR_DIR = Path(
    "/root/autodl-tmp/ICML/2-Mechanism Verification VOC 2007 Full/Results/Anchor/voc07_clean"
)
DEFAULT_RESULTS_BASE = Path(
    "/root/autodl-tmp/ICML/2-Mechanism Verification VOC 2007 Full/Results/L_Bias_only"
)
DEFAULT_DATA_MAP = {
    "per-sample": Path("/root/autodl-tmp/dataset/voc07_noisy_per-sample/voc07_noisy.yaml"),
    "per-anchor": Path("/root/autodl-tmp/dataset/voc07_noisy_per-anchor/voc07_noisy.yaml"),
}


def parse_int_list(values: Iterable[str]) -> List[int]:
    return [int(v) for v in values]


def build_overrides(
    data_yaml: Path,
    model_path: Path,
    project_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    """构造传给 BiasDetectionTrainer 的 overrides 字典。"""
    return {
        "task": "detect",
        "model": str(model_path),
        "data": str(data_yaml),
        "epochs": 10000,
        "batch": 128,
        "workers": 4,
        "patience": 20,
        "lr0": 0.001,
        "optimizer": "AdamW",
        "project": str(project_dir),
        "name": "results",
        "device": "0",
        "imgsz": 640,
        "save": True,
        "save_period": -1,
        "verbose": True,
        "plots": True,
        "amp": True,
        "cache": False,
        "resume": False,
        "seed": seed,
        "deterministic": True,
    }


def auto_select_anchor(anchor_arg: Path | None) -> Path:
    """若未显式指定 anchor，自动在 Anchor/voc07_clean 下取最新 anchor_voc_*.pt。"""
    if anchor_arg is not None:
        return anchor_arg
    candidates = sorted(
        DEFAULT_ANCHOR_DIR.glob("anchor_voc_*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"未找到 anchor 模型，请在 {DEFAULT_ANCHOR_DIR} 下放置 anchor_voc_*.pt 或使用 --anchor 指定"
        )
    return candidates[0]


def run_once(
    seed: int,
    data_yaml: Path,
    anchor_path: Path,
    lambda_bias: float,
    results_root: Path,
) -> None:
    """单个 seed 上运行 YOLO + L_bias，并保存 bias 评估结果。"""
    seed_root = results_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)

    overrides = build_overrides(
        data_yaml=data_yaml,
        model_path=anchor_path,
        project_dir=seed_root,
        seed=seed,
    )

    anchor = AnchorModel(anchor_path)
    trainer = BiasDetectionTrainer(
        anchor_model=anchor,
        lambda_bias=lambda_bias,
        use_sample_filter=False,
        filter_mode="mlp",
        mlpfilter_init_path=None,
        overrides=overrides,
    )

    print(f"\n=== VOC07 noisy | YOLO + L_bias | seed {seed} | λ={lambda_bias:g} ===")
    trainer.train()

    best_ckpt = Path(trainer.best)
    if best_ckpt.exists():
        model = trainer.model
        inner_model = model.inner if hasattr(model, "inner") else model
        inner_model.to(trainer.device)
        bias_loss = anchor.compute_bias_loss(inner_model)
        print(f"L_bias(θ_new, θ_good) = {bias_loss.item():.4f}")

        bias_info = {
            "seed": seed,
            "lambda_bias": lambda_bias,
            "anchor": str(anchor_path),
            "bias_loss": float(bias_loss.item()),
            "best_checkpoint": str(best_ckpt),
        }
        out_file = seed_root / "bias_evaluation.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(bias_info, f, indent=2, ensure_ascii=False)
        print(f"已保存 L_bias 评估结果到: {out_file}")
    else:
        print(f"未找到 best.pt，跳过 L_bias 评估: {best_ckpt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 voc07_noisy 上运行 YOLO + L_bias（无 g_phi）"
    )
    parser.add_argument(
        "--noise-mode",
        type=str,
        choices=["per-sample", "per-anchor"],
        default="per-sample",
        help="噪声模式：per-sample=整图中毒，per-anchor=混合/实例级噪声",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="voc07_noisy.yaml 路径（不填则按 noise-mode 使用默认路径）",
    )
    parser.add_argument(
        "--anchor",
        type=Path,
        default=None,
        help="θ_good（anchor_voc_*.pt）路径；默认自动在 Anchor/voc07_clean 下取最新",
    )
    parser.add_argument(
        "--lambda-bias",
        type=float,
        default=1e-4,
        help="L_bias 的损失权重 λ_bias",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["1088"],
        help="随机种子列表（默认 1088）",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_BASE,
        help="结果根目录（会在其下创建 noise-mode/lambda/seed_* 子目录，默认 Results/L_Bias_only）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    data_yaml = args.data or DEFAULT_DATA_MAP[args.noise_mode]
    if not data_yaml.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_yaml}")

    anchor_path = auto_select_anchor(args.anchor)
    if not anchor_path.exists():
        raise FileNotFoundError(f"anchor 模型不存在: {anchor_path}")

    results_root = args.results_root / args.noise_mode
    if args.lambda_bias == 0:
        lambda_dir = "lambda_0"
    else:
        lambda_dir = f"lambda_{args.lambda_bias:g}"
    results_root = results_root / lambda_dir
    results_root.mkdir(parents=True, exist_ok=True)

    print("🎯 VOC07 noisy YOLO + L_bias 实验")
    print(f"   Seeds: {seeds}")
    print(f"   λ_bias: {args.lambda_bias}")
    print(f"   noise_mode: {args.noise_mode}")
    print(f"   anchor: {anchor_path}")
    print(f"   data: {data_yaml}")
    print(f"   results root: {results_root}")

    for seed in seeds:
        run_once(
            seed=seed,
            data_yaml=data_yaml,
            anchor_path=anchor_path,
            lambda_bias=args.lambda_bias,
            results_root=results_root,
        )

    print("✅ 所有 VOC07 noisy YOLO + L_bias 实验完成")


if __name__ == "__main__":
    main()
