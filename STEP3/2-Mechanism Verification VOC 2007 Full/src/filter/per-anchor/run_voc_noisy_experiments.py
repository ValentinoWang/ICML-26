#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VOC2007 噪声集对比实验入口（支持 per-sample 与 per-anchor 两种噪声）：
- Baseline: 直接在噪声集上训练 YOLO（无 g_phi）
- Ours: 在噪声集上训练，加载规则蒸馏得到的 g_phi，启用过滤

结果目录：
- Baseline: Results/Bias_only/<scenario>/seed_<seed>/results/...
- Ours:     Results/Bias+Filter_rule/<scenario>/seed_<seed>/results/...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

from ICML.core.yolo_bias_finetune.anchor import AnchorModel
from ICML.core.yolo_bias_finetune.bias_trainer import BiasDetectionTrainer

# per-anchor 训练器（逐 Anchor 加权）
PER_ANCHOR_TRAINER_PATH = Path(__file__).resolve().parent / "per_anchor_trainer.py"
if PER_ANCHOR_TRAINER_PATH.exists():
    sys.path.insert(0, str(PER_ANCHOR_TRAINER_PATH.parent))
    try:
        from per_anchor_trainer import PerAnchorBiasDetectionTrainer
    except Exception:
        PerAnchorBiasDetectionTrainer = None
else:
    PerAnchorBiasDetectionTrainer = None


DEFAULT_ANCHOR_DIR = (
    Path("/root/autodl-tmp/ICML/2-Mechanism Verification VOC 2007 Full/Results/Anchor/voc07_clean")
)
DEFAULT_DISTILL_DIR_MAP = {
    "per-sample": Path(
        "/root/autodl-tmp/ICML/2-Mechanism Verification VOC 2007 Full/Results/distill_rule_per-sample/voc07_noisy"
    ),
    "per-anchor": Path(
        "/root/autodl-tmp/ICML/2-Mechanism Verification VOC 2007 Full/Results/distill_rule_per-anchor/voc07_noisy"
    ),
}
DEFAULT_DATA_MAP = {
    "per-sample": Path("/root/autodl-tmp/dataset/voc07_noisy_per-sample/voc07_noisy.yaml"),
    "per-anchor": Path("/root/autodl-tmp/dataset/voc07_noisy_per-anchor/voc07_noisy.yaml"),
}


def build_overrides(
    data_yaml: Path,
    model_path: Path,
    project_dir: Path,
    epochs: int,
    batch: int,
    patience: int,
    lr0: float,
    optimizer: str,
    imgsz: int,
    seed: int,
    device: str,
    fraction: float,
) -> Dict[str, Any]:
    return {
        "task": "detect",
        "model": str(model_path),
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch,
        "workers": 4,
        "patience": patience,
        "lr0": lr0,
        "optimizer": optimizer,
        "project": str(project_dir),
        "name": "results",
        "device": device,
        "imgsz": imgsz,
        "save": True,
        "save_period": -1,
        "verbose": True,
        "plots": True,
        "amp": True,
        "cache": False,
        "resume": False,
        "seed": seed,
        "deterministic": True,
        "fraction": fraction,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VOC2007 noisy baseline vs g_phi (full) experiments")
    parser.add_argument(
        "--noise-mode",
        type=str,
        choices=["per-sample", "per-anchor"],
        default="per-sample",
        help="噪声模式：per-sample=整图毒，per-anchor=混合/实例噪声",
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
        help="干净 VOC 训练得到的 anchor_voc_<model>.pt（默认自动在 Results/Anchor/voc07_clean 下取最新）",
    )
    parser.add_argument(
        "--mlp-filter",
        type=Path,
        default=None,
        help="规则蒸馏得到的 g_phi 权重路径（不填则按 noise-mode 自动选择最新）",
    )
    parser.add_argument(
        "--mlp-filter-dir",
        type=Path,
        default=None,
        help="指定查找 mlp_filter 的目录（覆盖 noise-mode 默认目录）",
    )
    parser.add_argument("--epochs", type=int, default=10000, help="训练 epochs")
    parser.add_argument("--batch", type=int, default=128, help="batch size（默认 128）")
    parser.add_argument("--patience", type=int, default=20, help="early stop patience")
    parser.add_argument("--lr0", type=float, default=0.001, help="初始学习率")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="优化器")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--seed", type=int, default=1088, help="随机种子")
    parser.add_argument("--device", type=str, default="0", help="YOLO device，如 '0' 或 'cpu'")
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="使用数据集的比例做快速冒烟（如 0.01 表示 1% 数据）",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="per-anchor 过滤启用前的热身轮数（权重全为 1，不做过滤）",
    )
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="如需同时跑 Baseline 则加此参数；默认只跑 g_phi 版本",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/root/autodl-tmp/ICML/2-Mechanism Verification VOC 2007 Full/Results"),
        help="结果根目录（会创建 Bias_only / Bias+Filter_rule 子目录）",
    )
    return parser.parse_args()


def train_once(
    data_yaml: Path,
    anchor_path: Path,
    mlp_filter_path: Path | None,
    use_filter: bool,
    overrides: Dict[str, Any],
    per_anchor: bool = False,
    per_anchor_kwargs: Dict[str, Any] | None = None,
) -> str:
    anchor = AnchorModel(anchor_path)
    if use_filter and per_anchor and PerAnchorBiasDetectionTrainer is not None:
        trainer = PerAnchorBiasDetectionTrainer(
            anchor_model=anchor,
            lambda_bias=0.0,
            mlpfilter_path=mlp_filter_path,
            **(per_anchor_kwargs or {}),
            overrides=overrides,
        )
    else:
        trainer = BiasDetectionTrainer(
            anchor_model=anchor,
            lambda_bias=0.0,  # 不加 L_bias，只用 g_phi 重加权
            use_sample_filter=use_filter,
            filter_mode="mlp",
            mlpfilter_init_path=mlp_filter_path if use_filter else None,
            overrides=overrides,
        )
    print(f"[trainer] Using {trainer.__class__.__name__}")
    trainer.train()
    return trainer.best


def main() -> None:
    args = parse_args()
    data_yaml = args.data or DEFAULT_DATA_MAP[args.noise_mode]
    if not data_yaml.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_yaml}")
    # 若无可用 GPU，自动回退到 CPU
    try:
        import torch

        if not torch.cuda.is_available() and args.device != "cpu":
            args.device = "cpu"
    except Exception:
        args.device = "cpu"

    anchor_path = args.anchor
    if anchor_path is None:
        candidates = sorted(
            DEFAULT_ANCHOR_DIR.glob("anchor_voc_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"未找到 anchor 模型，请指定 --anchor 或在 {DEFAULT_ANCHOR_DIR} 下放置 anchor_voc_*.pt"
            )
        anchor_path = candidates[0]
    if not anchor_path.exists():
        raise FileNotFoundError(f"anchor_voc.pt 不存在: {anchor_path}")

    distill_dir = args.mlp_filter_dir or DEFAULT_DISTILL_DIR_MAP[args.noise_mode]
    mlp_filter_path = args.mlp_filter
    if mlp_filter_path is None:
        candidates = sorted(
            distill_dir.glob("mlp_filter_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"未找到 mlp_filter 模型，请指定 --mlp-filter 或在 {distill_dir} 下放置 mlp_filter_*.pt"
            )
        mlp_filter_path = candidates[0]
    if mlp_filter_path is not None and not mlp_filter_path.exists():
        raise FileNotFoundError(f"mlp_filter_voc.pt 不存在: {mlp_filter_path}")

    results_root = args.results_root
    results_root.mkdir(parents=True, exist_ok=True)
    scenario = f"voc07_noisy_{args.noise_mode}"

    print(f"🎯 VOC noisy experiment | noise_mode={args.noise_mode}")
    print(f"   data: {data_yaml}")
    print(f"   anchor: {anchor_path}")
    print(f"   mlp_filter_dir: {distill_dir}")
    print(f"   mlp_filter: {mlp_filter_path}")
    print(f"   results_root: {results_root}")
    print(f"   warmup_epochs (per-anchor): {args.warmup_epochs}")

    if args.run_baseline:
        base_root = results_root / "Bias_only" / scenario / f"seed_{args.seed}"
        base_root.mkdir(parents=True, exist_ok=True)
        base_overrides = build_overrides(
            data_yaml=data_yaml,
            model_path=anchor_path,
            project_dir=base_root,
            epochs=args.epochs,
            batch=args.batch,
            patience=args.patience,
            lr0=args.lr0,
            optimizer=args.optimizer,
            imgsz=args.imgsz,
            seed=args.seed,
            device=args.device,
            fraction=args.fraction,
        )
        print(f"\n=== Baseline: YOLO on {scenario} (no g_phi) ===")
        best_baseline = train_once(
            data_yaml=data_yaml,
            anchor_path=anchor_path,
            mlp_filter_path=None,
            use_filter=False,
            overrides=base_overrides,
        )
        print(f"Baseline best checkpoint: {best_baseline}")
    else:
        print("\n=== 仅运行 g_phi 版本（Baseline 已跳过，若需 Baseline 请加 --run-baseline） ===")

    # Ours: load mlp_filter_voc.pt and enable g_phi
    ours_root = results_root / "Bias+Filter_rule" / scenario / f"seed_{args.seed}"
    ours_root.mkdir(parents=True, exist_ok=True)
    ours_overrides = build_overrides(
        data_yaml=data_yaml,
        model_path=anchor_path,
        project_dir=ours_root,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        lr0=args.lr0,
        optimizer=args.optimizer,
        imgsz=args.imgsz,
            seed=args.seed,
            device=args.device,
            fraction=args.fraction,
        )
    per_anchor_mode = args.noise_mode == "per-anchor"
    per_anchor_kwargs = {
        "loss_thres": 0.5,  # 放宽阈值，减少过度判坏
        "bad_weight": 0.0,
        "topk": 500,
        "pos_scale": 2.0,  # 强化好样本权重，避免全体被压到下界
        "neg_scale": 1.0,
        "warmup_epochs": args.warmup_epochs,
    }
    print(f"\n=== Ours: YOLO on {scenario} with g_phi ({mlp_filter_path.name}) ===")
    best_ours = train_once(
        data_yaml=data_yaml,
        anchor_path=anchor_path,
        mlp_filter_path=mlp_filter_path,
        use_filter=True,
        overrides=ours_overrides,
        per_anchor=per_anchor_mode,
        per_anchor_kwargs=per_anchor_kwargs if per_anchor_mode else None,
    )
    print(f"Ours best checkpoint: {best_ours}")


if __name__ == "__main__":
    main()
