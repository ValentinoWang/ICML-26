#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
一键训练干净 VOC 锚点模型并输出 anchor_voc.pt，避免手动 cp。

流程：
- 调用 Ultralytics YOLO CLI 训练（默认 yolov8n，epochs=10000，patience=20，batch=64，imgsz=640）。
- 训练工程目录：<runs_root>/<run_name>，默认 runs_anchor/anchor_voc。
- 训练完成后自动将 best.pt 复制为 anchor_voc.pt（覆盖旧文件）。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train clean VOC anchor and export anchor_voc.pt")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/root/autodl-tmp/dataset/voc07_clean/voc07_noisy.yaml"),
        help="干净 VOC YOLO 格式数据的 yaml",
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO 基座模型（可改 yolov8s.pt 等）")
    parser.add_argument("--epochs", type=int, default=10000, help="训练 epoch 上限")
    parser.add_argument("--patience", type=int, default=20, help="早停 patience")
    parser.add_argument("--batch", type=int, default=128, help="batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="输入分辨率")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(__file__).resolve().parent / "Results" / "Anchor" / "runs_anchor",
        help="YOLO 训练的 project 目录（默认放在 Results/Anchor/runs_anchor 下）",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="anchor_voc",
        help="YOLO 训练 name（子目录）",
    )
    parser.add_argument(
        "--anchor-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Results" / "Anchor" / "voc07_clean",
        help="anchor 输出目录（文件名将自动带上模型名）",
    )
    parser.add_argument(
        "--anchor-out",
        type=Path,
        default=None,
        help="训练完成后 best.pt 拷贝到的输出路径；若不指定，则自动命名 anchor_voc_<model>.pt",
    )
    parser.add_argument("--seed", type=int, default=1088, help="随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.runs_root.mkdir(parents=True, exist_ok=True)
    model_tag = Path(args.model).stem
    anchor_out = args.anchor_out
    if anchor_out is None:
        anchor_out = args.anchor_dir / f"anchor_voc_{model_tag}.pt"

    print("🎯 训练干净 VOC 锚点模型 (YOLO)")
    print(f"   data: {args.data}")
    print(f"   model: {args.model}")
    print(f"   epochs: {args.epochs}, patience: {args.patience}, batch: {args.batch}, imgsz: {args.imgsz}")
    print(f"   project: {args.runs_root}, name: {args.run_name}")
    print(f"   anchor_out: {anchor_out}")

    env = os.environ.copy()
    # 强制指定 MKL/OMP，避免 oneMKL 加载 libtorch_cpu.so 冲突
    env["MKL_THREADING_LAYER"] = "GNU"
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")
    # 优先使用系统 libgomp，避免 iomp/gomp 冲突
    system_gomp = Path("/usr/lib/x86_64-linux-gnu/libgomp.so.1")
    if system_gomp.exists():
        env["LD_PRELOAD"] = str(system_gomp)
    # 避免 OpenMP 冲突直接报错
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    cmd = [
        "yolo",
        "detect",
        "train",
        f"data={args.data}",
        f"model={args.model}",
        f"epochs={args.epochs}",
        f"patience={args.patience}",
        f"batch={args.batch}",
        f"imgsz={args.imgsz}",
        f"project={args.runs_root}",
        f"name={args.run_name}",
        f"seed={args.seed}",
    ]

    subprocess.run(cmd, check=True, env=env)

    best_path = args.runs_root / args.run_name / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"未找到 best.pt: {best_path}")

    anchor_out.parent.mkdir(parents=True, exist_ok=True)
    # 清理旧锚点，避免混淆
    if anchor_out.exists():
        anchor_out.unlink()
    shutil.copy2(best_path, anchor_out)
    print(f"✅ anchor_voc 已生成: {anchor_out}")


if __name__ == "__main__":
    main()
