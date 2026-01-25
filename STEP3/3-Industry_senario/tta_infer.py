#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简易 TTA 推理脚本：对同一张图做原图 + 水平翻转的预测，按 IoU 匹配后平均置信度。
用法示例：
    python tta_infer.py --model anchor_mt.pt --source /path/to/img_or_dir --imgsz 640
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageOps
import torch.nn as nn
from ultralytics import YOLO


ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class DetBox:
    xyxy: np.ndarray  # [x1, y1, x2, y2]
    conf: float
    cls_id: int


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for two [4] xyxy arrays."""
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, (xa2 - xa1)) * max(0.0, (ya2 - ya1))
    area_b = max(0.0, (xb2 - xb1)) * max(0.0, (yb2 - yb1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def map_boxes(result, img_size: Tuple[int, int], augment: str) -> List[DetBox]:
    """
    将 YOLO 结果映射回原图坐标。
    augment: "orig" 或 "flip"。
    """
    w, h = img_size
    boxes = []
    xyxy = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.zeros((0, 4), dtype=np.float32)
    confs = result.boxes.conf.cpu().numpy() if len(result.boxes) else np.zeros((0,), dtype=np.float32)
    clss = result.boxes.cls.cpu().numpy().astype(int) if len(result.boxes) else np.zeros((0,), dtype=int)
    for coords, conf, cls_id in zip(xyxy, confs, clss):
        if augment == "flip":
            x1, y1, x2, y2 = coords
            coords = np.array([w - x2, y1, w - x1, y2], dtype=np.float32)
        boxes.append(DetBox(xyxy=coords, conf=float(conf), cls_id=int(cls_id)))
    return boxes


def average_boxes(boxes: List[DetBox], iou_thr: float = 0.6) -> List[DetBox]:
    """按 IoU 和类别聚类后，对置信度和坐标做平均。"""
    clusters: List[List[DetBox]] = []
    for box in boxes:
        placed = False
        for cluster in clusters:
            if cluster and cluster[0].cls_id == box.cls_id and box_iou(cluster[0].xyxy, box.xyxy) >= iou_thr:
                cluster.append(box)
                placed = True
                break
        if not placed:
            clusters.append([box])

    merged: List[DetBox] = []
    for cluster in clusters:
        if not cluster:
            continue
        coords = np.stack([b.xyxy for b in cluster], axis=0).mean(axis=0)
        conf = float(np.mean([b.conf for b in cluster]))
        merged.append(DetBox(xyxy=coords, conf=conf, cls_id=cluster[0].cls_id))
    return merged


def tta_predict(model: YOLO, image: Image.Image, imgsz: int, conf_thr: float, iou_thr: float) -> List[DetBox]:
    """原图 + 水平翻转 TTA，平均置信度。"""
    w, h = image.size
    variants = [("orig", image), ("flip", ImageOps.mirror(image))]
    all_boxes: List[DetBox] = []
    for tag, img in variants:
        res = model.predict(img, imgsz=imgsz, conf=conf_thr, verbose=False)[0]
        mapped = map_boxes(res, (w, h), augment=tag)
        all_boxes.extend(mapped)
    return average_boxes(all_boxes, iou_thr=iou_thr)


def iter_images(source: Path) -> Iterable[Path]:
    if source.is_file():
        yield source
    else:
        for p in sorted(source.rglob("*")):
            if p.suffix.lower() in ALLOWED_EXTS:
                yield p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO TTA inference (orig + hflip, avg confidence)")
    p.add_argument("--model", type=Path, required=True, help="YOLO 权重，例如 anchor_mt.pt / best.pt")
    p.add_argument("--source", type=Path, required=True, help="图片或目录")
    p.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    p.add_argument("--conf-thr", type=float, default=0.25, help="基础置信度阈值")
    p.add_argument("--iou-merge", type=float, default=0.6, help="同类聚类 IoU 阈值，用于平均")
    p.add_argument("--save-json", action="store_true", help="保存每张图的 TTA 结果为 json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))
    for m in model.model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()  # 冻结 BN，避免统计被单批样本污染

    for img_path in iter_images(args.source):
        img = Image.open(img_path).convert("RGB")
        merged = tta_predict(model, img, imgsz=args.imgsz, conf_thr=args.conf_thr, iou_thr=args.iou_merge)

        print(f"\n{img_path}:")
        if not merged:
            print("  无检测")
            continue
        for b in merged:
            x1, y1, x2, y2 = b.xyxy
            print(f"  cls={b.cls_id} conf={b.conf:.3f} xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

        if args.save_json:
            out = [
                {"cls": b.cls_id, "conf": b.conf, "xyxy": [float(v) for v in b.xyxy]}
                for b in merged
            ]
            out_path = img_path.with_suffix(".tta.json")
            out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
