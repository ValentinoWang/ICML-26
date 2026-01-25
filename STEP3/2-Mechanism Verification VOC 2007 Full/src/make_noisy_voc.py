#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
严格按“干净 Test / 干净 Val / 噪声 Train”生成 VOC2007 YOLO 数据：
1) Global Test: VOC2007 Test（4952），干净不动；
2) Validation: 从 Trainval 抽 10%-20%（可调），干净不动；
3) Noisy Train: 其余 Trainval，按图像级抽样噪声：
   - 模式 A（image）：随机 30%~40% 图像“全图中毒”，清空原有 GT，在空白区域随机画 3~5 个假框并赋随机类别；
   - 模式 B（box）：随机 30%~40% 图像，仅篡改该图内部分框（默认 50%），其余框保持干净。

输出结构：
  <output_root>/
    images/{train,val,test}  (软链接到原图)
    labels/{train,val,test}  (train 含噪声，val/test 干净)
    voc07_noisy.yaml         (train/val/test 路径)
    noise_meta.txt           (记录噪声比例/数量/划分)
"""

from __future__ import annotations

import argparse
import os
import random
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import yaml


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def extract_if_needed(tar_path: Path, dst_root: Path) -> Path:
    """解压 tar 到 dst_root/VOC2007；若已存在 JPEGImages 则跳过。"""
    base_name = tar_path.name
    if base_name.endswith(".tar.gz"):
        base_name = base_name[:-7]
    elif base_name.endswith(".tgz"):
        base_name = base_name[:-4]
    dst_dir = dst_root / base_name  # VOC2007
    if (dst_dir / "JPEGImages").exists():
        return dst_dir
    print(f"[INFO] Extracting {tar_path} to {dst_root} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dst_root)
    return dst_dir


def parse_annotation(xml_path: Path) -> Tuple[int, int, List[dict]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    objs = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        objs.append({"name": name, "bbox": [xmin, ymin, xmax, ymax]})
    return w, h, objs


def iou(box1: List[float], box2: List[float]) -> float:
    xa1, ya1, xa2, ya2 = box1
    xb1, yb1, xb2, yb2 = box2
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area + 1e-9
    return inter_area / union


def sample_background_boxes(
    w: int,
    h: int,
    existing: List[List[float]],
    rng: random.Random,
    num_range: Tuple[int, int] = (1, 3),
    max_tries: int = 50,
) -> List[List[float]]:
    boxes = []
    num_boxes = rng.randint(num_range[0], num_range[1])

    for _ in range(num_boxes):
        candidate = None
        for _ in range(max_tries):
            bw = rng.uniform(0.05, 0.3) * w
            bh = rng.uniform(0.05, 0.3) * h
            cx = rng.uniform(bw / 2.0, w - bw / 2.0)
            cy = rng.uniform(bh / 2.0, h - bh / 2.0)
            x1 = max(0.0, cx - bw / 2.0)
            y1 = max(0.0, cy - bh / 2.0)
            x2 = min(float(w - 1), cx + bw / 2.0)
            y2 = min(float(h - 1), cy + bh / 2.0)
            candidate = [x1, y1, x2, y2]
            max_iou = 0.0
            for b in existing + boxes:
                max_iou = max(max_iou, iou(candidate, b))
            if max_iou < 0.05:
                break
        if candidate is not None:
            boxes.append(candidate)
    return boxes


def apply_background_poison(
    w: int,
    h: int,
    rng: random.Random,
    num_range: Tuple[int, int] = (3, 5),
) -> List[dict]:
    """生成纯假框，清空真实 GT。"""
    poison_boxes = sample_background_boxes(w, h, existing=[], rng=rng, num_range=num_range)
    noisy: List[dict] = []
    for b in poison_boxes:
        cls = rng.choice(VOC_CLASSES)
        noisy.append({"name": cls, "bbox": b})
    return noisy


def apply_instance_noise(
    objs: List[dict],
    w: int,
    h: int,
    rng: random.Random,
    frac: float = 0.5,
) -> List[dict]:
    """
    实例级噪声：仅篡改部分框（默认 50%），其余保持干净。
    改法：随机改类别，并对 bbox 做小幅平移扰动。
    """
    if not objs:
        return []
    noisy = list(objs)
    num_to_corrupt = max(1, int(len(noisy) * frac))
    idxs = rng.sample(range(len(noisy)), num_to_corrupt)
    for idx in idxs:
        obj = noisy[idx]
        # 改类别
        choices = [c for c in VOC_CLASSES if c != obj["name"]]
        obj["name"] = rng.choice(choices)
        # 小幅平移
        x1, y1, x2, y2 = obj["bbox"]
        dx = rng.uniform(-0.05, 0.05) * w
        dy = rng.uniform(-0.05, 0.05) * h
        nx1 = max(0.0, min(w - 1.0, x1 + dx))
        ny1 = max(0.0, min(h - 1.0, y1 + dy))
        nx2 = max(0.0, min(w - 1.0, x2 + dx))
        ny2 = max(0.0, min(h - 1.0, y2 + dy))
        if nx2 <= nx1 + 1.0:
            nx2 = min(w - 1.0, nx1 + 2.0)
        if ny2 <= ny1 + 1.0:
            ny2 = min(h - 1.0, ny1 + 2.0)
        obj["bbox"] = [nx1, ny1, nx2, ny2]
        noisy[idx] = obj
    return noisy


def to_yolo_line(name: str, bbox: List[float], w: int, h: int) -> str:
    xc = (bbox[0] + bbox[2]) / 2.0 / w
    yc = (bbox[1] + bbox[3]) / 2.0 / h
    bw = (bbox[2] - bbox[0]) / w
    bh = (bbox[3] - bbox[1]) / h
    cls_idx = VOC_CLASSES.index(name)
    return f"{cls_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def make_symlink(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)


def write_label(lines: List[str], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines))


def build_yaml(root: Path, yaml_path: Path) -> None:
    data = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(VOC_CLASSES)},
    }
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    print(f"[INFO] YAML written to {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make noisy VOC2007 YOLO dataset with clean test/val and noisy train only"
    )
    parser.add_argument(
        "--tar-path",
        type=Path,
        default=Path("/autodl-pub/data/VOCdevkit/VOC2007.tar.gz"),
        help="VOC2007.tar.gz 路径",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "voc07_noisy_strict",
        help="输出根目录（包含 images/labels/yaml）",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("/root/autodl-tmp/dataset"),
        help="VOC2007 解压目录（可写）",
    )
    parser.add_argument("--noise-ratio", type=float, default=0.35, help="训练集噪声比例（建议 0.3~0.4）")
    parser.add_argument(
        "--noise-mode",
        type=str,
        default="image",
        choices=["image", "box"],
        help="噪声模式：image=整图中毒（清空 GT+假框），box=实例级篡改（部分框改错）",
    )
    parser.add_argument(
        "--box-noise-frac",
        type=float,
        default=0.5,
        help="box 模式下篡改的框比例",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="从 trainval 抽取的 clean val 比例（推荐 0.1~0.2）",
    )
    parser.add_argument("--seed", type=int, default=1088, help="随机种子")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    extract_dir = args.extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)
    voc_dir = extract_if_needed(args.tar_path, extract_dir)
    img_dir = voc_dir / "JPEGImages"
    ann_dir = voc_dir / "Annotations"
    trainval_file = voc_dir / "ImageSets" / "Main" / "trainval.txt"
    test_file = voc_dir / "ImageSets" / "Main" / "test.txt"
    if not trainval_file.exists() or not test_file.exists():
        raise FileNotFoundError("trainval.txt 或 test.txt 不存在")

    trainval_ids = [line.strip() for line in trainval_file.read_text().splitlines() if line.strip()]
    test_ids = [line.strip() for line in test_file.read_text().splitlines() if line.strip()]
    if not trainval_ids:
        raise RuntimeError("trainval.txt 为空")
    if not test_ids:
        raise RuntimeError("test.txt 为空")

    rng.shuffle(trainval_ids)
    val_size = max(1, int(len(trainval_ids) * args.val_ratio))
    val_ids = trainval_ids[:val_size]
    train_ids = trainval_ids[val_size:]

    # 噪声只加在 train_ids
    num_noise = int(len(train_ids) * args.noise_ratio)
    noise_ids = set(rng.sample(train_ids, num_noise)) if num_noise > 0 else set()

    out_root = args.output_root
    img_train_dir = out_root / "images" / "train"
    img_val_dir = out_root / "images" / "val"
    img_test_dir = out_root / "images" / "test"
    lbl_train_dir = out_root / "labels" / "train"
    lbl_val_dir = out_root / "labels" / "val"
    lbl_test_dir = out_root / "labels" / "test"

    def process_split(name: str, id_list: List[str], img_out_dir: Path, lbl_out_dir: Path, noisy: bool) -> None:
        print(f"[INFO] Processing {name}: {len(id_list)} images (noisy={noisy})")
        for vid in id_list:
            xml_path = ann_dir / f"{vid}.xml"
            jpg_path = img_dir / f"{vid}.jpg"
            if not xml_path.exists() or not jpg_path.exists():
                continue
            w, h, objs = parse_annotation(xml_path)
            if noisy and vid in noise_ids:
                if args.noise_mode == "image":
                    objs = apply_background_poison(w, h, rng)
                else:  # box
                    objs = apply_instance_noise(objs, w, h, rng, frac=args.box_noise_frac)
            lines = [to_yolo_line(o["name"], o["bbox"], w, h) for o in objs]
            label_path = lbl_out_dir / f"{vid}.txt"
            write_label(lines, label_path)
            make_symlink(jpg_path, img_out_dir / f"{vid}.jpg")

    process_split("train", train_ids, img_train_dir, lbl_train_dir, noisy=True)
    process_split("val", val_ids, img_val_dir, lbl_val_dir, noisy=False)
    process_split("test", test_ids, img_test_dir, lbl_test_dir, noisy=False)

    yaml_path = out_root / "voc07_noisy.yaml"
    build_yaml(out_root, yaml_path)
    meta_path = out_root / "noise_meta.txt"
    meta = {
        "noise_ratio": args.noise_ratio,
        "noise_mode": args.noise_mode,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "num_noise_images": len(noise_ids),
        "train": len(train_ids),
        "val": len(val_ids),
        "test": len(test_ids),
    }
    if args.noise_mode == "image":
        meta["poison_boxes_per_image"] = "3-5"
    else:
        meta["box_noise_frac"] = args.box_noise_frac
    meta_path.write_text("\n".join(f"{k}={v}" for k, v in meta.items()))
    print(f"[INFO] Done. Noisy train / clean val+test at {out_root}")


if __name__ == "__main__":
    main()
