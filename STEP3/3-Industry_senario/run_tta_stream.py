#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
流式测试时自适应 (TTA) 脚本：
- 区分 Few-shot / Small / Medium / High 四种无标签池大小；
- 使用 YOLO 检测模型（默认 yolov8n.pt）做自适应；
- 适配器：Baseline 自训练、TENT、EATA-lite、Ours Bias-only；
- DataLoader shuffle=True，模拟随机到达；按 Epoch 定义：遍历一次池子即 1 Epoch。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Iterable, Any, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as TF

DATA_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DATA_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters import (  # noqa: E402
    SelfTrainingAdapter,
    TentAdapter,
    EataLiteAdapter,
    BiasOnlyAdapter,
    CoTTAAdapter,
    StabilizedCoTTAAdapter,
    StepResult,
)
from methods import METHOD_TO_DIR  # noqa: E402
from ultralytics import YOLO  # noqa: E402


@dataclass
class StreamScenario:
    name: str
    pool_file: Path
    pool_size: int
    batch_size: int
    epochs: int


SCENARIO_DIRS: Dict[str, List[str]] = {
    "few-shot": ["MT-tgt_few-shot_train", "MT-tgt_few-shot_val"],
    "small": ["MT-tgt_small_train", "MT-tgt_small_val"],
    "medium": ["MT-tgt_medium_train", "MT-tgt_medium_val"],
    "high": ["MT-tgt_high_train", "MT-tgt_high_val"],
}


SCENARIOS: Dict[str, StreamScenario] = {
    "few-shot": StreamScenario(
        name="few-shot",
        pool_file=DATA_ROOT / "pools" / "mt_few_shot.txt",
        pool_size=0,  # 全部 train+val
        batch_size=2,
        epochs=100,
    ),
    "small": StreamScenario(
        name="small",
        pool_file=DATA_ROOT / "pools" / "mt_small.txt",
        pool_size=0,  # 全部 train+val
        batch_size=4,
        epochs=100,
    ),
    "medium": StreamScenario(
        name="medium",
        pool_file=DATA_ROOT / "pools" / "mt_medium.txt",
        pool_size=0,  # 全部 train+val
        batch_size=8,
        epochs=40,
    ),
    "high": StreamScenario(
        name="high",
        pool_file=DATA_ROOT / "pools" / "mt_high.txt",
        pool_size=0,  # 0 表示使用全部 train+val
        batch_size=16,
        epochs=20,
    ),
}

GLOBAL_TEST_IMAGES = Path("/root/autodl-tmp/dataset/MT-tgt-split/MT-tgt_global_test/images")
GLOBAL_TEST_YAML = DATA_ROOT / "eval_yaml" / "mt-tgt-global-test.yaml"
SCENARIO_YAML: Dict[str, Path] = {name: GLOBAL_TEST_YAML for name in SCENARIOS.keys()}


def ensure_global_test_yaml(path: Path) -> None:
    """生成/覆盖一个仅指向 global_test 的统一 eval YAML。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            "# MT-tgt global test eval yaml (val/test -> global_test)",
            f"train: {GLOBAL_TEST_IMAGES}",
            f"val: {GLOBAL_TEST_IMAGES}",
            f"test: {GLOBAL_TEST_IMAGES}",
            "nc: 6",
            "names: ['chamfer', 'multifaceted', 'bump', 'impurity', 'crack', 'grind']",
            "scenario: global_test",
        ]
    )
    path.write_text(content + "\n")


def gather_scenario_images(root: Path, scenario: str) -> List[Path]:
    """
    汇总指定场景的 train+val 图像，供构建流式池。
    """
    if scenario not in SCENARIO_DIRS:
        raise ValueError(f"未知场景: {scenario}")
    dirs = SCENARIO_DIRS[scenario]
    candidates: List[Path] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".PNG", ".JPEG"}
    for d in dirs:
        img_dir = root / d / "images"
        if not img_dir.exists():
            continue
        for p in img_dir.iterdir():
            if p.suffix in exts:
                candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"未在 {root} 下找到 {scenario} 的 train/val 图像，请检查路径。")
    return sorted(set(candidates))


def ensure_pool_file(scenario: StreamScenario, master: List[Path], seed: int = 2025) -> None:
    """
    确保池子文件存在且容量匹配，不匹配则按 seed 重新采样生成。
    """
    dst = scenario.pool_file
    dst.parent.mkdir(parents=True, exist_ok=True)
    need = len(master) if scenario.pool_size <= 0 else scenario.pool_size
    if dst.exists():
        lines = [l.strip() for l in dst.read_text().splitlines() if l.strip()]
        if len(lines) == need:
            return
        print(f"[INFO] 池文件 {dst.name} 行数={len(lines)} 与需求 {need} 不符，重新生成。")

    if need > len(master):
        raise RuntimeError(f"可用图像 {len(master)} < 需求 {need}，无法生成 {dst}")

    if scenario.pool_size <= 0:
        chosen = master  # 直接用该场景全部 train+val
    else:
        rng = random.Random(seed)
        chosen = rng.sample(master, need)
    dst.write_text("\n".join(str(p) for p in chosen))
    print(f"[INFO] 已生成池文件 {dst}，共 {need} 张。")


class StreamImageDataset(Dataset):
    """根据文件列表加载图像，生成 weak/strong 两路增广，用于伪标与训练。"""

    def __init__(self, paths: Sequence[Path], size: int) -> None:
        self.paths = list(paths)
        self.size = size
        self.weak_tf = T.Compose(
            [
                T.Resize((size, size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
            ]
        )
        self.strong_tf = T.Compose(
            [
                T.Resize((size, size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                T.RandomGrayscale(p=0.1),
                T.ToTensor(),
                T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            ]
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:  # type: ignore[override]
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        img_weak = self.weak_tf(img)
        img_strong = self.strong_tf(img)
        return img_weak, img_strong, str(img_path)


def create_yolo_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """加载 YOLO 检测模型，确保参数可训练后返回。"""
    yolo = YOLO(str(model_path))
    model = yolo.model
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()  # 冻结 BN，避免流式自适应中统计被噪声拖偏
    for p in model.parameters():
        p.requires_grad = True  # 确保优化器有可训练参数（某些 checkpoint 可能默认冻结）
    return model.to(device)


def _tta_variants(img: Image.Image) -> List[Tuple[str, Image.Image]]:
    """生成 TTA 版本（原图 + 水平翻转）。"""
    return [("orig", img), ("flip", TF.hflip(img))]


def _map_xyxy(xyxy: np.ndarray, w: int, h: int, tag: str) -> np.ndarray:
    if tag == "flip":
        x1, y1, x2, y2 = xyxy
        return np.array([w - x2, y1, w - x1, y2], dtype=np.float32)
    return xyxy


def _boxes_from_result(res, img_w: int, img_h: int, tag: str) -> List[Tuple[np.ndarray, float, int]]:
    if not len(res.boxes):
        return []
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    mapped = []
    for coords, conf, cls_id in zip(xyxy, confs, clss):
        mapped.append((_map_xyxy(coords, img_w, img_h, tag), float(conf), int(cls_id)))
    return mapped


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
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


def tta_merge(model: YOLO, img: Image.Image, imgsz: int, conf: float, iou_merge: float) -> List[Tuple[np.ndarray, float, int]]:
    """
    原图 + 翻转，按 IoU/类别平均置信度与坐标。
    返回列表元素为 (xyxy, conf, cls_id)。
    """
    w, h = img.size
    all_boxes: List[Tuple[np.ndarray, float, int]] = []
    for tag, variant in _tta_variants(img):
        res = model.predict(variant, imgsz=imgsz, conf=conf, verbose=False)[0]
        all_boxes.extend(_boxes_from_result(res, w, h, tag))

    clusters: List[List[Tuple[np.ndarray, float, int]]] = []
    for box in all_boxes:
        placed = False
        for cluster in clusters:
            if cluster and cluster[0][2] == box[2] and _box_iou(cluster[0][0], box[0]) >= iou_merge:
                cluster.append(box)
                placed = True
                break
        if not placed:
            clusters.append([box])

    merged: List[Tuple[np.ndarray, float, int]] = []
    for cluster in clusters:
        coords = np.stack([b[0] for b in cluster], axis=0).mean(axis=0)
        conf_avg = float(np.mean([b[1] for b in cluster]))
        merged.append((coords, conf_avg, cluster[0][2]))
    return merged


def freeze_backbone(model: torch.nn.Module) -> None:
    """
    冻结 backbone，仅训练检测头（适合 few-shot/小样本）。
    尝试定位 ultralytics YOLO 的 Detect 头（通常在 model.model[-1]）。
    若无法定位，回退为不冻结，以避免空参数列表。
    """
    head = None
    # 先全部冻结
    for p in model.parameters():
        p.requires_grad = False
    # ultralytics YOLO: model.model 是 ModuleList，最后一层为 Detect 头
    if hasattr(model, "model"):
        layers = getattr(model, "model")
        try:
            if len(layers) > 0:
                head = layers[-1]
        except Exception:
            head = None
    # 兼容可能的 head 属性
    if head is None and hasattr(model, "head"):
        head = getattr(model, "head")
    if head is not None:
        for p in head.parameters():
            p.requires_grad = True
    # 如果仍然没有可训练参数，解除冻结（避免空 optimizer）
    if not any(p.requires_grad for p in model.parameters()):
        for p in model.parameters():
            p.requires_grad = True


def evaluate_map(model: torch.nn.Module, model_path: Path, data_yaml: Path, imgsz: int, batch: int, use_tta: bool = True) -> Dict[str, float]:
    """
    在带标签测试集上评估 mAP，返回关键指标。
    需构造 YOLO 包装器；使用当前模型的深拷贝以避免破坏训练中的 requires_grad 状态。
    """
    eval_model = copy.deepcopy(model)
    for p in eval_model.parameters():
        p.requires_grad = False
    yolo = YOLO(str(model_path))  # init to get task/config
    yolo.model = eval_model
    import torch.utils.data._utils.pin_memory as pin_memory

    orig_pin_loop = pin_memory._pin_memory_loop

    def safe_pin_memory_loop(in_queue, out_queue, device_id, done_event, local_pin_memory):
        try:
            orig_pin_loop(in_queue, out_queue, device_id, done_event, local_pin_memory)
        except Exception:
            # 屏蔽 DataLoader pin_memory 线程在 DataLoader 关闭时的 EOF/ConnectionReset 等噪声
            return

    pin_memory._pin_memory_loop = safe_pin_memory_loop  # type: ignore[attr-defined]
    try:
        metrics = yolo.val(
            data=str(data_yaml),
            split="test",
            imgsz=imgsz,
            batch=batch,
            verbose=False,
            save_json=False,
            plots=False,
            augment=use_tta,  # 开启 YOLO 内置 TTA（含翻转/缩放，内部做合并）
        )
    finally:
        pin_memory._pin_memory_loop = orig_pin_loop
    if hasattr(metrics, "results_dict"):
        res = metrics.results_dict
        return {
            "map50": float(res.get("metrics/mAP50(B)", 0.0)),
            "map5095": float(res.get("metrics/mAP50-95(B)", 0.0)),
            "precision": float(res.get("metrics/precision(B)", 0.0)),
            "recall": float(res.get("metrics/recall(B)", 0.0)),
        }
    return {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aggregate_epoch(epoch: int, step_results: List[StepResult]) -> Dict[str, Any]:
    mean_loss = float(sum(r.loss for r in step_results) / max(len(step_results), 1))
    agg: Dict[str, Any] = {"epoch": epoch, "mean_loss": mean_loss, "steps": len(step_results)}
    metric_keys = {k for r in step_results for k in r.metrics.keys()}
    for key in metric_keys:
        values = [r.metrics[key] for r in step_results if key in r.metrics]
        if not values:
            continue
        agg[f"mean_{key}"] = float(sum(values) / len(values))
        agg[f"sum_{key}"] = float(sum(values))
    return agg


def build_adapter(method: str, model: torch.nn.Module, args, lambda_override: float | None = None) -> Any:
    lambda_bias = args.lambda_bias if lambda_override is None else lambda_override
    lambda_l2 = args.lambda_l2 if lambda_override is None else lambda_override
    if method == "baseline":
        return SelfTrainingAdapter(
            model,
            lr=args.lr,
            device=args.device_torch,
            pseudo_threshold=args.pseudo_threshold,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
        )
    if method == "tent":
        return TentAdapter(model, lr=args.lr, device=args.device_torch, optimizer=args.optimizer, weight_decay=args.weight_decay)
    if method == "eata":
        return EataLiteAdapter(
            model,
            lr=args.lr,
            device=args.device_torch,
            entropy_threshold=args.entropy_threshold,
            lambda_l2=lambda_l2,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
        )
    if method == "bias_only":
        return BiasOnlyAdapter(
            model,
            lr=args.lr,
            device=args.device_torch,
            pseudo_threshold=args.pseudo_threshold,
            lambda_bias=lambda_bias,
            lambda_sensitivity=args.lambda_sensitivity,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
        )
    if method == "cotta":
        return CoTTAAdapter(
            model,
            lr=args.lr,
            device=args.device_torch,
            pseudo_threshold=args.pseudo_threshold,
            ema_alpha=args.cotta_alpha,
            kl_weight=args.cotta_kl_weight,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
        )
    if method == "st_cotta":
        return StabilizedCoTTAAdapter(
            model,
            lr=args.lr,
            device=args.device_torch,
            pseudo_threshold=args.pseudo_threshold,
            ema_alpha=args.cotta_alpha,
            kl_weight=args.cotta_kl_weight,
            lambda_bias=lambda_bias,
            lambda_sensitivity=args.lambda_sensitivity,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"未知方法: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream-style TTA on MT scenarios with different unlabeled pool sizes")
    parser.add_argument("--methods", nargs="*", default=["baseline", "tent", "eata", "bias_only"])
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("anchor_mt.pt"),
        help="源域 θ_good 权重路径（先在 MT 源域训练得到 best.pt 并复制/重命名为 anchor_mt.pt）",
    )
    parser.add_argument("--input-size", type=int, default=640, help="输入缩放尺寸")
    parser.add_argument("--lr", type=float, default=1e-5, help="TTA 学习率（小批量场景推荐 1e-5）")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"], help="优化器")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="权重衰减")
    parser.add_argument("--pseudo-threshold", type=float, default=0.6, help="伪标签置信度阈值")
    parser.add_argument("--entropy-threshold", type=float, default=1.5, help="熵筛选阈值 (EATA-lite)")
    parser.add_argument("--lambda-l2", type=float, default=1e-3, help="EATA-lite L2 正则系数")
    parser.add_argument("--lambda-bias", type=float, default=5e-4, help="Bias-only L_bias 系数（可加大抑制漂移）")
    parser.add_argument("--lambda-sensitivity", type=float, default=5.0, help="动态 λ 的敏感度 α（Bias-only / Stabilized-CoTTA）")
    parser.add_argument(
        "--lambda-bias-map",
        type=str,
        default="medium:0.001,high:0.005",
        help="按场景指定 λ_bias（并用于 EATA 的 lambda_l2），格式: medium:0.001,high:0.005",
    )
    parser.add_argument("--cotta-alpha", type=float, default=0.999, help="CoTTA EMA 衰减系数")
    parser.add_argument("--cotta-kl-weight", type=float, default=1.0, help="CoTTA KL 一致性权重")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=list(SCENARIOS.keys()),
        choices=list(SCENARIOS.keys()),
        help="选择要运行的场景，默认跑全部（few-shot/small/medium/high）",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="自定义结果输出根目录（未指定则使用 Results_stream_[freeze|nofreeze]）",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="仅训练检测头，冻结 backbone（few-shot/小样本建议开启，避免 Backbone 被噪声拖垮）",
    )
    parser.add_argument("--device", type=str, default="auto", help="cuda:0 / cpu / auto")
    parser.add_argument("--seeds", nargs="*", type=int, default=[1088, 2195, 4960], help="运行的随机种子列表")
    parser.add_argument("--eval-interval", type=int, default=2, help="多少个 epoch 评估一次 mAP（<=0 表示仅末轮）")
    parser.add_argument("--eval-batch", type=int, default=16, help="评估 batch 大小（test 集）")
    parser.add_argument("--regen-pools", action="store_true", help="强制重新生成池文件")
    parser.add_argument("--smoke-epochs", type=int, default=None, help="可选：覆盖场景 epochs（例如 1 做冒烟测试）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    args.device_torch = torch.device(device_str)
    if not args.model.exists():
        raise FileNotFoundError(
            f"未找到源域锚点模型: {args.model}\n"
            "请先参考 /root/autodl-tmp/ICML/MT 的源域训练流程，得到 best.pt 后复制/重命名为 anchor_mt.pt 再运行。"
        )

    data_root = Path("/root/autodl-tmp/dataset/MT-tgt-split")
    master_by_scenario: Dict[str, List[Path]] = {}
    for name in args.scenarios:
        master_by_scenario[name] = gather_scenario_images(data_root, name)

    for name in args.scenarios:
        scenario = SCENARIOS[name]
        if args.regen_pools and scenario.pool_file.exists():
            scenario.pool_file.unlink()
        ensure_pool_file(scenario, master_by_scenario[scenario.name], seed=2025)

    if args.results_root:
        results_root = Path(args.results_root)
    else:
        suffix = "freeze" if args.freeze_backbone else "nofreeze"
        results_root = DATA_ROOT / f"Results_stream_{suffix}"
    results_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    # 确保统一的 global_test YAML 存在
    ensure_global_test_yaml(GLOBAL_TEST_YAML)

    print("🎯 Stream TTA 实验配置（适应：train+val 池；监控：global_test）")
    print(f"   方法: {args.methods}")
    print(f"   seeds: {args.seeds}")
    print(f"   optimizer: {args.optimizer}, lr: {args.lr}, weight_decay: {args.weight_decay}")
    for name in args.scenarios:
        sc = SCENARIOS[name]
        if args.smoke_epochs is not None:
            sc.epochs = args.smoke_epochs
        pool_size = len(master_by_scenario[name]) if sc.pool_size <= 0 else sc.pool_size
        print(
            f"   - {name}: pool={pool_size} batch={sc.batch_size} epochs={sc.epochs} "
            f"pool_file={sc.pool_file.name} test_yaml={SCENARIO_YAML[name].name}"
        )

    lambda_map: Dict[str, float] = {}
    if args.lambda_bias_map:
        for item in args.lambda_bias_map.split(","):
            if not item.strip():
                continue
            name, val = item.split(":")
            lambda_map[name.strip()] = float(val)

    for scenario_name in args.scenarios:
        scenario = SCENARIOS[scenario_name]
        paths = [Path(p.strip()) for p in scenario.pool_file.read_text().splitlines() if p.strip()]
        expected = len(master_by_scenario[scenario.name]) if scenario.pool_size <= 0 else scenario.pool_size
        if len(paths) != expected:
            raise RuntimeError(f"{scenario.pool_file} 行数 {len(paths)} != 期望 {expected}")
        dataset = StreamImageDataset(paths, size=args.input_size)
        loader = DataLoader(
            dataset,
            batch_size=scenario.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        steps_per_epoch = len(loader)

        for seed in args.seeds:
            set_seed(seed)
            for method in args.methods:
                model = create_yolo_model(args.model, device=args.device_torch)
                if args.freeze_backbone:
                    freeze_backbone(model)
                lambda_override = lambda_map.get(scenario_name, None)
                adapter = build_adapter(method, model, args, lambda_override=lambda_override)

                epoch_logs: List[Dict[str, Any]] = []
                best_map50 = float("-inf")
                best_state = None
                best_epoch = 0
                for epoch in range(1, scenario.epochs + 1):
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    step_results: List[StepResult] = []
                    for batch in loader:
                        t0 = time.perf_counter()
                        res = adapter.train_batch(batch)
                        elapsed = time.perf_counter() - t0
                        bsz = batch.shape[0] if hasattr(batch, "shape") else scenario.batch_size
                        res.metrics["batch_time"] = float(elapsed)
                        res.metrics["batch_fps"] = float(bsz / elapsed) if elapsed > 0 else 0.0
                        step_results.append(res)
                    agg = aggregate_epoch(epoch, step_results)
                    if args.eval_interval <= 0:
                        should_eval = epoch == scenario.epochs
                    else:
                        interval = max(args.eval_interval, 1)
                        should_eval = (epoch % interval == 0) or (epoch == scenario.epochs)

                    if should_eval:
                        eval_metrics = evaluate_map(
                            adapter.model,
                            model_path=args.model,
                            data_yaml=SCENARIO_YAML[scenario_name],
                            imgsz=args.input_size,
                            batch=args.eval_batch,
                        )
                        agg.update({f"eval_{k}": v for k, v in eval_metrics.items()})
                        map50 = agg.get("eval_map50", None)
                        if map50 is not None and map50 > best_map50:
                            best_map50 = map50
                            best_state = copy.deepcopy(adapter.model.state_dict())
                            best_epoch = epoch
                    if torch.cuda.is_available():
                        vram_bytes = torch.cuda.max_memory_allocated()
                        agg["vram_max_bytes"] = int(vram_bytes)
                        agg["vram_max_mb"] = float(vram_bytes / (1024 * 1024))
                    epoch_logs.append(agg)
                    print(
                        f"[{scenario_name}][{method}][seed {seed}] "
                        f"epoch {epoch}/{scenario.epochs} loss={agg['mean_loss']:.4f} "
                        f"steps={agg['steps']} "
                        + (
                            f"mAP50={agg.get('eval_map50', 0):.4f} "
                            f"mAP50-95={agg.get('eval_map5095', 0):.4f}"
                            if "eval_map50" in agg
                            else ""
                        )
                    )

                method_dir = METHOD_TO_DIR.get(method, method)
                save_dir = results_root / method_dir / scenario_name / f"seed_{seed}"
                save_dir.mkdir(parents=True, exist_ok=True)

                (save_dir / "metrics.json").write_text(json.dumps(epoch_logs, indent=2, ensure_ascii=False))

                # 便于画曲线：导出 CSV（按键名排序，epoch/mean_loss/steps 优先）
                if epoch_logs:
                    all_keys = set().union(*(log.keys() for log in epoch_logs))
                    preferred = ["epoch", "mean_loss", "steps", "eval_map50", "eval_map5095", "eval_precision", "eval_recall"]
                    ordered = preferred + [k for k in sorted(all_keys) if k not in preferred]
                    with (save_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=ordered)
                        writer.writeheader()
                        for log in epoch_logs:
                            writer.writerow({k: log.get(k, "") for k in ordered})
                (save_dir / "config.json").write_text(
                    json.dumps(
                        {
                            "scenario": scenario_name,
                            "pool_file": str(scenario.pool_file),
                            "pool_size": scenario.pool_size,
                            "batch_size": scenario.batch_size,
                            "epochs": scenario.epochs,
                            "steps_per_epoch": steps_per_epoch,
                            "methods": args.methods,
                            "seed": seed,
                            "model_path": str(args.model),
                            "input_size": args.input_size,
                            "lr": args.lr,
                            "pseudo_threshold": args.pseudo_threshold,
                            "entropy_threshold": args.entropy_threshold,
                            "lambda_l2": lambda_map.get(scenario_name, args.lambda_l2),
                            "lambda_bias": lambda_map.get(scenario_name, args.lambda_bias),
                            "device": str(args.device_torch),
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
                final_state = adapter.model.state_dict()
                torch.save(final_state, save_dir / "last.pt")
                torch.save(final_state, save_dir / "final_model.pt")  # 兼容旧命名
                if best_state is None:
                    best_state = final_state
                torch.save(best_state, save_dir / "best.pt")

                try:
                    rel = save_dir.relative_to(DATA_ROOT)
                except ValueError:
                    rel = save_dir
                print(
                    f"[Done] {scenario_name} | {method} | seed {seed} -> {rel} "
                    f"(epochs={scenario.epochs}, steps/epoch={steps_per_epoch})"
                )


if __name__ == "__main__":
    main()
