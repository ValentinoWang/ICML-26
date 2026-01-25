#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
长时测试时自适应（Long-term TTA）模拟脚本。

实现 4 种无标签自适应策略，在 MT 目标域四个稀缺度场景上跑 50 轮：
- Baseline (Self-Training): 伪标签自训练
- TENT (SOTA 1): 熵最小化
- EATA-lite (SOTA 2 简化): 熵阈值筛选 + L2 正则
- Ours (Bias_only): 伪标签自训练 + L_bias 锚定

结果目录：3-Industry_senario/Results/<Method>/<scenario>/seed_<seed>/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Iterable, Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader

DATA_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DATA_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters import (  # noqa: E402
    SelfTrainingAdapter,
    TentAdapter,
    EataLiteAdapter,
    BiasOnlyAdapter,
    StepResult,
)
from methods import METHOD_TO_DIR  # noqa: E402
from ultralytics import YOLO  # noqa: E402


@dataclass
class ScenarioInfo:
    name: str
    yaml_path: Path
    num_classes: int
    unlabeled_dir: Path


@dataclass
class LongTermTTAConfig:
    seeds: List[int]
    scenarios: Dict[str, ScenarioInfo]
    rounds: int = 50
    batches_per_round: int = 4
    batch_size: int = 16
    input_size: int = 128
    lr: float = 1e-3
    pseudo_threshold: float = 0.6
    entropy_threshold: float = 1.5
    lambda_l2: float = 1e-3
    lambda_bias: float = 1e-4
    results_root: Path = DATA_ROOT / "Results"
    model_path: Path = Path("yolov8n.pt")


def list_images(img_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if not img_dir.exists():
        raise FileNotFoundError(f"未找到图像目录: {img_dir}")
    imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in exts]
    if not imgs:
        raise RuntimeError(f"目录中未找到图片: {img_dir}")
    return sorted(imgs)


def load_scenarios() -> Dict[str, ScenarioInfo]:
    unified_cfg = load_unified_config()
    cfg_root = Path(__file__).resolve().parents[1] / "MT" / "config"
    scenario_infos: Dict[str, ScenarioInfo] = {}

    for name in unified_cfg["experiment"]["scenarios_to_run"]:
        info = unified_cfg["data"]["scenarios"][name]
        yaml_path = cfg_root / info["config"]
        data_cfg = yaml.safe_load(yaml_path.read_text())
        names = data_cfg.get("names", [])
        if isinstance(names, dict):
            num_classes = len(names)
        else:
            num_classes = len(names)
        unlabeled_dir = Path(data_cfg.get("test") or data_cfg.get("val") or data_cfg["train"])
        scenario_infos[name] = ScenarioInfo(
            name=name,
            yaml_path=yaml_path,
            num_classes=num_classes,
            unlabeled_dir=Path(unlabeled_dir),
        )
    return scenario_infos


def load_unified_config() -> Dict[str, Any]:
    cfg_root = Path(__file__).resolve().parents[1] / "MT" / "config"
    cfg_path = cfg_root / "unified_config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


class UnlabeledImageDataset(Dataset):
    """简单的无标签图像集，将图片缩放到固定尺寸并标准化到 [0,1]。"""

    def __init__(self, images: List[Path], size: int) -> None:
        self.images = images
        self.size = size

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:  # type: ignore[override]
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.size, self.size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # CHW
        return torch.from_numpy(arr)


def infinite_batches(loader: DataLoader) -> Iterable[torch.Tensor]:
    """循环从 DataLoader 中取 batch，避免 50 轮时手动重建迭代器。"""
    while True:
        for batch in loader:
            yield batch


def create_yolo_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """加载 YOLO 检测模型，返回可微分的 nn.Module（默认 yolov8n.pt）。"""
    yolo = YOLO(str(model_path))
    model = yolo.model
    return model.to(device)


def run_rounds(
    adapter,
    loader: DataLoader,
    rounds: int,
    batches_per_round: int,
) -> List[Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []
    batch_iter = infinite_batches(loader)

    for rnd in range(1, rounds + 1):
        step_results: List[StepResult] = []
        for _ in range(batches_per_round):
            batch = next(batch_iter)
            res = adapter.train_batch(batch)
            step_results.append(res)

        mean_loss = float(sum(r.loss for r in step_results) / max(len(step_results), 1))
        agg_metrics: Dict[str, Any] = {"round": rnd, "mean_loss": mean_loss, "steps": len(step_results)}

        # 逐键聚合（均值 + 总和），方便后续画图
        metric_keys = {k for r in step_results for k in r.metrics.keys()}
        for key in metric_keys:
            values = [r.metrics[key] for r in step_results if key in r.metrics]
            if not values:
                continue
            agg_metrics[f"mean_{key}"] = float(sum(values) / len(values))
            agg_metrics[f"sum_{key}"] = float(sum(values))

        logs.append(agg_metrics)
    return logs


def build_adapter(method: str, model: torch.nn.Module, cfg: LongTermTTAConfig, device: torch.device):
    if method == "baseline":
        return SelfTrainingAdapter(model, lr=cfg.lr, device=device, pseudo_threshold=cfg.pseudo_threshold)
    if method == "tent":
        return TentAdapter(model, lr=cfg.lr, device=device)
    if method == "eata":
        return EataLiteAdapter(
            model,
            lr=cfg.lr,
            device=device,
            entropy_threshold=cfg.entropy_threshold,
            lambda_l2=cfg.lambda_l2,
        )
    if method == "bias_only":
        return BiasOnlyAdapter(
            model,
            lr=cfg.lr,
            device=device,
            pseudo_threshold=cfg.pseudo_threshold,
            lambda_bias=cfg.lambda_bias,
        )
    raise ValueError(f"未知方法: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-term TTA simulation on MT scenarios")
    parser.add_argument("--methods", nargs="*", default=["baseline", "tent", "eata", "bias_only"])
    parser.add_argument("--rounds", type=int, default=50, help="自适应轮次")
    parser.add_argument("--batches-per-round", type=int, default=4, help="每轮使用多少个 batch 更新")
    parser.add_argument("--batch-size", type=int, default=8, help="无标签流 batch 大小")
    parser.add_argument("--input-size", type=int, default=640, help="输入缩放尺寸 (square)")
    parser.add_argument("--lr", type=float, default=1e-3, help="TTA 学习率")
    parser.add_argument("--pseudo-threshold", type=float, default=0.6, help="伪标签置信度阈值 (Self-Training/Ours)")
    parser.add_argument("--entropy-threshold", type=float, default=1.5, help="熵筛选阈值 (EATA-lite)")
    parser.add_argument("--lambda-l2", type=float, default=1e-3, help="EATA-lite 的 L2 正则系数")
    parser.add_argument("--lambda-bias", type=float, default=1e-4, help="Bias-only 的 L_bias 系数")
    parser.add_argument("--device", type=str, default="auto", help="cuda:0 / cpu / auto")
    parser.add_argument("--model", type=Path, default=Path("yolov8n.pt"), help="YOLO 检测模型权重路径")
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="可选：覆盖默认的种子列表（默认使用 MT unified_config.json 中的 seeds）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)

    scenarios = load_scenarios()
    unified_cfg = load_unified_config()
    seeds = args.seeds if args.seeds else unified_cfg["experiment"]["seeds"]

    cfg = LongTermTTAConfig(
        seeds=seeds,
        scenarios=scenarios,
        rounds=args.rounds,
        batches_per_round=args.batches_per_round,
        batch_size=args.batch_size,
        input_size=args.input_size,
        lr=args.lr,
        pseudo_threshold=args.pseudo_threshold,
        entropy_threshold=args.entropy_threshold,
        lambda_l2=args.lambda_l2,
        lambda_bias=args.lambda_bias,
        model_path=Path(args.model),
    )

    results_root = cfg.results_root

    print("🎯 Long-term TTA (50 轮默认)")
    print(f"   场景: {list(scenarios.keys())}")
    print(f"   方法: {args.methods}")
    print(f"   Seeds: {seeds}")
    print(f"   结果目录: {results_root}")

    for scenario_name, scenario in scenarios.items():
        images = list_images(Path(scenario.unlabeled_dir))
        dataset = UnlabeledImageDataset(images, size=cfg.input_size)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            for method in args.methods:
                model = create_yolo_model(cfg.model_path, device=device)
                adapter = build_adapter(method, model, cfg, device=device)

                logs = run_rounds(adapter, loader, rounds=cfg.rounds, batches_per_round=cfg.batches_per_round)

                method_dir = METHOD_TO_DIR.get(method, method)
                save_dir = results_root / method_dir / scenario_name / f"seed_{seed}"
                save_dir.mkdir(parents=True, exist_ok=True)

                metrics_path = save_dir / "metrics.json"
                metrics_path.write_text(json.dumps(logs, indent=2, ensure_ascii=False))

                config_path = save_dir / "config.json"
                config_path.write_text(
                    json.dumps(
                        {
                            "method": method,
                            "seed": seed,
                            "rounds": cfg.rounds,
                            "batches_per_round": cfg.batches_per_round,
                            "batch_size": cfg.batch_size,
                            "lr": cfg.lr,
                            "pseudo_threshold": cfg.pseudo_threshold,
                            "entropy_threshold": cfg.entropy_threshold,
                            "lambda_l2": cfg.lambda_l2,
                            "lambda_bias": cfg.lambda_bias,
                            "model_path": str(cfg.model_path),
                            "scenario_yaml": str(scenario.yaml_path),
                            "unlabeled_dir": str(scenario.unlabeled_dir),
                            "num_classes": scenario.num_classes,
                            "device": str(device),
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )

                # 保存最终权重，便于后续可视化漂移或继续微调
                torch.save(adapter.model.state_dict(), save_dir / "final_model.pt")

                print(
                    f"[Done] {method_dir} | {scenario_name} | seed {seed} -> {metrics_path.relative_to(DATA_ROOT.parent)}"
                )


if __name__ == "__main__":
    main()
