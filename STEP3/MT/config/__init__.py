"""
配置与路径管理模块

职责：
- 提供统一的实验配置对象
- 管理 θ_good（锚点模型）和目标域数据配置的路径
- 暴露纯数据结构，避免直接耦合训练细节
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import json


# Baseline 根目录：.../Baseline
# 当前文件路径: Baseline/ICML/config/yolo_bias_finetune/__init__.py
# parents[0] = yolo_bias_finetune, [1] = config, [2] = ICML, [3] = Baseline
BASELINE_ROOT = Path(__file__).resolve().parents[3]
# 本模块配置根目录：.../Toy/config/yolo_bias_finetune
CONFIG_ROOT = Path(__file__).resolve().parent
LOCAL_CONFIG_PATH = CONFIG_ROOT / "unified_config.json"


@dataclass
class BiasFinetuneConfig:
    """YOLO + L_bias 微调实验的核心配置"""

    # 随机数种子
    seeds: List[int]

    # 要跑的目标域场景（few-shot / small / medium / high）
    scenarios: List[str]

    # YOLO 训练超参数（直接复用 Target-Only 的 unified_config）
    training: Dict[str, Any]

    # θ_good（源域黄金模型）路径
    theta_good_path: Path

    # 目标域数据集配置映射：scenario -> data.yaml 路径
    scenario_data_cfg: Dict[str, Path]

    # 结果输出根目录
    results_root: Path

    # L_bias 权重
    lambda_bias: float = 1e-4


def load_target_only_unified_config() -> Dict[str, Any]:
    """
    加载偏置实验的统一配置文件。
    """
    if LOCAL_CONFIG_PATH.exists():
        cfg_path = LOCAL_CONFIG_PATH
    else:
        cfg_path = BASELINE_ROOT / "Target-Only" / "configs" / "unified_config.json"

    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_bias_finetune_config(
    theta_good_seed: int = 1088,
    lambda_bias: float = 1e-4,
) -> BiasFinetuneConfig:
    """
    根据已有工程结构，构造一个默认的 BiasFinetuneConfig（干净 MT-tgt train）。

    - θ_good：使用 Pretrain-Finetune/shared_pretrain/seed_xxx/weights/best.pt
    - seeds、训练参数、场景：来自本目录 unified_config.json
    - 数据 yaml：来自本目录 configs/ 下的四个 mt-tgt-*.yaml
    """
    unified_cfg = load_target_only_unified_config()

    seeds = unified_cfg["experiment"]["seeds"]
    scenarios = unified_cfg["experiment"]["scenarios_to_run"]
    training = unified_cfg["training"]

    # θ_good 路径
    theta_good_path = (
        BASELINE_ROOT
        / "Pretrain-Finetune"
        / "Results"
        / "shared_pretrain"
        / f"seed_{theta_good_seed}"
        / "weights"
        / "best.pt"
    )

    # 场景数据配置（相对于本 config 目录的 configs 子目录）
    scenario_data_cfg: Dict[str, Path] = {}
    for name, info in unified_cfg["data"]["scenarios"].items():
        scenario_data_cfg[name] = CONFIG_ROOT / info["config"]

    # 结果目录统一放在 ICML/mt/Results/Bias_only 下，用于存放「仅 YOLO + L_bias」的实验结果
    results_root = BASELINE_ROOT / "ICML" / "mt" / "Results" / "Bias_only"

    return BiasFinetuneConfig(
        seeds=seeds,
        scenarios=scenarios,
        training=training,
        theta_good_path=theta_good_path,
        scenario_data_cfg=scenario_data_cfg,
        results_root=results_root,
        lambda_bias=lambda_bias,
    )


def build_bias_finetune_noise_config(
    theta_good_seed: int = 1088,
    lambda_bias: float = 1e-4,
) -> BiasFinetuneConfig:
    """
    构造一份带显式噪声 train（D_noise + OOD）的 BiasFinetuneConfig，仅针对 few-shot / small 场景。

    - few-shot：使用 mt-tgt-few-shot-noise.yaml 的混合 train；
    - small：使用 mt-tgt-small-noise.yaml 的混合 train；
    - 结果目录放在 ICML/Results/Bias_only_noise 下，避免与干净实验混在一起。
    """
    unified_cfg = load_target_only_unified_config()

    seeds = unified_cfg["experiment"]["seeds"]
    # 只在 few-shot / small 上构造带噪声的配置
    scenarios = [s for s in unified_cfg["experiment"]["scenarios_to_run"] if s in ("few-shot", "small")]
    training = unified_cfg["training"]

    # θ_good 路径与干净版本共享
    theta_good_path = (
        BASELINE_ROOT
        / "Pretrain-Finetune"
        / "Results"
        / "shared_pretrain"
        / f"seed_{theta_good_seed}"
        / "weights"
        / "best.pt"
    )

    # 噪声场景的数据配置，直接指向 *_noise.yaml
    scenario_data_cfg: Dict[str, Path] = {}
    scenario_data_cfg["few-shot"] = CONFIG_ROOT / "configs" / "mt-tgt-few-shot-noise.yaml"
    scenario_data_cfg["small"] = CONFIG_ROOT / "configs" / "mt-tgt-small-noise.yaml"

    # 结果目录放在 ICML/mt/Results/Bias_only_noise 下
    results_root = BASELINE_ROOT / "ICML" / "mt" / "Results" / "Bias_only_noise"

    return BiasFinetuneConfig(
        seeds=seeds,
        scenarios=scenarios,
        training=training,
        theta_good_path=theta_good_path,
        scenario_data_cfg=scenario_data_cfg,
        results_root=results_root,
        lambda_bias=lambda_bias,
    )


__all__ = ["BiasFinetuneConfig", "build_bias_finetune_config", "build_bias_finetune_noise_config"]
