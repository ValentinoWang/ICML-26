"""
City → Foggy YOLO + L_bias 微调实验配置

职责：
- 管理 Cityscapes → FoggyCityscapes 实验的统一配置；
- 指定 θ_good（Cityscapes 上的黄金模型）路径；
- 指定 Foggy 多场景 data.yaml 路径；
- 提供与 MT 版 BiasFinetuneConfig 兼容的数据结构，便于复用训练代码。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json


# Baseline 根目录：.../Baseline
BASELINE_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = Path(__file__).resolve().parent
LOCAL_CONFIG_PATH = CONFIG_ROOT / "unified_config.json"


@dataclass
class CityFogBiasFinetuneConfig:
    """City → Foggy YOLO + L_bias 微调实验配置"""

    seeds: List[int]
    scenarios: List[str]
    training: Dict[str, Any]
    theta_good_path: Path
    scenario_data_cfg: Dict[str, Path]
    results_root: Path
    lambda_bias: float = 1e-4


def load_unified_config() -> Dict[str, Any]:
    """
    加载 City-Foggy 实验的统一配置文件。
    """
    if not LOCAL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"找不到 City-Foggy unified_config.json: {LOCAL_CONFIG_PATH}")
    with LOCAL_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_cityfog_bias_finetune_config(
    theta_good_seed: int = 1088,
    lambda_bias: float = 1e-4,
) -> CityFogBiasFinetuneConfig:
    """
    构造 City → Foggy YOLO + L_bias 微调实验配置。

    约定：
    - θ_good: 来自 Pretrain-Finetune/Results/shared_pretrain_city/seed_xxx/weights/best.pt
      （需要用户先在 Cityscapes-DET 上完成 shared_pretrain_city 训练）；
    - data.yaml: 来自本目录 configs/ 下 Foggy-tgt-*.yaml；
    - 结果目录: Baseline/ICML/cityfog/Results/Bias_only。
    """
    unified_cfg = load_unified_config()

    seeds = unified_cfg["experiment"]["seeds"]
    scenarios = unified_cfg["experiment"]["scenarios_to_run"]
    training = unified_cfg["training"]

    theta_good_path = (
        BASELINE_ROOT
        / "Pretrain-Finetune"
        / "Results"
        / "shared_pretrain_city"
        / f"seed_{theta_good_seed}"
        / "weights"
        / "best.pt"
    )

    scenario_data_cfg: Dict[str, Path] = {}
    for name, info in unified_cfg["data"]["scenarios"].items():
        scenario_data_cfg[name] = CONFIG_ROOT / info["config"]

    # 结果目录：使用实际目录名 City→Foggy，避免写入虚拟别名路径
    results_root = BASELINE_ROOT / "ICML" / "City→Foggy" / "Results" / "Bias_only"

    return CityFogBiasFinetuneConfig(
        seeds=seeds,
        scenarios=scenarios,
        training=training,
        theta_good_path=theta_good_path,
        scenario_data_cfg=scenario_data_cfg,
        results_root=results_root,
        lambda_bias=lambda_bias,
    )


__all__ = ["CityFogBiasFinetuneConfig", "build_cityfog_bias_finetune_config"]
