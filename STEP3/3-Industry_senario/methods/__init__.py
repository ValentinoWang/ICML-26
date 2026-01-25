"""
Long-term TTA 方法注册表。

用于：
- 统一方法名称 -> 结果目录名的映射；
- 预留方法子目录，便于未来扩展自定义代码/实验脚本。
"""

from __future__ import annotations

# 方法名（命令行标识） -> 结果/子目录名
METHOD_TO_DIR = {
    "baseline": "Baseline_SelfTraining",
    "tent": "TENT",
    "eata": "EATA_Lite",
    "bias_only": "Ours_Bias_only",
    "cotta": "CoTTA",
    "st_cotta": "Stabilized_CoTTA",
}

__all__ = ["METHOD_TO_DIR"]
