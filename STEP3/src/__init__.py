"""
Compatibility alias package.

某些旧 checkpoint 在保存时使用了 `ICML.src.*` 路径；
当前代码实际放在 `ICML/core` 下。这里提供一个别名包，避免加载
旧权重时报 `ModuleNotFoundError: ICML.src`.
"""

from __future__ import annotations

# 暴露 yolo_bias_finetune 子包的别名
from . import yolo_bias_finetune  # noqa: F401

__all__ = ["yolo_bias_finetune"]

