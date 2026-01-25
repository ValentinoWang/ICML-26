"""
Compatibility shim for checkpoints saved under `ICML.src.yolo_bias_finetune.*`.

实际代码位于 `ICML.core.yolo_bias_finetune`，这里简单转发 import。
"""

from __future__ import annotations

from ICML.core.yolo_bias_finetune.anchor import *  # noqa: F401,F403
from ICML.core.yolo_bias_finetune.bias_trainer import *  # noqa: F401,F403

# 兼容 PyTorch 2.6 默认 weights_only=True 的安全反序列化：
# 将需要的自定义类加入 safe_globals，避免加载旧 checkpoint 报 Unsupported global。
try:  # torch>=2.6
    import torch.serialization as _ts

    if "BiasWrappedModel" in globals():
        _ts.add_safe_globals([globals()["BiasWrappedModel"]])
except Exception:
    pass

__all__ = []
__all__ += [n for n in dir() if not n.startswith("_")]
