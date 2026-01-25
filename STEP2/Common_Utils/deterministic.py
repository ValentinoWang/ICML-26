import os
import random
from typing import Dict, Callable

import numpy as np
import torch


def set_deterministic(seed: int = 42) -> None:
    """
    Configure deterministic behavior across Python, NumPy, and PyTorch (CPU/GPU).
    Also configures cuDNN / TF32 flags and sets an environment variable for cublas.
    """
    # For some GEMM paths determinism requires this env var (must be set before the first GEMM).
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN / TF32 switches
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Enforce deterministic algorithms (may raise if an op has no deterministic version)
    torch.use_deterministic_algorithms(True)

    # Avoid non-deterministic SDPA backends on CUDA (e.g., Flash / mem-efficient).
    # Force math SDP which is deterministic but slower.
    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)


def make_dataloader_seed(seed: int) -> Dict[str, Callable]:
    """
    Return kwargs (worker_init_fn, generator) to plug into DataLoader
    so that workers inherit deterministic seeds.
    """
    def seed_worker(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
    return {"worker_init_fn": seed_worker, "generator": g}
