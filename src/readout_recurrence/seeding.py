"""Seeding and device helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch.

    Note that the task generators use their own NumPy generators.  Seeding the
    global RNGs here is for model initialisation and any library-level randomness.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(device)


def configure_threads(num_threads: int) -> None:
    if num_threads > 0:
        torch.set_num_threads(num_threads)


def mps_fallback_enabled() -> bool:
    return os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
