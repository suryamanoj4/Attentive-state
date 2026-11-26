"""Task registry."""

from __future__ import annotations

from ..config import TaskConfig
from .base import Batch, EvalSpec, Task
from .copy import CopyTask
from .group import GroupTask
from .mqar import MQARTask


def build_task(config: TaskConfig, seed: int = 0) -> Task:
    if config.name == "mqar":
        return MQARTask(config.mqar, seed=seed)
    if config.name in {"group", "a5", "s5", "parity"}:
        return GroupTask(config.group, seed=seed, task_name=config.name)
    if config.name == "copy":
        return CopyTask(config.copy, seed=seed)
    raise ValueError(f"unknown task name: {config.name!r}")


__all__ = ["Batch", "EvalSpec", "Task", "build_task"]
