"""Task interface shared by all synthetic benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class Batch:
    """A fixed-length batch of token ids and per-position labels.

    ``labels`` uses ``-100`` for positions that do not contribute to the loss.
    All sequences in a batch have the same length; length generalisation is done
    by constructing a separate batch per evaluation length.
    """

    input_ids: torch.Tensor  # (B, T), int64
    labels: torch.Tensor  # (B, T), int64, -100 = ignore
    meta: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> Batch:
        meta = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in self.meta.items()
        }
        return Batch(
            input_ids=self.input_ids.to(device),
            labels=self.labels.to(device),
            meta=meta,
        )

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])


@dataclass
class EvalSpec:
    """A named evaluation condition, e.g. a length or an MQAR configuration."""

    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    """A synthetic sequence task with on-the-fly sampling."""

    name: str
    vocab_size: int
    out_dim: int

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def _rng_or(self, rng: np.random.Generator | None) -> np.random.Generator:
        return self._rng if rng is None else rng

    @abstractmethod
    def sample(
        self,
        batch_size: int,
        *,
        rng: np.random.Generator | None = None,
        **kwargs: Any,
    ) -> Batch:
        """Sample a batch.  Extra keyword arguments are task-specific overrides."""

    def eval_specs(self) -> list[EvalSpec]:
        """Named conditions used by the training loop's evaluation pass."""
        return [EvalSpec("default", {})]
