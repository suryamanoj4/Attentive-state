"""Cumulative group multiplication: A5, S5, and parity (Z2).

Each token is a group element; the target at position ``t`` is the product of
all tokens up to ``t``.  The model sees the elements one at a time and must
maintain the running product.  A5 is the standard NC1-complete state-tracking
probe; a diagonal recurrence cannot represent the required permutations, while
a block-dense transition of width at least 5 can.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch

from ..config import GroupConfig
from .base import Batch, EvalSpec, Task


def _permutations(n: int) -> list[tuple[int, ...]]:
    return list(itertools.permutations(range(n)))


def _sign(permutation: tuple[int, ...]) -> int:
    inversions = 0
    for i in range(len(permutation)):
        for j in range(i + 1, len(permutation)):
            inversions += permutation[i] > permutation[j]
    return -1 if inversions % 2 else 1


def _compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    """Return ``left ∘ right``: apply ``right`` first, then ``left``."""
    return tuple(left[right[i]] for i in range(len(left)))


class GroupTask(Task):
    name = "group"

    def __init__(
        self,
        config: GroupConfig,
        seed: int = 0,
        *,
        task_name: str | None = None,
    ) -> None:
        super().__init__(seed)
        self.config = config
        group = config.group
        if task_name == "a5":
            group = "A5"
        elif task_name == "s5":
            group = "S5"
        elif task_name == "parity":
            group = "Z2"
        self.group = group

        if group == "Z2":
            self.elements = [(0,), (1,)]
            self.table = np.array([[0, 1], [1, 0]], dtype=np.int64)
        else:
            permutations = _permutations(5)
            if group == "A5":
                permutations = [p for p in permutations if _sign(p) == 1]
            self.elements = permutations
            index = {element: i for i, element in enumerate(permutations)}
            table = np.empty((len(permutations), len(permutations)), dtype=np.int64)
            for i, left in enumerate(permutations):
                for j, right in enumerate(permutations):
                    table[i, j] = index[_compose(left, right)]
            self.table = table

        self.group_size = len(self.elements)
        # +1 for a padding token; fixed-length batches do not use it, but keeping
        # it makes the task interface uniform with the other generators.
        self.vocab_size = self.group_size + 1
        self.out_dim = self.group_size

    def sample(
        self,
        batch_size: int,
        *,
        rng: np.random.Generator | None = None,
        length: int | None = None,
        **_: object,
    ) -> Batch:
        rng = self._rng_or(rng)
        cfg = self.config
        if length is None:
            length = int(rng.integers(cfg.train_min_len, cfg.train_max_len + 1))
        length = int(length)
        if length < 1:
            raise ValueError("length must be >= 1")

        tokens = rng.integers(0, self.group_size, size=(batch_size, length)).astype(np.int64)
        labels = np.empty((batch_size, length), dtype=np.int64)
        state = np.zeros(batch_size, dtype=np.int64)  # identity index
        for t in range(length):
            state = self.table[state, tokens[:, t]]
            labels[:, t] = state

        return Batch(
            input_ids=torch.from_numpy(tokens),
            labels=torch.from_numpy(labels),
        )

    def eval_specs(self) -> list[EvalSpec]:
        return [EvalSpec(f"len{length}", {"length": length}) for length in self.config.eval_lengths]
