"""Copying task: reproduce a random content sequence after a delimiter.

The model is trained with next-token prediction on the second copy:

    [BOS] x1 ... xN [SEP] x1 ... xN

At the ``SEP`` position it predicts ``x1``; at the first copied token it predicts
``x2``; ...; at the last copied token it predicts ``EOS``.  The first copy is
input only.  The state must hold the sequence until the delimiter, which makes
this a clean memory-capacity probe: attention over slots can improve readout but
cannot create more state than the recurrence has.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import CopyConfig
from .base import Batch, EvalSpec, Task


class CopyTask(Task):
    name = "copy"

    def __init__(self, config: CopyConfig, seed: int = 0) -> None:
        super().__init__(seed)
        self.config = config
        self.bos_id = config.content_vocab
        self.sep_id = config.content_vocab + 1
        self.eos_id = config.content_vocab + 2
        self.pad_id = config.content_vocab + 3
        self.vocab_size = config.content_vocab + 4
        self.out_dim = self.vocab_size

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

        content = rng.integers(0, cfg.content_vocab, size=(batch_size, length)).astype(np.int64)
        seq_len = 2 * length + 2
        input_ids = np.empty((batch_size, seq_len), dtype=np.int64)
        labels = np.full((batch_size, seq_len), -100, dtype=np.int64)

        input_ids[:, 0] = self.bos_id
        input_ids[:, 1 : 1 + length] = content
        input_ids[:, 1 + length] = self.sep_id
        input_ids[:, 2 + length :] = content

        # First output is at the delimiter; then teacher-forced next tokens.
        labels[:, 1 + length] = content[:, 0]
        if length > 1:
            labels[:, 2 + length : 2 * length + 1] = content[:, 1:]
        labels[:, 2 * length + 1] = self.eos_id

        return Batch(
            input_ids=torch.from_numpy(input_ids),
            labels=torch.from_numpy(labels),
        )

    def eval_specs(self) -> list[EvalSpec]:
        return [EvalSpec(f"len{length}", {"length": length}) for length in self.config.eval_lengths]
