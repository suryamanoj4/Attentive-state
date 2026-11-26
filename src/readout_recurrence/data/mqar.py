"""Multi-query associative recall (MQAR).

Sequence layout (with the query marker enabled):

    k1 v1 k2 v2 ... kP vP [distractor pairs] <Q> q1 <Q> q2 ...

``q_i`` is a key id, and the target at that position is the corresponding value
index.  Only query positions contribute to the loss.  Distractors are extra
unqueried pairs; their keys are drawn from the **value** vocabulary, so they can
never be confused with a queried key.  This makes the task well-defined while
still creating interference in the state.

Keys and values are separate token ids (``0..K-1`` and ``K..K+V-1``), so the
model must actually bind a key to its value rather than memorising a joint token.
The causal input convolution in the model is what makes adjacent-token binding
possible, mirroring the short convolution in Mamba-style models.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import MQARConfig
from .base import Batch, EvalSpec, Task


class MQARTask(Task):
    name = "mqar"

    def __init__(self, config: MQARConfig, seed: int = 0) -> None:
        super().__init__(seed)
        self.config = config
        self.vocab_size = config.n_keys + config.n_values + 2
        self.out_dim = config.n_values
        self.query_marker_id = config.n_keys + config.n_values
        self.pad_id = config.n_keys + config.n_values + 1

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample(
        self,
        batch_size: int,
        *,
        rng: np.random.Generator | None = None,
        n_pairs: int | None = None,
        n_queries: int | None = None,
        n_distractors: int | None = None,
        **_: object,
    ) -> Batch:
        rng = self._rng_or(rng)
        cfg = self.config
        n_pairs = cfg.n_pairs if n_pairs is None else int(n_pairs)
        n_queries = cfg.n_queries if n_queries is None else int(n_queries)
        n_distractors = cfg.n_distractors if n_distractors is None else int(n_distractors)
        if not 1 <= n_queries <= n_pairs:
            raise ValueError("n_queries must be in [1, n_pairs]")
        if n_distractors < 0:
            raise ValueError("n_distractors must be >= 0")
        if n_pairs > cfg.n_keys or n_pairs > cfg.n_values:
            raise ValueError("n_pairs cannot exceed the key/value vocabulary sizes")

        batch = batch_size
        # Distinct keys and values per row.  Random floats have negligible tie
        # probability; argsort gives a uniform random subset.
        keys = np.argsort(rng.random((batch, cfg.n_keys)), axis=1)[:, :n_pairs]
        values = np.argsort(rng.random((batch, cfg.n_values)), axis=1)[:, :n_pairs]
        if cfg.shuffle_pairs:
            order = np.argsort(rng.random((batch, n_pairs)), axis=1)
            keys = np.take_along_axis(keys, order, axis=1)
            values = np.take_along_axis(values, order, axis=1)

        # Keep the real key/value pairs aligned; queries index into these.
        query_pair_idx = np.argsort(rng.random((batch, n_pairs)), axis=1)[:, :n_queries]
        query_keys = np.take_along_axis(keys, query_pair_idx, axis=1)
        query_values = np.take_along_axis(values, query_pair_idx, axis=1)

        # Build the pair section.  Distractor keys come from the value vocab so
        # that a query key can never collide with a distractor key.
        if n_distractors > 0:
            distractor_keys = cfg.n_keys + rng.integers(
                0, cfg.n_values, size=(batch, n_distractors)
            )
            distractor_values = rng.integers(0, cfg.n_values, size=(batch, n_distractors))
            all_keys = np.concatenate([keys, distractor_keys], axis=1)
            all_values = np.concatenate([values, distractor_values], axis=1)
            if cfg.shuffle_pairs:
                pair_order = np.argsort(rng.random((batch, n_pairs + n_distractors)), axis=1)
                all_keys = np.take_along_axis(all_keys, pair_order, axis=1)
                all_values = np.take_along_axis(all_values, pair_order, axis=1)
        else:
            all_keys = keys
            all_values = values

        n_total_pairs = n_pairs + n_distractors
        pair_len = 2 * n_total_pairs
        query_len = n_queries * (2 if cfg.query_marker else 1)
        seq_len = pair_len + query_len

        input_ids = np.empty((batch, seq_len), dtype=np.int64)
        labels = np.full((batch, seq_len), -100, dtype=np.int64)

        # Pair tokens: even positions are keys, odd positions are values.
        input_ids[:, 0:pair_len:2] = all_keys
        input_ids[:, 1:pair_len:2] = cfg.n_keys + all_values

        pos = pair_len
        for q in range(n_queries):
            if cfg.query_marker:
                input_ids[:, pos] = self.query_marker_id
                pos += 1
            input_ids[:, pos] = query_keys[:, q]
            labels[:, pos] = query_values[:, q]
            pos += 1

        query_mask = labels != -100
        return Batch(
            input_ids=torch.from_numpy(input_ids),
            labels=torch.from_numpy(labels),
            meta={"query_mask": torch.from_numpy(query_mask)},
        )

    # ------------------------------------------------------------------
    # Evaluation grid
    # ------------------------------------------------------------------
    def eval_specs(self) -> list[EvalSpec]:
        specs: list[EvalSpec] = [EvalSpec("train", {})]
        seen: set[tuple[int, int]] = {(self.config.n_pairs, self.config.n_distractors)}
        for pairs in self.config.eval_pairs:
            key = (pairs, 0)
            if key not in seen:
                # Never ask for more queries than there are pairs in this
                # evaluation condition.
                n_queries = min(self.config.n_queries, pairs)
                specs.append(
                    EvalSpec(
                        f"pairs{pairs}_dist0",
                        {"n_pairs": pairs, "n_queries": n_queries},
                    )
                )
                seen.add(key)
        for distractors in self.config.eval_distractors:
            if distractors == 0:
                continue
            key = (self.config.n_pairs, distractors)
            if key not in seen:
                specs.append(
                    EvalSpec(
                        f"pairs{self.config.n_pairs}_dist{distractors}",
                        {"n_distractors": distractors},
                    )
                )
                seen.add(key)
        return specs
