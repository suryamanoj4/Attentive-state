"""Token encoder, block-structured transition, and recurrent layer.

State geometry
--------------
The state is ``H ∈ R^{B × L × d}``: ``L`` slots of width ``d``.  Each slot is
partitioned into ``n_blocks = d / b`` blocks of size ``b``.  A transition block
row for destination slot ``i`` and block ``r`` has shape ``b × (S·b)``, where
``S`` is the number of source slots allowed by the state type:

* ``diagonal``: ``b = 1``, ``S = 1``, transition is 1 (softmax) or ±1 (signed).
* ``block_diagonal``: ``S = 1``, dense ``b × b`` block.
* ``cascade_chain``: ``S = 2``, previous slot plus self.
* ``cascade_full``: ``S = L``, lower-triangular source mask.
* ``full``: ``S = L``, all sources.

Normalisation is row-wise: softmax gives a positive row-stochastic transition;
``signed_l1`` applies ``tanh`` and divides by the row L1 norm, which allows
signed permutation-like blocks while keeping the row L1 norm at 1.  Both are
stable under the infinity norm.  The state update is a gated convex combination

    H_t = g_t ⊙ (A_t H_{t-1}) + (1 − g_t) ⊙ W_t u_t

so the forward state is bounded by the maximum of the previous state and the
write, even when the transition is input-dependent.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelConfig


class TokenEncoder(nn.Module):
    """Embedding plus an optional causal convolution and MLP.

    The convolution is the input-side binding mechanism: at a value position it
    can see the preceding key token, which is what lets a vector-state model
    represent key/value associations.  It is causal (left padding only).
    """

    def __init__(self, vocab_size: int, config: ModelConfig) -> None:
        super().__init__()
        self.kernel = config.token_conv_kernel
        self.embed = nn.Embedding(vocab_size, config.d_model)
        self.conv = nn.Linear(self.kernel * config.d_model, config.d_model)
        self.mlp = (
            nn.Sequential(
                nn.Linear(config.d_model, config.token_mlp_hidden),
                nn.GELU(),
                nn.Linear(config.token_mlp_hidden, config.d_model),
            )
            if config.token_mlp_hidden > 0
            else nn.Identity()
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embed(input_ids)  # (B, T, d)
        if self.kernel == 1:
            mixed = emb
        else:
            # Stack [e_t, e_{t-1}, ..., e_{t-k+1}] along the feature axis.
            shifts = [emb]
            for offset in range(1, self.kernel):
                shifted = F.pad(emb[:, : emb.shape[1] - offset, :], (0, 0, offset, 0))
                shifts.append(shifted)
            mixed = torch.cat(shifts, dim=-1)
        out = self.conv(mixed)
        out = self.mlp(out)
        out = self.dropout(out)
        return self.norm(out)


class BlockTransition(nn.Module):
    """Input-dependent block-structured transition."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_slots = config.n_slots
        self.slot_dim = config.slot_dim
        self.block_size = config.block_size
        self.n_blocks = self.slot_dim // self.block_size
        self.norm = config.transition_norm
        self.temperature = config.transition_temperature
        self.state_type = config.state_type

        source_index, source_mask = self._build_sources()
        self.register_buffer("source_index", source_index, persistent=False)
        self.register_buffer("source_mask", source_mask, persistent=False)
        self.n_sources = int(source_index.shape[1])

        # Cross-slot (leakage) source mask.  For cascade/full state types this
        # lets ``cascade_init='block_diagonal'`` start from a block-diagonal
        # transition and learn to open leakage through a scalar gate.
        dest = torch.arange(self.n_slots).view(self.n_slots, 1)
        cross_mask = (source_index != dest) & source_mask
        self.register_buffer("cross_mask", cross_mask, persistent=False)
        self.has_cross = bool(cross_mask.any())
        if self.has_cross and config.cascade_init == "block_diagonal":
            # The default logit -10 gives sigmoid(-10) ~= 4.5e-5: cross-source
            # logits are effectively masked at initialization, but the gate can
            # open during training.  ``cascade_leakage_init`` controls how
            # tightly closed the gate starts.
            self.leakage_logit = nn.Parameter(
                torch.tensor(float(config.cascade_leakage_init))
            )
        else:
            self.leakage_logit = None

        self.slot_emb = nn.Parameter(torch.randn(self.n_slots, config.d_model) * 0.02)
        self.out_size = self.n_blocks * self.block_size * self.n_sources * self.block_size
        self._needs_projection = not (self.state_type == "diagonal" and self.norm == "softmax")
        if self._needs_projection:
            self.proj = nn.Linear(config.d_model, self.out_size)
            nn.init.normal_(self.proj.weight, std=0.1)
            nn.init.zeros_(self.proj.bias)
            self.slot_bias = nn.Parameter(torch.zeros(self.n_slots, self.out_size))
            self._init_slot_bias()
        else:
            self.proj = None
            self.slot_bias = None

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _build_sources(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.n_slots
        if self.state_type in {"diagonal", "block_diagonal"}:
            idx = torch.arange(n).view(n, 1)
            mask = torch.ones(n, 1, dtype=torch.bool)
        elif self.state_type == "cascade_chain":
            idx = torch.zeros(n, 2, dtype=torch.long)
            mask = torch.zeros(n, 2, dtype=torch.bool)
            idx[:, 0] = torch.arange(n)  # self
            mask[:, 0] = True
            if n > 1:
                idx[1:, 1] = torch.arange(n - 1)  # previous slot
                mask[1:, 1] = True
        elif self.state_type == "cascade_full":
            idx = torch.arange(n).view(1, n).repeat(n, 1)
            mask = torch.arange(n).view(1, n) <= torch.arange(n).view(n, 1)
        elif self.state_type == "full":
            idx = torch.arange(n).view(1, n).repeat(n, 1)
            mask = torch.ones(n, n, dtype=torch.bool)
        else:
            raise ValueError(f"unknown state_type: {self.state_type}")
        return idx, mask

    def _init_slot_bias(self) -> None:
        """Start near the identity transition, not a uniform average.

        A row of zero logits gives uniform attention over sources, which washes
        the state out.  Putting a positive bias on the self-diagonal entries makes
        the initial transition close to identity, so the model starts with memory
        and learns to route from there.
        """
        if self.slot_bias is None:
            return
        # The raw bias is scaled by temperature so the effective diagonal logit
        # is 2.0 regardless of the chosen sharpness.
        init = float(self.config.transition_temperature) * 2.0
        with torch.no_grad():
            bias = self.slot_bias.view(
                self.n_slots, self.n_blocks, self.block_size, self.n_sources, self.block_size
            )
            for i in range(self.n_slots):
                for s in range(self.n_sources):
                    if not bool(self.source_mask[i, s]) or int(self.source_index[i, s]) != i:
                        continue
                    for r in range(self.n_blocks):
                        for c in range(self.block_size):
                            bias[i, r, c, s, c] = init

    # ------------------------------------------------------------------
    # Transition computation
    # ------------------------------------------------------------------
    def _gate_cross_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Attenuate cross-slot logits when block-diagonal initialisation is on.

        The self-slot logits are untouched.  For softmax transitions the closed
        cross logits are set to a large negative value; for signed-L1 transitions
        they are set to zero.  Both are differentiable in the scalar gate.
        """
        if self.leakage_logit is None:
            return logits
        view = logits.view(
            *logits.shape[:-1],
            self.n_sources,
            self.block_size,
        )
        prefix = view.shape[:-5]
        mask = self.cross_mask.view(
            *([1] * len(prefix)),
            self.n_slots,
            1,
            1,
            self.n_sources,
            1,
        )
        leak = torch.sigmoid(self.leakage_logit)
        if self.norm == "softmax":
            gated = torch.where(mask, leak * view + (1.0 - leak) * (-10.0), view)
        else:
            gated = torch.where(mask, leak * view, view)
        return gated.reshape(logits.shape)

    def project(self, z: torch.Tensor) -> torch.Tensor | None:
        """Map input features to block-row logits.

        ``z`` may have any leading shape, e.g. ``(B, d_model)`` for one step or
        ``(B, C, d_model)`` for a chunk.  The output is
        ``(..., L, n_blocks, b, S·b)``.
        """
        if not self._needs_projection:
            return None
        # The caller has already added the per-slot embedding, so ``z`` has
        # shape (..., L, d_model) here.
        logits = self.proj(z) + self.slot_bias
        logits = logits.view(
            *logits.shape[:-1],
            self.n_blocks,
            self.block_size,
            self.n_sources * self.block_size,
        )
        return self._gate_cross_logits(logits)

    def normalize(self, logits: torch.Tensor | None) -> torch.Tensor | None:
        """Turn block-row logits into a transition matrix."""
        if logits is None:
            return None
        leading = logits.shape[:-4]  # prefix before L
        reshaped = logits.view(
            *leading,
            self.n_slots,
            self.n_blocks,
            self.block_size,
            self.n_sources,
            self.block_size,
        )
        # One leading singleton per prefix dimension, then L, n_blocks, b, S, b.
        mask = self.source_mask.view(
            *([1] * len(leading)),
            self.n_slots,
            1,
            1,
            self.n_sources,
            1,
        )

        if self.norm == "softmax":
            reshaped = reshaped.masked_fill(~mask, float("-inf"))
            flat = reshaped.reshape(
                *leading,
                self.n_slots,
                self.n_blocks,
                self.block_size,
                self.n_sources * self.block_size,
            )
            return torch.softmax(flat / self.temperature, dim=-1)

        if self.norm == "signed_l1":
            activated = torch.tanh(reshaped / self.temperature)
            activated = activated * mask.to(activated.dtype)
            flat = activated.reshape(
                *leading,
                self.n_slots,
                self.n_blocks,
                self.block_size,
                self.n_sources * self.block_size,
            )
            denom = flat.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
            return flat / denom

        raise ValueError(f"unknown transition norm: {self.norm}")

    def mix(self, transition: torch.Tensor | None, prev: torch.Tensor) -> torch.Tensor:
        """Apply one transition block row to the previous state.

        ``prev`` is ``(B, L, d)``; the result is the same shape.  When the
        transition is the identity special case (diagonal + softmax), this is
        just ``prev``.
        """
        if transition is None:
            return prev

        batch = prev.shape[0]
        h = prev.view(batch, self.n_slots, self.n_blocks, self.block_size)
        # Gather the source slots for every destination, then zero the padding.
        gathered = h[:, self.source_index]  # (B, L, S, n_blocks, b)
        gathered = gathered * self.source_mask.view(1, self.n_slots, self.n_sources, 1, 1).to(
            gathered.dtype
        )
        gathered = gathered.permute(0, 1, 3, 2, 4).reshape(
            batch, self.n_slots, self.n_blocks, self.n_sources * self.block_size
        )
        mixed = torch.einsum("blnij,blnj->blni", transition, gathered)
        return mixed.reshape(batch, self.n_slots, self.slot_dim)

    def to_dense(self, transition: torch.Tensor | None) -> torch.Tensor:
        """Expand a single-step block transition into a dense ``(B, Ld, Ld)``.

        Used by tests and diagnostics; not on the training path.
        """
        batch = 1 if transition is None else transition.shape[0]
        total = self.n_slots * self.slot_dim
        dense = self.slot_emb.new_zeros(batch, self.n_slots, self.slot_dim, total)
        if transition is None:
            for i in range(self.n_slots):
                dense[:, i, :, i * self.slot_dim : (i + 1) * self.slot_dim] = torch.eye(
                    self.slot_dim, device=dense.device, dtype=dense.dtype
                )
            return dense.reshape(batch, total, total)

        rows = transition.view(
            batch,
            self.n_slots,
            self.n_blocks,
            self.block_size,
            self.n_sources,
            self.block_size,
        )
        for i in range(self.n_slots):
            for r in range(self.n_blocks):
                for c in range(self.block_size):
                    for s in range(self.n_sources):
                        if not bool(self.source_mask[i, s]):
                            continue
                        j = int(self.source_index[i, s])
                        start = j * self.slot_dim + r * self.block_size
                        dense[:, i, r * self.block_size + c, start : start + self.block_size] = (
                            rows[:, i, r, c, s, :]
                        )
        return dense.reshape(batch, total, total)


class RecurrentLayer(nn.Module):
    """One gated recurrent layer with a block-structured transition."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_slots = config.n_slots
        self.slot_dim = config.slot_dim
        self.state_dim = self.n_slots * self.slot_dim
        self.transition = BlockTransition(config)

        if config.gate_mode == "input":
            self.gate = nn.Linear(config.d_model, self.slot_dim)
            nn.init.normal_(self.gate.weight, std=0.02)
            nn.init.constant_(self.gate.bias, config.gate_bias_init)
            self.write = nn.Linear(config.d_model, self.slot_dim)
        else:
            self.gate = None
            self.write = None
        self.out_proj = nn.Linear(self.state_dim, config.d_model)
        self.write_activation = config.write_activation

        if config.initial_state == "learned":
            self.initial_state = nn.Parameter(torch.randn(self.n_slots, self.slot_dim) * 0.02)
        else:
            self.register_buffer(
                "initial_state", torch.zeros(self.n_slots, self.slot_dim), persistent=False
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the layer over a sequence.

        Returns
        -------
        states : (B, T, L, d)
            The state at every position.
        output : (B, T, d_model)
            A projection of the flattened state, used for stacking layers.
        """
        batch, length, _ = x.shape
        states = x.new_empty(batch, length, self.n_slots, self.slot_dim)
        # A learned initial state gives the model a position-free way to start
        # from an identity-like representation.  With a zero start, a gated
        # convex update cannot both initialise on token 0 and retain afterwards
        # without positional information.
        state = self.initial_state.unsqueeze(0).expand(batch, -1, -1).clone()
        chunk_size = self.config.transition_chunk_size

        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            chunk = x[:, start:end]
            z = chunk.unsqueeze(2) + self.transition.slot_emb  # (B, C, L, d_model)
            logits = self.transition.project(z)  # (B, C, L, n_blocks, b, S·b)
            transition = self.transition.normalize(logits)
            if self.gate is not None:
                gates = torch.sigmoid(self.gate(z))  # (B, C, L, d)
                writes = self.write(z)
                if self.write_activation == "tanh":
                    writes = torch.tanh(writes)
            else:
                gates = None
                writes = None

            for offset in range(end - start):
                t = start + offset
                mixed = self.transition.mix(
                    None if transition is None else transition[:, offset], state
                )
                if gates is None:
                    state = self.config.constant_gate * mixed
                else:
                    g = gates[:, offset]
                    w = writes[:, offset]
                    state = g * mixed + (1.0 - g) * w
                states[:, t] = state

        flat = states.reshape(batch, length, self.state_dim)
        output = self.out_proj(flat)
        return states, output
