"""Readout families.

The scientific contrast is:

* ``linear`` / ``input_linear`` / ``gated_linear``: the readout weights are a
  function of the input only, never of the state.  ``input_linear`` implements
  the low-rank form of the standard ``C_t H_t`` family; ``gated_linear`` is a
  strong diagonal version of the same class.
* ``mlp``: nonlinear, but still state-only.
* ``attention``: the attention weights depend on the **state** through the keys,
  so this is the readout class that the separation argument says is strictly
  larger at a fixed state.

Every readout returns ``(logits, aux)``.  ``aux`` carries detached diagnostics
(attention entropy, top weight) for analysis; training ignores it.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..config import ModelConfig

Aux = dict[str, torch.Tensor]


class Readout(nn.Module):
    out_dim: int

    def forward(self, states: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, Aux]:
        raise NotImplementedError


class StateLinearReadout(Readout):
    """``y_t = W H_t``: no input dependence at all."""

    def __init__(self, state_dim: int, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Linear(state_dim, out_dim)

    def forward(self, states: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, Aux]:
        batch, length = states.shape[:2]
        flat = states.reshape(batch, length, -1)
        return self.proj(flat), {}


class InputLinearReadout(Readout):
    """Low-rank ``C(x_t) H_t`` with ``C(x) = A(x) B``.

    With ``rank > 0`` this is a rank-limited member of the input-dependent linear
    readout family.  With ``rank == 0`` the full ``out_dim × state_dim`` matrix is
    materialised per token; that is faithful but memory-hungry and should be used
    only for small state/output sizes.
    """

    def __init__(
        self,
        state_dim: int,
        d_model: int,
        out_dim: int,
        rank: int,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.rank = rank
        self.state_dim = state_dim
        if rank > 0:
            self.state_proj = nn.Linear(state_dim, rank, bias=False)
            self.input_proj = nn.Linear(d_model, out_dim * rank, bias=False)
        else:
            self.state_proj = nn.Identity()
            self.input_proj = nn.Linear(d_model, out_dim * state_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.normal_(self.input_proj.weight, std=0.02)

    def forward(self, states: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, Aux]:
        batch, length = states.shape[:2]
        flat = states.reshape(batch, length, -1)
        if self.rank > 0:
            h = self.state_proj(flat)  # (B, T, r)
            a = self.input_proj(x).view(batch, length, self.out_dim, self.rank)
            out = torch.einsum("btor,btr->bto", a, h)
        else:
            a = self.input_proj(x).view(batch, length, self.out_dim, self.state_dim)
            out = torch.einsum("btod,btd->bto", a, flat)
        return out + self.bias, {}


class GatedLinearReadout(Readout):
    """``y_t = W_out(g(x_t) ⊙ W_s H_t)``.

    The gate depends on the input but not on the state, so at fixed ``x`` this is
    linear in ``H``.  It is the strongest fair member of the linear readout family
    and the key control for "is attention doing anything beyond input gating?".
    """

    def __init__(
        self,
        state_dim: int,
        d_model: int,
        out_dim: int,
        hidden: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.state_proj = nn.Linear(state_dim, hidden, bias=False)
        self.gate = nn.Linear(d_model, hidden)
        self.out = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, Aux]:
        batch, length = states.shape[:2]
        flat = states.reshape(batch, length, -1)
        h = self.state_proj(flat)
        g = torch.sigmoid(self.gate(x))
        return self.out(self.dropout(g * h)), {}


class MLPReadout(Readout):
    """A two-layer MLP over the flattened state (nonlinear in ``H``)."""

    def __init__(
        self,
        state_dim: int,
        out_dim: int,
        hidden: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, states: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, Aux]:
        batch, length = states.shape[:2]
        return self.net(states.reshape(batch, length, -1)), {}


class AttentionReadout(Readout):
    """Depth attention over the ``L`` state slots.

    ``Q`` depends on the input, ``K`` and ``V`` on the state.  This is the only
    readout here whose mixing weights are state-dependent.  ``query_mode``
    provides the controls: ``constant`` removes input dependence and ``random``
    removes content-based routing while keeping the same parameter count.
    """

    def __init__(
        self,
        n_slots: int,
        slot_dim: int,
        d_model: int,
        out_dim: int,
        hidden: int,
        temperature: float = 1.0,
        slot_bias: bool = True,
        query_mode: str = "content",
        output_gate: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.n_slots = n_slots
        self.hidden = hidden
        self.temperature = temperature
        self.query_mode = query_mode

        self.q_proj = nn.Linear(d_model, hidden, bias=False)
        self.k_proj = nn.Linear(slot_dim, hidden, bias=False)
        self.v_proj = nn.Linear(slot_dim, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

        if query_mode == "constant":
            self.const_q = nn.Parameter(torch.randn(hidden) * 0.02)
        else:
            self.const_q = None

        if slot_bias:
            self.slot_bias = nn.Parameter(torch.zeros(n_slots))
        else:
            self.slot_bias = None

        if output_gate:
            self.out_gate = nn.Linear(d_model, hidden)
            nn.init.zeros_(self.out_gate.bias)
        else:
            self.out_gate = None

    def forward(self, states: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, Aux]:
        batch, length, n_slots, _ = states.shape
        if n_slots != self.n_slots:
            raise ValueError("state slot count does not match the readout")
        if self.query_mode == "content":
            q = self.q_proj(x)  # (B, T, hidden)
        elif self.query_mode == "constant":
            q = self.const_q.view(1, 1, -1).expand(batch, length, -1)
        elif self.query_mode == "random":
            q = torch.randn(batch, length, self.hidden, device=states.device, dtype=states.dtype)
        else:
            raise ValueError(f"unknown query mode: {self.query_mode}")

        k = self.k_proj(states)  # (B, T, L, hidden)
        v = self.v_proj(states)  # (B, T, L, hidden)
        scores = torch.einsum("btd,btld->btl", q, k) / math.sqrt(self.hidden)
        scores = scores / self.temperature
        if self.slot_bias is not None:
            scores = scores + self.slot_bias.view(1, 1, -1)

        weights = torch.softmax(scores, dim=-1)
        mixed = torch.einsum("btl,btld->btd", weights, v)
        mixed = self.dropout(mixed)
        if self.out_gate is not None:
            mixed = mixed * torch.sigmoid(self.out_gate(x))
        out = self.out_proj(mixed)

        with torch.no_grad():
            entropy = -(weights * (weights + 1e-9).log()).sum(dim=-1).mean()
            max_weight = weights.max(dim=-1).values.mean()
        return out, {"attn_entropy": entropy, "attn_max": max_weight}


class HybridReadout(Readout):
    """Gate between depth attention and the gated linear mixer.

    ``y = σ(a(x)) ⊙ attention(H, x) + (1 − σ(a(x))) ⊙ gated_linear(H, x)``.
    The attention branch must earn its place; the linear branch is a strong
    parameter-matched fallback.
    """

    def __init__(
        self,
        n_slots: int,
        slot_dim: int,
        d_model: int,
        out_dim: int,
        hidden: int,
        temperature: float = 1.0,
        slot_bias: bool = True,
        query_mode: str = "content",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.attention = AttentionReadout(
            n_slots=n_slots,
            slot_dim=slot_dim,
            d_model=d_model,
            out_dim=out_dim,
            hidden=hidden,
            temperature=temperature,
            slot_bias=slot_bias,
            query_mode=query_mode,
            output_gate=True,
            dropout=dropout,
        )
        self.linear = GatedLinearReadout(
            state_dim=n_slots * slot_dim,
            d_model=d_model,
            out_dim=out_dim,
            hidden=hidden,
            dropout=dropout,
        )
        self.gate = nn.Linear(d_model, out_dim)
        nn.init.zeros_(self.gate.bias)

    def forward(self, states: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, Aux]:
        attn_out, aux = self.attention(states, x)
        lin_out, _ = self.linear(states, x)
        gate = torch.sigmoid(self.gate(x))
        return gate * attn_out + (1.0 - gate) * lin_out, aux


def _count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


class ResidualLinearAdapter(nn.Module):
    """A low-rank residual linear map on the flattened state.

    It starts as the identity (the ``up`` projection is zero), so adding it does
    not destroy a trained or freshly initialised readout.  It is a linear map of
    the state, hence still inside the linear readout family for the purpose of the
    readout-vs-recurrence comparison.
    """

    def __init__(self, state_dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(state_dim, rank, bias=False)
        self.up = nn.Linear(rank, state_dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        batch, length, n_slots, slot_dim = states.shape
        flat = states.reshape(batch, length, -1)
        delta = self.up(self.down(flat)).reshape(batch, length, n_slots, slot_dim)
        return states + delta


class AdaptedReadout(Readout):
    """Wrap a readout with a residual linear adapter to a parameter budget."""

    def __init__(self, readout: Readout, state_dim: int, rank: int) -> None:
        super().__init__()
        self.readout = readout
        self.adapter = ResidualLinearAdapter(state_dim, rank)
        self.out_dim = readout.out_dim

    def forward(self, states: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, Aux]:
        return self.readout(self.adapter(states), x)


def maybe_adapt_readout(readout: Readout, state_dim: int, budget: int) -> Readout:
    """Add a residual linear adapter if the budget allows a rank-1 map."""
    if budget <= 0:
        return readout
    current = _count_parameters(readout)
    remaining = budget - current
    if remaining < 2 * state_dim:
        return readout
    rank = min(state_dim, remaining // (2 * state_dim))
    return AdaptedReadout(readout, state_dim, rank)


def build_readout(
    config: ModelConfig,
    d_model: int,
    n_slots: int,
    slot_dim: int,
    out_dim: int,
) -> Readout:
    state_dim = n_slots * slot_dim
    if config.readout == "linear":
        return StateLinearReadout(state_dim, out_dim)
    if config.readout == "input_linear":
        return InputLinearReadout(state_dim, d_model, out_dim, rank=config.readout_rank)
    if config.readout == "gated_linear":
        return GatedLinearReadout(
            state_dim,
            d_model,
            out_dim,
            hidden=config.readout_hidden,
            dropout=config.readout_dropout,
        )
    if config.readout == "mlp":
        return MLPReadout(
            state_dim,
            out_dim,
            hidden=config.readout_hidden,
            dropout=config.readout_dropout,
        )
    if config.readout == "attention":
        return AttentionReadout(
            n_slots=n_slots,
            slot_dim=slot_dim,
            d_model=d_model,
            out_dim=out_dim,
            hidden=config.readout_hidden,
            temperature=config.attention_temperature,
            slot_bias=config.attention_slot_bias,
            query_mode=config.attention_query_mode,
            output_gate=config.attention_output_gate,
            dropout=config.readout_dropout,
        )
    if config.readout == "hybrid":
        return HybridReadout(
            n_slots=n_slots,
            slot_dim=slot_dim,
            d_model=d_model,
            out_dim=out_dim,
            hidden=config.readout_hidden,
            temperature=config.attention_temperature,
            slot_bias=config.attention_slot_bias,
            query_mode=config.attention_query_mode,
            dropout=config.readout_dropout,
        )
    raise ValueError(f"unknown readout: {config.readout}")
