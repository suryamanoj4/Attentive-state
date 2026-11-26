"""Full model: token encoder, stacked recurrent layers, and a readout."""

from __future__ import annotations

from torch import nn

from ..config import ModelConfig
from .layers import RecurrentLayer, TokenEncoder
from .readouts import build_readout, maybe_adapt_readout


class RecurrentModel(nn.Module):
    """A small gated state-space model with a pluggable readout."""

    def __init__(self, config: ModelConfig, vocab_size: int, out_dim: int) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.out_dim = out_dim

        self.encoder = TokenEncoder(vocab_size, config)
        self.layers = nn.ModuleList([RecurrentLayer(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model) if config.layer_norm else nn.Identity()
        self.readout = maybe_adapt_readout(
            build_readout(
                config,
                d_model=config.d_model,
                n_slots=config.n_slots,
                slot_dim=config.slot_dim,
                out_dim=out_dim,
            ),
            state_dim=config.n_slots * config.slot_dim,
            budget=config.readout_param_budget,
        )

    def forward(self, input_ids):
        """Return ``(logits, aux)`` with logits of shape ``(B, T, out_dim)``.

        The readout's input-side features are the **token encoder output**, not
        the post-layer residual.  The residual contains a projection of the
        recurrent state; feeding it to the readout would make even the
        ``gated_linear`` / ``input_linear`` weights state-dependent through their
        input argument, which would break the readout-vs-recurrence separation.
        Only the attention keys/values (and the readout's state projection) see
        the state.
        """
        readout_input = self.encoder(input_ids)
        x = readout_input
        states = None
        for layer in self.layers:
            states, layer_out = layer(x)
            x = self.norm(x + layer_out)
        logits, aux = self.readout(states, readout_input)
        return logits, aux


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


def parameter_breakdown(model: RecurrentModel) -> dict[str, int]:
    breakdown = {
        "encoder": count_parameters(model.encoder),
        "recurrence": count_parameters(model.layers),
        "norm": count_parameters(model.norm),
        "readout": count_parameters(model.readout),
    }
    breakdown["total"] = count_parameters(model)
    return breakdown


def build_model(config: ModelConfig, vocab_size: int, out_dim: int) -> RecurrentModel:
    return RecurrentModel(config, vocab_size=vocab_size, out_dim=out_dim)


__all__ = [
    "RecurrentModel",
    "build_model",
    "count_parameters",
    "parameter_breakdown",
]
