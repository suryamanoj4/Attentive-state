"""Model components."""

from __future__ import annotations

from .layers import BlockTransition, RecurrentLayer, TokenEncoder
from .model import RecurrentModel, build_model, count_parameters, parameter_breakdown
from .readouts import (
    AttentionReadout,
    GatedLinearReadout,
    HybridReadout,
    InputLinearReadout,
    MLPReadout,
    StateLinearReadout,
    build_readout,
)

__all__ = [
    "AttentionReadout",
    "BlockTransition",
    "GatedLinearReadout",
    "HybridReadout",
    "InputLinearReadout",
    "MLPReadout",
    "RecurrentLayer",
    "RecurrentModel",
    "StateLinearReadout",
    "TokenEncoder",
    "build_model",
    "build_readout",
    "count_parameters",
    "parameter_breakdown",
]
