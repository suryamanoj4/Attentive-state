"""Configuration dataclasses, YAML loading, and dotted-key overrides.

The config is intentionally explicit: every run is described by a small nested
dataclass tree that can be dumped to YAML and restored exactly.  A run directory
contains a copy of this YAML, so an experiment is always self-describing.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml

STATE_TYPES = {
    "diagonal",
    "block_diagonal",
    "cascade_chain",
    "cascade_full",
    "full",
}
READOUTS = {
    "linear",
    "input_linear",
    "gated_linear",
    "mlp",
    "attention",
    "hybrid",
}
TRANSITION_NORMS = {"softmax", "signed_l1"}
QUERY_MODES = {"content", "constant", "random"}
TASK_NAMES = {"mqar", "group", "a5", "s5", "parity", "copy"}
GROUP_NAMES = {"A5", "S5", "Z2"}
SCHEDULES = {"constant", "linear", "cosine"}


@dataclass
class MQARConfig:
    """Multi-query associative recall.

    Keys and values are separate token ids.  A pair is two adjacent tokens
    ``(key, value)``.  Distractors are extra *unqueried* pairs, which is the
    interference axis: ``n_distractors`` counts extra pairs, not extra tokens.
    """

    n_keys: int = 64
    n_values: int = 64
    n_pairs: int = 8
    n_queries: int = 8
    n_distractors: int = 0
    query_marker: bool = True
    shuffle_pairs: bool = True

    # Evaluation grid.  The task exposes these as named eval specs.
    eval_pairs: list[int] = field(default_factory=lambda: [8, 16, 32, 64])
    eval_distractors: list[int] = field(default_factory=lambda: [0, 16, 64])

    def validate(self) -> None:
        if self.n_pairs < 1:
            raise ValueError("mqar.n_pairs must be >= 1")
        if self.n_queries < 1:
            raise ValueError("mqar.n_queries must be >= 1")
        if self.n_queries > self.n_pairs:
            raise ValueError("mqar.n_queries must be <= mqar.n_pairs")
        if self.n_keys < self.n_pairs:
            raise ValueError("mqar.n_keys must be >= mqar.n_pairs")
        if self.n_values < self.n_pairs:
            raise ValueError("mqar.n_values must be >= mqar.n_pairs")
        if self.n_distractors < 0:
            raise ValueError("mqar.n_distractors must be >= 0")
        if any(p < 1 for p in self.eval_pairs):
            raise ValueError("mqar.eval_pairs must all be >= 1")
        if any(p > self.n_keys or p > self.n_values for p in self.eval_pairs):
            raise ValueError("mqar.eval_pairs cannot exceed the key/value vocabulary sizes")
        if any(d < 0 for d in self.eval_distractors):
            raise ValueError("mqar.eval_distractors must all be >= 0")


@dataclass
class GroupConfig:
    """Cumulative group multiplication (A5, S5, or parity / Z2)."""

    group: str = "A5"
    train_min_len: int = 3
    train_max_len: int = 20
    eval_lengths: list[int] = field(
        default_factory=lambda: [20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
    )

    def validate(self) -> None:
        if self.group not in GROUP_NAMES:
            raise ValueError(f"group.group must be one of {sorted(GROUP_NAMES)}")
        if self.train_min_len < 1:
            raise ValueError("group.train_min_len must be >= 1")
        if self.train_max_len < self.train_min_len:
            raise ValueError("group.train_max_len must be >= group.train_min_len")
        if any(length < 1 for length in self.eval_lengths):
            raise ValueError("group.eval_lengths must all be >= 1")


@dataclass
class CopyConfig:
    """Copying: reproduce a random content sequence after a delimiter."""

    content_vocab: int = 16
    train_min_len: int = 8
    train_max_len: int = 32
    eval_lengths: list[int] = field(default_factory=lambda: [16, 32, 64, 128, 256, 512])

    def validate(self) -> None:
        if self.content_vocab < 1:
            raise ValueError("copy.content_vocab must be >= 1")
        if self.train_min_len < 1:
            raise ValueError("copy.train_min_len must be >= 1")
        if self.train_max_len < self.train_min_len:
            raise ValueError("copy.train_max_len must be >= copy.train_min_len")
        if any(length < 1 for length in self.eval_lengths):
            raise ValueError("copy.eval_lengths must all be >= 1")


@dataclass
class TaskConfig:
    name: str = "mqar"
    mqar: MQARConfig = field(default_factory=MQARConfig)
    group: GroupConfig = field(default_factory=GroupConfig)
    copy: CopyConfig = field(default_factory=CopyConfig)

    def validate(self) -> None:
        if self.name not in TASK_NAMES:
            raise ValueError(f"task.name must be one of {sorted(TASK_NAMES)}")
        if self.name == "mqar":
            self.mqar.validate()
        elif self.name in {"group", "a5", "s5", "parity"}:
            self.group.validate()
        elif self.name == "copy":
            self.copy.validate()


@dataclass
class ModelConfig:
    """The recurrent core and the readout.

    ``n_slots * slot_dim`` is the total state width.  ``block_size`` is the dense
    block size inside a slot.  For ``state_type='diagonal'`` the block size is
    forced to 1.
    """

    d_model: int = 128
    n_layers: int = 1

    # State geometry.
    n_slots: int = 4
    slot_dim: int = 32
    block_size: int = 4
    state_type: str = "diagonal"
    transition_norm: str = "softmax"
    transition_temperature: float = 1.0
    transition_chunk_size: int = 64

    # Input encoder.
    token_conv_kernel: int = 1
    token_mlp_hidden: int = 0
    dropout: float = 0.0

    # Recurrence gates.  ``gate_mode='constant'`` removes the input-dependent
    # gate and the write entirely; the recurrence is then a pure transition
    # H_t = g A_t H_{t-1} with a learned initial state.  This isolates the
    # transition on the group-tracking tasks.
    gate_mode: str = "input"  # input | constant
    constant_gate: float = 1.0
    gate_bias_init: float = 2.0
    write_activation: str = "none"  # none | tanh
    layer_norm: bool = True
    initial_state: str = "learned"  # zeros | learned

    # Readout.
    readout: str = "gated_linear"
    readout_hidden: int = 64
    readout_rank: int = 16
    readout_dropout: float = 0.0
    # If > 0, a residual linear adapter is added before the readout until the
    # readout parameter count is as close to this budget as the adapter rank
    # allows.  Used by ``sweep --match-params``.
    readout_param_budget: int = 0
    attention_heads: int = 1
    attention_temperature: float = 1.0
    attention_slot_bias: bool = True
    attention_query_mode: str = "content"
    attention_output_gate: bool = True

    def __post_init__(self) -> None:
        # The diagonal state type is the block_size=1 special case.  Normalising
        # here lets a single base config sweep state_type without hand-editing
        # block_size for every variant.
        if self.state_type == "diagonal":
            self.block_size = 1

    def validate(self) -> None:
        if self.d_model < 1:
            raise ValueError("model.d_model must be >= 1")
        if self.n_layers < 1:
            raise ValueError("model.n_layers must be >= 1")
        if self.n_slots < 1:
            raise ValueError("model.n_slots must be >= 1")
        if self.slot_dim < 1:
            raise ValueError("model.slot_dim must be >= 1")
        if self.state_type not in STATE_TYPES:
            raise ValueError(f"model.state_type must be one of {sorted(STATE_TYPES)}")
        if self.transition_norm not in TRANSITION_NORMS:
            raise ValueError(f"model.transition_norm must be one of {sorted(TRANSITION_NORMS)}")
        if self.state_type == "diagonal":
            if self.block_size != 1:
                raise ValueError("diagonal state_type requires block_size == 1")
        elif self.block_size < 1 or self.slot_dim % self.block_size != 0:
            raise ValueError("model.block_size must divide model.slot_dim")
        if self.readout not in READOUTS:
            raise ValueError(f"model.readout must be one of {sorted(READOUTS)}")
        if self.readout_hidden < 1:
            raise ValueError("model.readout_hidden must be >= 1")
        if self.readout_rank < 0:
            raise ValueError("model.readout_rank must be >= 0")
        if self.readout_param_budget < 0:
            raise ValueError("model.readout_param_budget must be >= 0")
        if self.attention_heads != 1:
            raise ValueError("only one attention head is implemented for now")
        if self.attention_query_mode not in QUERY_MODES:
            raise ValueError(f"model.attention_query_mode must be one of {sorted(QUERY_MODES)}")
        if self.token_conv_kernel < 1:
            raise ValueError("model.token_conv_kernel must be >= 1")
        if self.token_mlp_hidden < 0:
            raise ValueError("model.token_mlp_hidden must be >= 0")
        if self.transition_chunk_size < 1:
            raise ValueError("model.transition_chunk_size must be >= 1")
        if self.write_activation not in {"none", "tanh"}:
            raise ValueError("model.write_activation must be 'none' or 'tanh'")
        if self.initial_state not in {"zeros", "learned"}:
            raise ValueError("model.initial_state must be 'zeros' or 'learned'")
        if self.gate_mode not in {"input", "constant"}:
            raise ValueError("model.gate_mode must be 'input' or 'constant'")
        if not 0.0 <= self.constant_gate <= 1.0:
            raise ValueError("model.constant_gate must be in [0, 1]")


@dataclass
class TrainConfig:
    steps: int = 10_000
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 100
    schedule: str = "cosine"
    min_lr_ratio: float = 0.0

    log_every: int = 50
    eval_every: int = 500
    eval_batches: int = 8
    eval_batch_size: int = 64

    amp: bool = False
    deterministic: bool = False
    num_threads: int = 0

    save_checkpoint: bool = True
    checkpoint_every: int = 1_000

    def validate(self) -> None:
        if self.steps < 1:
            raise ValueError("train.steps must be >= 1")
        if self.batch_size < 1:
            raise ValueError("train.batch_size must be >= 1")
        if self.lr <= 0:
            raise ValueError("train.lr must be > 0")
        if self.grad_clip < 0:
            raise ValueError("train.grad_clip must be >= 0")
        if self.warmup_steps < 0:
            raise ValueError("train.warmup_steps must be >= 0")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"train.schedule must be one of {sorted(SCHEDULES)}")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("train.min_lr_ratio must be in [0, 1]")
        if self.log_every < 1:
            raise ValueError("train.log_every must be >= 1")
        if self.eval_every < 1:
            raise ValueError("train.eval_every must be >= 1")
        if self.eval_batches < 1:
            raise ValueError("train.eval_batches must be >= 1")
        if self.eval_batch_size < 1:
            raise ValueError("train.eval_batch_size must be >= 1")
        if self.checkpoint_every < 1:
            raise ValueError("train.checkpoint_every must be >= 1")


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    seed: int = 0
    device: str = "auto"
    output_dir: str = "runs"
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        self.task.validate()
        self.model.validate()
        self.train.validate()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentConfig:
        cfg = _build_dataclass(cls, dict(data))
        cfg.validate()
        return cfg


def _build_dataclass(cls: type, data: Mapping[str, Any]) -> Any:
    """Recursively build a dataclass tree from a mapping.

    Unknown keys are rejected.  This is important for sweep overrides: a typo
    should fail loudly instead of silently doing nothing.
    """
    if not dataclasses.is_dataclass(cls):
        return data
    hints = get_type_hints(cls)
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        typ = hints[name]
        if dataclasses.is_dataclass(typ) and isinstance(value, Mapping):
            value = _build_dataclass(typ, value)
        kwargs[name] = value
    return cls(**kwargs)


def parse_scalar(text: str) -> Any:
    """Parse a CLI override value as YAML, falling back to a string."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_override(data: dict[str, Any], override: str) -> None:
    """Apply ``a.b.c=value`` to a nested config dictionary in place."""
    if "=" not in override:
        raise ValueError(f"override must be key=value, got {override!r}")
    key, raw_value = override.split("=", 1)
    parts = key.split(".")
    node: dict[str, Any] = data
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise KeyError(f"unknown config section in override: {key!r}")
        node = node[part]
    node[parts[-1]] = parse_scalar(raw_value)


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    for override in overrides or []:
        apply_override(data, override)
    return ExperimentConfig.from_dict(data)


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False)
