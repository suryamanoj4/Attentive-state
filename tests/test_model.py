from __future__ import annotations

import pytest
import torch

from readout_recurrence.config import ExperimentConfig
from readout_recurrence.data import build_task
from readout_recurrence.models import (
    build_model,
    count_parameters,
    parameter_breakdown,
)
from readout_recurrence.models.layers import BlockTransition

STATE_TYPES = ["diagonal", "block_diagonal", "cascade_chain", "cascade_full", "full"]
READOUTS = [
    "linear",
    "input_linear",
    "gated_linear",
    "mlp",
    "attention",
    "hybrid",
    "slot_gated",
    "slot_max",
]


def make_config(state_type: str, readout: str) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "name": "test",
            "task": {
                "name": "mqar",
                "mqar": {
                    "n_keys": 8,
                    "n_values": 8,
                    "n_pairs": 3,
                    "n_queries": 3,
                    "n_distractors": 1,
                    "eval_pairs": [3],
                    "eval_distractors": [0],
                },
            },
            "model": {
                "d_model": 16,
                "n_layers": 1,
                "n_slots": 2,
                "slot_dim": 8,
                "block_size": 2,
                "state_type": state_type,
                "transition_chunk_size": 4,
                "token_conv_kernel": 2,
                "readout": readout,
                "readout_hidden": 8,
                "readout_rank": 4,
            },
            "train": {"steps": 1, "batch_size": 2},
        }
    )


@pytest.mark.parametrize("state_type", STATE_TYPES)
@pytest.mark.parametrize("readout", READOUTS)
def test_forward_backward_all_variants(state_type: str, readout: str):
    cfg = make_config(state_type, readout)
    task = build_task(cfg.task, seed=0)
    model = build_model(cfg.model, task.vocab_size, task.out_dim)
    batch = task.sample(2)
    logits, aux = model(batch.input_ids)
    assert logits.shape == (2, batch.input_ids.shape[1], task.out_dim)
    assert torch.isfinite(logits).all()
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, task.out_dim),
        batch.labels.reshape(-1),
        ignore_index=-100,
    )
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_parameter_breakdown_sums_to_total():
    cfg = make_config("cascade_chain", "attention")
    task = build_task(cfg.task, seed=0)
    model = build_model(cfg.model, task.vocab_size, task.out_dim)
    breakdown = parameter_breakdown(model)
    assert breakdown["total"] == count_parameters(model)
    assert breakdown["total"] == sum(
        breakdown[key] for key in ["encoder", "recurrence", "norm", "readout"]
    )


@pytest.mark.parametrize("state_type", STATE_TYPES)
@pytest.mark.parametrize("norm", ["softmax", "signed_l1"])
def test_block_transition_matches_dense_matrix(state_type: str, norm: str):
    cfg = ExperimentConfig.from_dict(
        {
            "model": {
                "d_model": 16,
                "n_slots": 4,
                "slot_dim": 12,
                "block_size": 3,
                "state_type": state_type,
                "transition_norm": norm,
            },
            "train": {"steps": 1},
        }
    ).model
    transition = BlockTransition(cfg)
    z = torch.randn(2, 4, 16)  # (B, L, d_model)
    logits = transition.project(z)
    matrix = transition.normalize(logits)
    state = torch.randn(2, 4, 12)
    mixed = transition.mix(matrix, state)
    dense = transition.to_dense(matrix)
    expected = torch.einsum("bij,bj->bi", dense, state.reshape(2, -1)).reshape(2, 4, 12)
    assert torch.allclose(mixed, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("state_type", ["block_diagonal", "cascade_chain", "cascade_full"])
def test_transition_rows_are_normalized(state_type: str):
    cfg = ExperimentConfig.from_dict(
        {
            "model": {
                "d_model": 16,
                "n_slots": 3,
                "slot_dim": 12,
                "block_size": 3,
                "state_type": state_type,
                "transition_norm": "softmax",
            },
            "train": {"steps": 1},
        }
    ).model
    transition = BlockTransition(cfg)
    matrix = transition.normalize(transition.project(torch.randn(2, 3, 16)))
    assert matrix is not None
    assert torch.allclose(matrix.sum(dim=-1), torch.ones_like(matrix.sum(dim=-1)))


@pytest.mark.parametrize("state_type", ["cascade_chain", "cascade_full", "full"])
@pytest.mark.parametrize("norm", ["softmax", "signed_l1"])
def test_cascade_block_diagonal_init_closes_cross_slots(state_type: str, norm: str):
    cfg = ExperimentConfig.from_dict(
        {
            "model": {
                "d_model": 16,
                "n_slots": 4,
                "slot_dim": 12,
                "block_size": 3,
                "state_type": state_type,
                "transition_norm": norm,
                "cascade_init": "block_diagonal",
            },
            "train": {"steps": 1},
        }
    ).model
    transition = BlockTransition(cfg)
    assert transition.leakage_logit is not None
    matrix = transition.normalize(transition.project(torch.randn(2, 4, 16)))
    assert matrix is not None
    view = matrix.view(
        2,
        4,
        transition.n_blocks,
        transition.block_size,
        transition.n_sources,
        transition.block_size,
    )
    cross = transition.cross_mask.view(1, 4, 1, 1, transition.n_sources, 1)
    if norm == "softmax":
        cross_mass = (view * cross).sum(dim=(4, 5))
        total_mass = view.sum(dim=(4, 5)).clamp_min(1e-9)
    else:
        # Signed-L1 rows sum to one in absolute value, not in signed value.
        cross_mass = (view.abs() * cross).sum(dim=(4, 5))
        total_mass = view.abs().sum(dim=(4, 5)).clamp_min(1e-9)
    assert (cross_mass / total_mass).max().item() < 1e-3


def test_cascade_leakage_init_value():
    cfg = ExperimentConfig.from_dict(
        {
            "model": {
                "d_model": 16,
                "n_slots": 4,
                "slot_dim": 12,
                "block_size": 3,
                "state_type": "cascade_chain",
                "cascade_init": "block_diagonal",
                "cascade_leakage_init": -6.0,
            },
            "train": {"steps": 1},
        }
    ).model
    transition = BlockTransition(cfg)
    assert transition.leakage_logit is not None
    assert transition.leakage_logit.item() == pytest.approx(-6.0)


def test_cascade_source_masks():
    cfg = ExperimentConfig.from_dict(
        {
            "model": {
                "d_model": 8,
                "n_slots": 3,
                "slot_dim": 6,
                "block_size": 3,
                "state_type": "cascade_full",
            },
            "train": {"steps": 1},
        }
    ).model
    transition = BlockTransition(cfg)
    mask = transition.source_mask
    assert mask.shape == (3, 3)
    assert mask[0].tolist() == [True, False, False]
    assert mask[1].tolist() == [True, True, False]
    assert mask[2].tolist() == [True, True, True]


def test_readout_input_is_token_encoding_only():
    cfg = make_config("block_diagonal", "gated_linear")
    task = build_task(cfg.task, seed=0)
    model = build_model(cfg.model, task.vocab_size, task.out_dim)
    captured: dict[str, torch.Tensor] = {}

    def hook(module, inputs, output):
        captured["x"] = inputs[1]

    handle = model.readout.register_forward_hook(hook)
    batch = task.sample(2)
    with torch.no_grad():
        model(batch.input_ids)
        expected = model.encoder(batch.input_ids)
    handle.remove()
    assert torch.allclose(captured["x"], expected)
    # The state itself still reaches the readout as the first argument.
    assert captured["x"].shape == (2, batch.input_ids.shape[1], cfg.model.d_model)


def test_hybrid_gate_bias_control():
    cfg = ExperimentConfig.from_dict(
        {
            "model": {
                "d_model": 16,
                "n_slots": 2,
                "slot_dim": 8,
                "block_size": 2,
                "state_type": "block_diagonal",
                "readout": "hybrid",
                "readout_hidden": 8,
                "hybrid_gate_bias_init": 2.0,
            },
            "task": {
                "name": "mqar",
                "mqar": {
                    "n_keys": 8,
                    "n_values": 8,
                    "n_pairs": 2,
                    "n_queries": 2,
                    "eval_pairs": [2],
                    "eval_distractors": [0],
                },
            },
            "train": {"steps": 1},
        }
    )
    task = build_task(cfg.task, seed=0)
    model = build_model(cfg.model, task.vocab_size, task.out_dim)
    assert torch.allclose(
        model.readout.gate.bias,
        torch.full_like(model.readout.gate.bias, 2.0),
    )
    batch = task.sample(2)
    logits, _ = model(batch.input_ids)
    assert logits.shape == (2, batch.input_ids.shape[1], task.out_dim)


def test_readout_param_adapter():
    base_cfg = make_config("block_diagonal", "gated_linear")
    task = build_task(base_cfg.task, seed=0)
    base = build_model(base_cfg.model, task.vocab_size, task.out_dim)
    natural = parameter_breakdown(base)["readout"]

    adapted_cfg = make_config("block_diagonal", "gated_linear")
    adapted_cfg.model.readout_param_budget = natural + 2 * (
        adapted_cfg.model.n_slots * adapted_cfg.model.slot_dim
    )
    adapted = build_model(adapted_cfg.model, task.vocab_size, task.out_dim)
    adapted_params = parameter_breakdown(adapted)["readout"]
    assert adapted_params > natural
    assert hasattr(adapted.readout, "adapter")
    # The adapter starts as the identity, so it does not perturb the readout.
    assert float(adapted.readout.adapter.up.weight.detach().abs().sum()) == 0.0

    batch = task.sample(2)
    logits, _ = adapted(batch.input_ids)
    assert logits.shape == (2, batch.input_ids.shape[1], task.out_dim)
    assert torch.isfinite(logits).all()
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, task.out_dim),
        batch.labels.reshape(-1),
        ignore_index=-100,
    )
    loss.backward()


@pytest.mark.parametrize("query_mode", ["content", "constant", "random"])
def test_attention_query_modes(query_mode: str):
    cfg = ExperimentConfig.from_dict(
        {
            "model": {
                "d_model": 16,
                "n_layers": 1,
                "n_slots": 3,
                "slot_dim": 8,
                "block_size": 2,
                "state_type": "block_diagonal",
                "readout": "attention",
                "readout_hidden": 8,
                "attention_query_mode": query_mode,
            },
            "task": {
                "name": "mqar",
                "mqar": {
                    "n_keys": 8,
                    "n_values": 8,
                    "n_pairs": 2,
                    "n_queries": 2,
                    "eval_pairs": [2, 4, 8],
                    "eval_distractors": [0],
                },
            },
            "train": {"steps": 1},
        }
    )
    task = build_task(cfg.task, seed=0)
    model = build_model(cfg.model, task.vocab_size, task.out_dim)
    batch = task.sample(2)
    logits, aux = model(batch.input_ids)
    assert logits.shape == (2, batch.input_ids.shape[1], task.out_dim)
    assert torch.isfinite(logits).all()
    if query_mode != "constant":
        assert "attn_entropy" in aux
