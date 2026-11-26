from __future__ import annotations

import pytest
import torch

from readout_recurrence.config import ExperimentConfig
from readout_recurrence.data import build_task
from readout_recurrence.models import build_model
from readout_recurrence.models.layers import TokenEncoder

STATE_TYPES = ["diagonal", "block_diagonal", "cascade_chain", "cascade_full", "full"]
READOUTS = ["linear", "input_linear", "gated_linear", "mlp", "attention", "hybrid"]


def make_config(state_type: str, readout: str) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "task": {
                "name": "a5",
                "group": {"group": "A5", "train_min_len": 6, "train_max_len": 6},
            },
            "model": {
                "d_model": 16,
                "n_layers": 1,
                "n_slots": 2,
                "slot_dim": 8,
                "block_size": 2,
                "state_type": state_type,
                "transition_chunk_size": 3,
                "token_conv_kernel": 3,
                "readout": readout,
                "readout_hidden": 8,
                "readout_rank": 4,
            },
            "train": {"steps": 1},
        }
    )


@pytest.mark.parametrize("state_type", STATE_TYPES)
@pytest.mark.parametrize("readout", READOUTS)
def test_future_tokens_do_not_change_past_logits(state_type: str, readout: str):
    cfg = make_config(state_type, readout)
    task = build_task(cfg.task, seed=0)
    model = build_model(cfg.model, task.vocab_size, task.out_dim).eval()
    batch = task.sample(2, length=6)
    changed = batch.input_ids.clone()
    changed[:, -1] = (changed[:, -1] + 1) % task.vocab_size
    with torch.no_grad():
        original, _ = model(batch.input_ids)
        perturbed, _ = model(changed)
    assert torch.allclose(original[:, :-1], perturbed[:, :-1], atol=1e-6)


def test_token_encoder_convolution_is_causal():
    cfg = ExperimentConfig.from_dict(
        {
            "model": {
                "d_model": 16,
                "n_slots": 1,
                "slot_dim": 8,
                "block_size": 2,
                "token_conv_kernel": 3,
            },
            "train": {"steps": 1},
        }
    ).model
    encoder = TokenEncoder(vocab_size=32, config=cfg).eval()
    tokens = torch.randint(0, 32, (2, 7))
    changed = tokens.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 32
    with torch.no_grad():
        original = encoder(tokens)
        perturbed = encoder(changed)
    assert torch.allclose(original[:, :-1], perturbed[:, :-1], atol=1e-6)
