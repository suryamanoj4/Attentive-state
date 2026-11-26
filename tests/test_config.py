from __future__ import annotations

import pytest
import yaml

from readout_recurrence.config import (
    ExperimentConfig,
    apply_override,
    load_config,
    save_config,
)


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        ExperimentConfig.from_dict({"model": {"not_a_field": 1}})


def test_diagonal_forces_block_size_one():
    cfg = ExperimentConfig.from_dict({"model": {"state_type": "diagonal", "block_size": 4}})
    assert cfg.model.block_size == 1


def test_block_size_must_divide_slot_dim():
    with pytest.raises(ValueError, match="divide"):
        ExperimentConfig.from_dict(
            {
                "model": {
                    "state_type": "block_diagonal",
                    "slot_dim": 10,
                    "block_size": 4,
                }
            }
        )


def test_mqar_query_validation():
    with pytest.raises(ValueError, match="n_queries"):
        ExperimentConfig.from_dict(
            {
                "task": {
                    "name": "mqar",
                    "mqar": {"n_pairs": 2, "n_queries": 3},
                }
            }
        )


def test_apply_override_nested():
    data = {"model": {"readout": "linear"}, "train": {"steps": 10}}
    apply_override(data, "model.readout=attention")
    apply_override(data, "train.steps=20")
    assert data["model"]["readout"] == "attention"
    assert data["train"]["steps"] == 20


def test_yaml_roundtrip(tmp_path):
    cfg = ExperimentConfig.from_dict(
        {
            "name": "roundtrip",
            "seed": 3,
            "task": {"name": "mqar", "mqar": {"n_pairs": 4, "n_queries": 4}},
            "model": {"state_type": "cascade_chain", "n_slots": 4},
        }
    )
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.to_dict() == cfg.to_dict()
    # The YAML is plain data, not a pickled object.
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    assert raw["model"]["n_slots"] == 4


def test_load_config_with_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "base",
                "model": {"state_type": "diagonal", "readout": "linear"},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(path, overrides=["model.readout=attention", "seed=7"])
    assert cfg.model.readout == "attention"
    assert cfg.seed == 7
