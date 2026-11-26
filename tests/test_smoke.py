from __future__ import annotations

import json
from pathlib import Path

from readout_recurrence.config import ExperimentConfig
from readout_recurrence.training.loop import run_experiment


def tiny_config(name: str, task: str) -> ExperimentConfig:
    if task == "mqar":
        task_cfg = {
            "name": "mqar",
            "mqar": {
                "n_keys": 8,
                "n_values": 8,
                "n_pairs": 2,
                "n_queries": 2,
                "n_distractors": 0,
                "eval_pairs": [2],
                "eval_distractors": [0],
            },
        }
    else:
        task_cfg = {
            "name": "a5",
            "group": {
                "group": "A5",
                "train_min_len": 3,
                "train_max_len": 4,
                "eval_lengths": [4],
            },
        }
    return ExperimentConfig.from_dict(
        {
            "name": name,
            "seed": 0,
            "task": task_cfg,
            "model": {
                "d_model": 16,
                "n_layers": 1,
                "n_slots": 2,
                "slot_dim": 8,
                "block_size": 2,
                "state_type": "block_diagonal",
                "transition_chunk_size": 2,
                "token_conv_kernel": 2,
                "readout": "attention" if task == "mqar" else "linear",
                "readout_hidden": 8,
                "readout_rank": 4,
            },
            "train": {
                "steps": 4,
                "batch_size": 2,
                "lr": 0.001,
                "warmup_steps": 1,
                "log_every": 2,
                "eval_every": 2,
                "eval_batches": 1,
                "eval_batch_size": 2,
                "save_checkpoint": True,
            },
        }
    )


def test_smoke_mqar(tmp_path: Path):
    summary = run_experiment(
        tiny_config("smoke_mqar_test", "mqar"),
        device="cpu",
        output_dir=tmp_path,
    )
    assert summary["completed"] is True
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metrics.jsonl").exists()
    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["completed"] is True
    assert payload["param_counts"]["total"] > 0
    assert payload["final_metrics"]
    assert payload["last_train"]
    assert "loss" in next(iter(payload["final_metrics"].values()))


def test_smoke_a5(tmp_path: Path):
    summary = run_experiment(
        tiny_config("smoke_a5_test", "a5"),
        device="cpu",
        output_dir=tmp_path,
    )
    assert summary["completed"] is True
    payload = json.loads((Path(summary["run_dir"]) / "run.json").read_text(encoding="utf-8"))
    assert payload["final_metrics"]
    assert payload["param_counts"]["total"] > 0
