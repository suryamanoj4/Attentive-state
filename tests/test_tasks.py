from __future__ import annotations

import numpy as np

from readout_recurrence.config import (
    CopyConfig,
    ExperimentConfig,
    GroupConfig,
    MQARConfig,
)
from readout_recurrence.data import build_task
from readout_recurrence.data.group import GroupTask


def test_mqar_shapes_and_query_labels():
    cfg = MQARConfig(
        n_keys=16,
        n_values=16,
        n_pairs=4,
        n_queries=4,
        n_distractors=2,
        shuffle_pairs=False,
        eval_pairs=[4, 8],
        eval_distractors=[2],
    )
    task = build_task(
        ExperimentConfig.from_dict({"task": {"name": "mqar", "mqar": cfg.__dict__}}).task,
        seed=3,
    )
    batch = task.sample(5)
    b, t = batch.input_ids.shape
    assert b == 5
    assert batch.labels.shape == (b, t)
    assert int((batch.labels != -100).sum()) == b * 4

    query_positions = batch.labels != -100
    assert torch_all_in_range(batch.labels[query_positions], 0, 16)
    # A query position is a key id, never a value id or the marker.
    assert torch_all_in_range(batch.input_ids[query_positions], 0, 16)
    # Real pair key positions are key ids; pair value positions are value ids.
    pair_len = 2 * (4 + 2)
    assert torch_all_in_range(batch.input_ids[:, 0 : 2 * 4 : 2], 0, 16)
    assert torch_all_in_range(batch.input_ids[:, 1 : 2 * 4 : 2], 16, 32)
    # Distractor pairs use value-vocabulary tokens for both positions, so they
    # can never collide with a queried key.
    assert torch_all_in_range(batch.input_ids[:, 2 * 4 : pair_len : 2], 16, 32)
    assert torch_all_in_range(batch.input_ids[:, 2 * 4 + 1 : pair_len : 2], 16, 32)
    assert int(batch.input_ids.max()) < task.vocab_size


def test_mqar_deterministic_with_seeded_rng():
    cfg = MQARConfig(
        n_keys=8,
        n_values=8,
        n_pairs=3,
        n_queries=3,
        eval_pairs=[3, 6],
        eval_distractors=[0],
    )
    task = build_task(
        ExperimentConfig.from_dict({"task": {"name": "mqar", "mqar": cfg.__dict__}}).task,
        seed=0,
    )
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    batch_a = task.sample(4, rng=rng_a)
    batch_b = task.sample(4, rng=rng_b)
    assert (batch_a.input_ids == batch_b.input_ids).all()
    assert (batch_a.labels == batch_b.labels).all()


def test_mqar_eval_specs_clamp_queries_to_pairs():
    cfg = MQARConfig(
        n_keys=16,
        n_values=16,
        n_pairs=16,
        n_queries=16,
        eval_pairs=[4, 8, 16],
        eval_distractors=[0],
    )
    task = build_task(
        ExperimentConfig.from_dict({"task": {"name": "mqar", "mqar": cfg.__dict__}}).task,
        seed=0,
    )
    specs = {spec.name: spec for spec in task.eval_specs()}
    assert specs["pairs4_dist0"].kwargs["n_queries"] == 4
    assert specs["pairs8_dist0"].kwargs["n_queries"] == 8
    for spec in specs.values():
        batch = task.sample(2, **spec.kwargs)
        assert batch.labels.shape == batch.input_ids.shape


def test_mqar_distractors_change_length_not_query_count():
    cfg = MQARConfig(
        n_keys=16,
        n_values=16,
        n_pairs=4,
        n_queries=4,
        n_distractors=0,
        eval_pairs=[4, 8, 16],
        eval_distractors=[0, 2],
    )
    task = build_task(
        ExperimentConfig.from_dict({"task": {"name": "mqar", "mqar": cfg.__dict__}}).task,
        seed=0,
    )
    clean = task.sample(3)
    noisy = task.sample(3, n_distractors=5)
    assert noisy.input_ids.shape[1] == clean.input_ids.shape[1] + 2 * 5
    assert int((noisy.labels != -100).sum()) == 3 * 4
    assert int(noisy.input_ids.max()) < task.vocab_size


def test_group_labels_are_cumulative_products():
    cfg = GroupConfig(group="A5", train_min_len=5, train_max_len=5)
    task = GroupTask(cfg, seed=1)
    assert task.group_size == 60
    tokens = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=np.int64)
    # Build a deterministic batch from known tokens via the internal table.
    labels = np.empty_like(tokens)
    state = np.zeros(2, dtype=np.int64)
    for t in range(tokens.shape[1]):
        state = task.table[state, tokens[:, t]]
        labels[:, t] = state
    # The first token's label is the token itself (identity * g = g).
    assert np.array_equal(labels[:, 0], tokens[:, 0])
    assert np.array_equal(
        labels[:, 1],
        np.array(
            [
                task.table[tokens[0, 0], tokens[0, 1]],
                task.table[tokens[1, 0], tokens[1, 1]],
            ]
        ),
    )


def test_parity_labels():
    cfg = GroupConfig(group="Z2", train_min_len=4, train_max_len=4)
    task = GroupTask(cfg, seed=0)
    # Use the generator with a seeded RNG and check the invariant manually.
    batch = task.sample(2, length=4, rng=np.random.default_rng(7))
    expected = np.zeros((2, 4), dtype=np.int64)
    state = np.zeros(2, dtype=np.int64)
    for t in range(4):
        state = task.table[state, batch.input_ids[:, t].numpy()]
        expected[:, t] = state
    assert np.array_equal(batch.labels.numpy(), expected)


def test_compose_is_associative_on_random_permutations():
    cfg = GroupConfig(group="S5")
    task = GroupTask(cfg, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b, c = rng.integers(0, 120, size=3)
        left = task.table[task.table[a, b], c]
        right = task.table[a, task.table[b, c]]
        assert left == right


def test_copy_structure_and_labels():
    cfg = CopyConfig(content_vocab=8, train_min_len=4, train_max_len=4)
    task = build_task(
        ExperimentConfig.from_dict({"task": {"name": "copy", "copy": cfg.__dict__}}).task,
        seed=0,
    )
    batch = task.sample(2, length=4, rng=np.random.default_rng(0))
    n = 4
    input_ids = batch.input_ids
    labels = batch.labels
    assert input_ids.shape[1] == 2 * n + 2
    assert (input_ids[:, 0] == task.bos_id).all()
    assert (input_ids[:, 1 + n] == task.sep_id).all()
    assert (labels[:, 1 + n] == input_ids[:, n + 2]).all()
    assert (labels[:, n + 2 : 2 * n + 1] == input_ids[:, n + 3 : 2 * n + 2]).all()
    assert (labels[:, 2 * n + 1] == task.eos_id).all()


def test_registry_builds_all_tasks():
    for name in ["mqar", "a5", "s5", "parity", "copy"]:
        cfg = ExperimentConfig.from_dict(
            {
                "task": {
                    "name": name,
                    "group": {"group": "S5"},
                    "mqar": {
                        "n_keys": 8,
                        "n_values": 8,
                        "n_pairs": 2,
                        "n_queries": 2,
                        "eval_pairs": [2, 4, 8],
                        "eval_distractors": [0],
                    },
                    "copy": {"content_vocab": 4},
                }
            }
        )
        task = build_task(cfg.task, seed=0)
        batch = task.sample(2, length=4)
        assert batch.input_ids.shape[0] == 2
        assert batch.input_ids.max().item() < task.vocab_size


def torch_all_in_range(tensor, low: int, high: int) -> bool:
    return bool(((tensor >= low) & (tensor < high)).all().item())
