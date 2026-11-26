"""Sweep runner.

A sweep is a Cartesian product of dotted-key overrides crossed with seeds.  It
runs sequentially in one process; on a multi-GPU node, launch one shard per GPU
with ``--shard i/n --device cuda:i``.  Each variant's config is written to
``<output>/_configs/`` before it runs, and the manifest is updated after every
run so partial results survive an interrupted sweep.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from .config import ExperimentConfig, apply_override, parse_scalar, save_config
from .data import build_task
from .models import build_model, parameter_breakdown
from .training.loop import run_experiment
from .utils import write_json

logger = logging.getLogger(__name__)


def add_sweep_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="base YAML config")
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="KEY=V1,V2",
        help="dotted-key grid axis; repeat for a Cartesian product",
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--seeds", default="0", help="comma-separated seeds")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None, help="e.g. cuda:0, mps, cpu")
    parser.add_argument("--shard", default=None, help="i/n, zero-indexed")
    parser.add_argument("--name-prefix", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--match-params",
        action="store_true",
        help="pad smaller readouts with a residual linear adapter to the largest "
        "readout budget within each recurrence geometry",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")


def _parse_grid(specs: list[str]) -> tuple[list[str], list[list[Any]]]:
    keys: list[str] = []
    values: list[list[Any]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"grid axis must be KEY=V1,V2, got {spec!r}")
        key, raw = spec.split("=", 1)
        keys.append(key)
        values.append([parse_scalar(part) for part in raw.split(",")])
    return keys, values


def _parse_shard(text: str | None) -> tuple[int, int]:
    if text is None:
        return 0, 1
    if "/" not in text:
        raise ValueError("--shard must look like i/n")
    index_text, count_text = text.split("/", 1)
    index, count = int(index_text), int(count_text)
    if count < 1 or not 0 <= index < count:
        raise ValueError("shard index must satisfy 0 <= i < n")
    return index, count


def _variant_name(prefix: str, keys: list[str], combo: tuple[Any, ...]) -> str:
    parts = [
        _sanitize(f"{key.replace('.', '-')}={value}")
        for key, value in zip(keys, combo, strict=False)
    ]
    return "_".join([prefix, *parts]) if parts else prefix


def _match_parameters(variants: list[tuple[int, int, ExperimentConfig, list[str]]]) -> None:
    """Set a common readout parameter budget within each recurrence geometry.

    The probe model is built with ``readout_param_budget=0`` so the natural
    readout size is measured.  The target is the largest natural readout in the
    group, or an explicit budget if one was supplied.  ``maybe_adapt_readout``
    then adds a residual linear adapter to the smaller readouts.
    """
    groups: dict[tuple[Any, ...], list[tuple[int, int, int]]] = {}
    for index, (_, seed, cfg, _) in enumerate(variants):
        task = build_task(cfg.task, seed=seed)
        explicit_budget = cfg.model.readout_param_budget
        cfg.model.readout_param_budget = 0
        probe = build_model(cfg.model, task.vocab_size, task.out_dim)
        natural = parameter_breakdown(probe)["readout"]
        cfg.model.readout_param_budget = explicit_budget
        del probe

        group_name = ""
        if cfg.task.name in {"group", "a5", "s5", "parity"}:
            group_name = cfg.task.group.group
            if cfg.task.name == "a5":
                group_name = "A5"
            elif cfg.task.name == "s5":
                group_name = "S5"
            elif cfg.task.name == "parity":
                group_name = "Z2"
        key = (
            cfg.task.name,
            group_name,
            task.vocab_size,
            task.out_dim,
            cfg.model.state_type,
            cfg.model.n_slots,
            cfg.model.slot_dim,
            cfg.model.block_size,
            cfg.model.d_model,
            cfg.model.n_layers,
        )
        groups.setdefault(key, []).append((index, natural, explicit_budget))

    for items in groups.values():
        target = max(max(natural, budget) for _, natural, budget in items)
        for index, _, _ in items:
            variants[index][2].model.readout_param_budget = target


def run_sweep(args: argparse.Namespace) -> int:
    base_path = Path(args.config)
    with base_path.open("r", encoding="utf-8") as handle:
        base_raw = yaml.safe_load(handle) or {}
    if not isinstance(base_raw, dict):
        raise ValueError("base config must be a YAML mapping")

    base_name = str(base_raw.get("name", "sweep"))
    prefix = args.name_prefix or base_name
    output_dir = Path(
        args.output_dir or Path(base_raw.get("output_dir", "runs")) / f"{base_name}_sweep"
    )
    config_dir = output_dir / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    keys, values = _parse_grid(args.grid)
    combos = list(itertools.product(*values)) if keys else [()]
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")

    variants: list[tuple[int, int, ExperimentConfig, list[str]]] = []
    run_index = 0
    for combo in combos:
        overrides = list(args.set)
        overrides += [f"{key}={value}" for key, value in zip(keys, combo, strict=False)]
        for seed in seeds:
            raw = copy.deepcopy(base_raw)
            for override in overrides:
                apply_override(raw, override)
            cfg = ExperimentConfig.from_dict(raw)
            cfg.name = _variant_name(prefix, keys, combo)
            cfg.seed = seed
            cfg.output_dir = str(output_dir)
            variants.append((run_index, seed, cfg, overrides))
            run_index += 1

    if args.limit is not None:
        variants = variants[: args.limit]

    if args.match_params:
        _match_parameters(variants)

    shard_index, shard_count = _parse_shard(args.shard)
    selected = [variant for variant in variants if variant[0] % shard_count == shard_index]
    logger.info(
        "sweep: %d variants, shard %d/%d -> %d runs, output %s",
        len(variants),
        shard_index,
        shard_count,
        len(selected),
        output_dir,
    )

    manifest: dict[str, Any] = {
        "base_config": str(base_path),
        "grid": args.grid,
        "sets": args.set,
        "seeds": seeds,
        "shard": [shard_index, shard_count],
        "match_params": args.match_params,
        "output_dir": str(output_dir),
        "runs": [],
    }
    manifest_path = output_dir / "manifest.json"

    for _, seed, cfg, overrides in selected:
        run_name = f"{cfg.name}_seed{seed}"
        config_path = config_dir / f"{run_name}.yaml"
        save_config(cfg, config_path)
        logger.info("run %s | overrides=%s", run_name, overrides)
        if args.dry_run:
            manifest["runs"].append(
                {
                    "run_name": run_name,
                    "config_path": str(config_path),
                    "status": "dry-run",
                }
            )
            write_json(manifest_path, manifest)
            continue

        entry: dict[str, Any] = {
            "run_name": run_name,
            "config_path": str(config_path),
        }
        try:
            summary = run_experiment(cfg, device=args.device, output_dir=output_dir)
            entry.update(
                {
                    "status": "ok",
                    "run_dir": summary["run_dir"],
                    "param_counts": summary["param_counts"],
                    "duration_sec": summary["duration_sec"],
                }
            )
        except Exception as exc:
            logger.exception("run failed: %s", run_name)
            entry.update({"status": "error", "error": repr(exc)})
            manifest["runs"].append(entry)
            write_json(manifest_path, manifest)
            if args.fail_fast:
                raise
            continue
        manifest["runs"].append(entry)
        write_json(manifest_path, manifest)

    logger.info("sweep manifest: %s", manifest_path)
    return 0
