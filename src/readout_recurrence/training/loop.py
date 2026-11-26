"""Training loop, evaluation, logging, and checkpoints.

A run is deliberately self-contained: one YAML in, one directory out with
``config.yaml``, ``metrics.jsonl``, ``run.json``, and checkpoints.  Evaluation
batches are generated once from deterministic seeds, so the same checkpoint is
always scored on the same data and different readouts can be compared pairwise.
"""

from __future__ import annotations

import logging
import math
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import ExperimentConfig, TrainConfig, save_config
from ..data import Batch, EvalSpec, build_task
from ..models import build_model, parameter_breakdown
from ..seeding import configure_threads, resolve_device, seed_everything
from ..utils import (
    append_jsonl,
    format_duration,
    git_hash,
    stable_hash,
    utc_now,
    write_json,
)
from .metrics import RunningMetrics, masked_accuracy, masked_loss

logger = logging.getLogger(__name__)


@dataclass
class RunPaths:
    root: Path
    config: Path
    metrics: Path
    run_json: Path
    best: Path
    last: Path


def make_run_paths(config: ExperimentConfig, output_dir: str | Path | None = None) -> RunPaths:
    base = Path(output_dir if output_dir is not None else config.output_dir)
    root = base / f"{config.name}_seed{config.seed}"
    if root.exists():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        root = root.with_name(f"{root.name}_{stamp}")
    root.mkdir(parents=True, exist_ok=False)
    return RunPaths(
        root=root,
        config=root / "config.yaml",
        metrics=root / "metrics.jsonl",
        run_json=root / "run.json",
        best=root / "best.pt",
        last=root / "last.pt",
    )


def lr_at_step(step: int, config: TrainConfig) -> float:
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.lr * float(step + 1) / float(config.warmup_steps)
    if config.schedule == "constant":
        return config.lr
    warmup = config.warmup_steps
    progress = (step - warmup) / max(1, config.steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    if config.schedule == "cosine":
        factor = config.min_lr_ratio + (1.0 - config.min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
    elif config.schedule == "linear":
        factor = 1.0 - progress * (1.0 - config.min_lr_ratio)
    else:
        raise ValueError(f"unknown schedule: {config.schedule}")
    return config.lr * factor


def prepare_eval_batches(
    task,
    specs: list[EvalSpec],
    *,
    batch_size: int,
    n_batches: int,
    seed: int,
    device: torch.device,
) -> dict[str, list[Batch]]:
    prepared: dict[str, list[Batch]] = {}
    for spec in specs:
        rng = np.random.default_rng((seed + stable_hash(spec.name)) % (2**32))
        batches = [
            task.sample(batch_size, rng=rng, **spec.kwargs).to(device) for _ in range(n_batches)
        ]
        prepared[spec.name] = batches
    return prepared


def evaluate_batches(
    model: torch.nn.Module,
    batches: list[Batch],
    out_dim: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    correct = 0
    valid = 0
    tokens = 0
    aux_values: dict[str, list[float]] = {}

    with torch.no_grad():
        for batch in batches:
            logits, aux = model(batch.input_ids)
            loss_sum += float(masked_loss(logits, batch.labels, reduction="sum").item())
            batch_correct, batch_valid = masked_accuracy(logits, batch.labels)
            correct += int(batch_correct.item())
            valid += int(batch_valid.item())
            tokens += int(batch.input_ids.numel())
            for key, value in aux.items():
                if isinstance(value, torch.Tensor):
                    aux_values.setdefault(key, []).append(
                        float(value.detach().float().mean().item())
                    )

    if was_training:
        model.train()

    metrics = {
        "loss": loss_sum / max(valid, 1),
        "accuracy": correct / max(valid, 1),
        "valid_tokens": float(valid),
        "tokens": float(tokens),
    }
    for key, values in aux_values.items():
        metrics[key] = float(np.mean(values))
    return metrics


def evaluate_all(
    model: torch.nn.Module,
    eval_batches: dict[str, list[Batch]],
    out_dim: int,
) -> dict[str, dict[str, float]]:
    return {
        name: evaluate_batches(model, batches, out_dim) for name, batches in eval_batches.items()
    }


def mean_accuracy(metrics: dict[str, dict[str, float]]) -> float:
    if not metrics:
        return float("-inf")
    values = [item["accuracy"] for item in metrics.values()]
    values = [v for v in values if not math.isnan(v)]
    return float(np.mean(values)) if values else float("-inf")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: ExperimentConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config.to_dict(),
        },
        path,
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    device: str | None = None,
    output_dir: str | Path | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train and evaluate one config.  Returns a summary dict."""
    config.validate()
    seed_everything(config.seed, deterministic=config.train.deterministic)
    configure_threads(config.train.num_threads)
    torch_device = resolve_device(device or config.device)

    task = build_task(config.task, seed=config.seed)
    model = build_model(config.model, task.vocab_size, task.out_dim).to(torch_device)
    param_counts = parameter_breakdown(model)
    logger.info(
        "run=%s seed=%d device=%s params=%s",
        config.name,
        config.seed,
        torch_device,
        param_counts,
    )

    paths = make_run_paths(config, output_dir=output_dir)
    save_config(config, paths.config)
    logger.info("writing to %s", paths.root)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    use_amp = bool(config.train.amp and torch_device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    specs = task.eval_specs()
    eval_batches = prepare_eval_batches(
        task,
        specs,
        batch_size=config.train.eval_batch_size,
        n_batches=config.train.eval_batches,
        seed=config.seed,
        device=torch_device,
    )

    started_at = utc_now()
    start_time = time.perf_counter()
    window = RunningMetrics()
    window_start = start_time
    last_eval_step = -1
    best_score = float("-inf")
    best_record: dict[str, Any] | None = None
    final_metrics: dict[str, dict[str, float]] = {}
    last_train: dict[str, float] = {}
    completed = False

    try:
        for step in range(config.train.steps):
            lr = lr_at_step(step, config.train)
            for group in optimizer.param_groups:
                group["lr"] = lr

            batch = task.sample(config.train.batch_size).to(torch_device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits, _ = model(batch.input_ids)
                loss = masked_loss(logits, batch.labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                if config.train.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config.train.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
                optimizer.step()

            window.update(loss, logits, batch.labels)
            global_step = step + 1

            should_log = (
                global_step % config.train.log_every == 0 or global_step == config.train.steps
            )
            if should_log:
                now = time.perf_counter()
                elapsed = now - window_start
                tokens_per_sec = window.tokens / elapsed if elapsed > 0 else 0.0
                record = {
                    "event": "train",
                    "step": global_step,
                    "loss": window.loss,
                    "accuracy": window.accuracy,
                    "lr": lr,
                    "tokens_per_sec": tokens_per_sec,
                    "elapsed_sec": elapsed,
                }
                append_jsonl(paths.metrics, record)
                logger.info(
                    "step %d/%d | loss %.4f | acc %.4f | lr %.2e | %.0f tok/s",
                    global_step,
                    config.train.steps,
                    window.loss,
                    window.accuracy,
                    lr,
                    tokens_per_sec,
                )
                if progress is not None:
                    progress(record)
                last_train = window.as_dict()
                window = RunningMetrics()
                window_start = now

            should_eval = (
                global_step % config.train.eval_every == 0 or global_step == config.train.steps
            )
            if should_eval:
                final_metrics = evaluate_all(model, eval_batches, task.out_dim)
                last_eval_step = global_step
                score = mean_accuracy(final_metrics)
                summary = " | ".join(
                    f"{name}: loss {m['loss']:.4f} acc {m['accuracy']:.4f}"
                    for name, m in final_metrics.items()
                )
                logger.info("eval @ %d | %s", global_step, summary)
                for name, metrics in final_metrics.items():
                    append_jsonl(
                        paths.metrics,
                        {
                            "event": "eval",
                            "step": global_step,
                            "spec": name,
                            **metrics,
                        },
                    )
                if score > best_score:
                    best_score = score
                    best_record = {
                        "step": global_step,
                        "score": score,
                        "metrics": final_metrics,
                    }
                    if config.train.save_checkpoint:
                        save_checkpoint(paths.best, model, optimizer, global_step, config)
                if config.train.save_checkpoint:
                    save_checkpoint(paths.last, model, optimizer, global_step, config)

        completed = True
    except Exception as exc:
        write_json(
            paths.run_json,
            {
                "name": config.name,
                "seed": config.seed,
                "config": config.to_dict(),
                "param_counts": param_counts,
                "device": str(torch_device),
                "torch_version": torch.__version__,
                "git_hash": git_hash(paths.root),
                "started_at": started_at,
                "finished_at": utc_now(),
                "completed": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise

    duration = time.perf_counter() - start_time
    if last_eval_step != config.train.steps:
        final_metrics = evaluate_all(model, eval_batches, task.out_dim)
        last_eval_step = config.train.steps
        score = mean_accuracy(final_metrics)
        if score > best_score:
            best_score = score
            best_record = {
                "step": last_eval_step,
                "score": score,
                "metrics": final_metrics,
            }

    if config.train.save_checkpoint:
        save_checkpoint(paths.last, model, optimizer, config.train.steps, config)
        # `best.pt` is written at the eval that achieved the best score; if the
        # final eval improved it, write it now.
        if best_record is not None and best_record["step"] == last_eval_step:
            save_checkpoint(paths.best, model, optimizer, last_eval_step, config)

    payload = {
        "name": config.name,
        "seed": config.seed,
        "config": config.to_dict(),
        "param_counts": param_counts,
        "device": str(torch_device),
        "torch_version": torch.__version__,
        "git_hash": git_hash(paths.root),
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_sec": duration,
        "completed": completed,
        "final_metrics": final_metrics,
        "best": best_record,
        "last_train": last_train,
    }
    write_json(paths.run_json, payload)
    logger.info(
        "done in %s | best acc %.4f | %s",
        format_duration(duration),
        best_score if best_score > float("-inf") else float("nan"),
        paths.root,
    )
    return {
        "run_dir": str(paths.root),
        "param_counts": param_counts,
        "final_metrics": final_metrics,
        "best": best_record,
        "completed": completed,
        "duration_sec": duration,
    }
