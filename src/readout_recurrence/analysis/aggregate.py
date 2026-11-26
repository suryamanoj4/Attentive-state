"""Aggregate run directories into tidy tables and the readout gap."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils import read_json


def collect_runs(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    paths = [root] if root.is_file() else sorted(root.rglob("run.json"))
    runs = []
    for path in paths:
        try:
            runs.append(read_json(path))
        except (OSError, ValueError):
            continue
    return runs


def flatten_runs(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        if not run.get("completed", False):
            continue
        cfg = run.get("config", {})
        task_cfg = cfg.get("task", {})
        model_cfg = cfg.get("model", {})
        task_name = task_cfg.get("name", "")
        group_name = ""
        n_pairs = 0
        n_distractors = 0
        if task_name == "mqar":
            mqar = task_cfg.get("mqar", {})
            n_pairs = int(mqar.get("n_pairs", 0))
            n_distractors = int(mqar.get("n_distractors", 0))
        elif task_name in {"group", "a5", "s5", "parity"}:
            group_name = task_cfg.get("group", {}).get("group", "")

        base = {
            "run_name": run.get("name", ""),
            "seed": int(run.get("seed", 0)),
            "task": task_name,
            "group": group_name,
            "state_type": model_cfg.get("state_type", ""),
            "readout": model_cfg.get("readout", ""),
            "n_slots": int(model_cfg.get("n_slots", 0)),
            "slot_dim": int(model_cfg.get("slot_dim", 0)),
            "d_model": int(model_cfg.get("d_model", 0)),
            "n_layers": int(model_cfg.get("n_layers", 0)),
            "block_size": int(model_cfg.get("block_size", 0)),
            "n_pairs": n_pairs,
            "n_distractors": n_distractors,
            "params": int(run.get("param_counts", {}).get("total", 0)),
            "duration_sec": float(run.get("duration_sec", 0.0)),
        }
        for spec, metrics in run.get("final_metrics", {}).items():
            rows.append(
                {
                    **base,
                    "spec": spec,
                    "loss": float(metrics.get("loss", float("nan"))),
                    "accuracy": float(metrics.get("accuracy", float("nan"))),
                    "valid_tokens": float(metrics.get("valid_tokens", 0.0)),
                    "attn_entropy": float(metrics.get("attn_entropy", float("nan"))),
                    "attn_max": float(metrics.get("attn_max", float("nan"))),
                }
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = [
        "task",
        "group",
        "state_type",
        "readout",
        "n_slots",
        "slot_dim",
        "d_model",
        "n_layers",
        "n_pairs",
        "n_distractors",
        "spec",
    ]
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_loss=("loss", "mean"),
            std_loss=("loss", "std"),
            count=("accuracy", "count"),
            mean_duration_sec=("duration_sec", "mean"),
            params=("params", "mean"),
        )
        .reset_index()
    )
    return grouped


def readout_gap(
    df: pd.DataFrame,
    baseline: str = "gated_linear",
    candidate: str = "attention",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df
    key_cols = [
        "task",
        "group",
        "state_type",
        "n_slots",
        "slot_dim",
        "d_model",
        "n_layers",
        "n_pairs",
        "n_distractors",
        "spec",
        "seed",
    ]
    pivot = df.pivot_table(
        index=key_cols,
        columns="readout",
        values="accuracy",
        aggfunc="mean",
    )
    if baseline not in pivot.columns or candidate not in pivot.columns:
        return pd.DataFrame(), pd.DataFrame()

    gap = (pivot[candidate] - pivot[baseline]).rename("delta").reset_index()
    gap = gap.dropna(subset=["delta"])
    summary_cols = [col for col in key_cols if col != "seed"]
    summary = (
        gap.groupby(summary_cols, dropna=False)["delta"]
        .agg(mean_delta="mean", std_delta="std", count="count")
        .reset_index()
    )
    return gap, summary


def add_aggregate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runs", required=True, help="run directory or run.json")
    parser.add_argument("--output", default="runs/aggregate", help="output directory")
    parser.add_argument("--baseline", default="gated_linear")
    parser.add_argument("--candidate", default="attention")


def run_aggregate(args: argparse.Namespace) -> int:
    runs = collect_runs(args.runs)
    if not runs:
        raise FileNotFoundError(f"no run.json files found under {args.runs}")
    df = flatten_runs(runs)
    if df.empty:
        raise ValueError("no completed runs with final metrics found")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(df)
    gap, gap_summary = readout_gap(df, baseline=args.baseline, candidate=args.candidate)

    df.to_csv(output / "summary.csv", index=False)
    summary.to_csv(output / "summary_mean.csv", index=False)
    gap.to_csv(output / "readout_gap.csv", index=False)
    gap_summary.to_csv(output / "readout_gap_summary.csv", index=False)

    print(f"runs: {len(runs)}  completed rows: {len(df)}  -> {output}")
    if not gap_summary.empty:
        best = gap_summary.sort_values("mean_delta", ascending=False).head(5)
        print("largest positive readout gaps:")
        print(best.to_string(index=False))
    return 0
