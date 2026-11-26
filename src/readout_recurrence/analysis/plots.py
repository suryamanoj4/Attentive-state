"""Small plotting helpers for phase diagrams and readout gaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _annotate(ax, matrix: np.ndarray, fmt: str = "{:.3f}") -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=9)


def plot_heatmap(
    summary_path: str | Path,
    output_path: str | Path,
    *,
    task: str | None = None,
    spec: str | None = None,
    kind: str = "accuracy",
    value: str = "mean_accuracy",
    title: str | None = None,
) -> None:
    df = pd.read_csv(summary_path)
    if task is not None:
        df = df[df["task"] == task]
    if spec is not None:
        df = df[df["spec"] == spec]
    if df.empty:
        raise ValueError("no rows left after filtering")

    if kind == "accuracy":
        matrix = df.pivot_table(index="readout", columns="state_type", values=value, aggfunc="mean")
        xlabel, ylabel = "state type", "readout"
    elif kind == "gap":
        matrix = df.pivot_table(index="state_type", columns="n_slots", values=value, aggfunc="mean")
        xlabel, ylabel = "number of slots", "state type"
    else:
        raise ValueError("kind must be 'accuracy' or 'gap'")

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * matrix.shape[1]), max(3, 1.2 * matrix.shape[0])))
    image = ax.imshow(matrix.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), [str(c) for c in matrix.columns])
    ax.set_yticks(range(matrix.shape[0]), [str(i) for i in matrix.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{task or 'all'} / {spec or 'all'} / {value}")
    _annotate(ax, matrix.values)
    fig.colorbar(image, ax=ax, label=value)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_learning_curves(
    metrics_path: str | Path,
    output_path: str | Path,
    *,
    spec: str | None = None,
) -> None:
    rows = []
    with Path(metrics_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            import json

            record = json.loads(line)
            if record.get("event") == "eval":
                rows.append(record)
    if not rows:
        raise ValueError("no eval records found")
    df = pd.DataFrame(rows)
    if spec is not None:
        df = df[df["spec"] == spec]
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, group in df.groupby("spec"):
        ax.plot(group["step"], group["accuracy"], marker="o", label=name)
    ax.set_xlabel("step")
    ax.set_ylabel("accuracy")
    ax.set_title("evaluation accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def add_plot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--spec", default=None)
    parser.add_argument("--kind", choices=["accuracy", "gap"], default="accuracy")
    parser.add_argument("--value", default="mean_accuracy")
    parser.add_argument("--title", default=None)


def run_plot(args: argparse.Namespace) -> int:
    plot_heatmap(
        args.summary,
        args.output,
        task=args.task,
        spec=args.spec,
        kind=args.kind,
        value=args.value,
        title=args.title,
    )
    print(f"wrote {args.output}")
    return 0
