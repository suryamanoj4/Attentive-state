"""Command-line entry point.

Set the MPS fallback before importing torch so that unsupported MPS ops fall
back to CPU instead of failing.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import logging
import sys

from .analysis.aggregate import add_aggregate_args, run_aggregate
from .analysis.plots import add_plot_args, run_plot
from .config import load_config
from .sweep import add_sweep_args, run_sweep
from .training.loop import run_experiment


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="readout-recurrence",
        description="Tiny-scale readout vs. recurrence experiments",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train one config")
    train.add_argument("--config", required=True)
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    train.add_argument("--output-dir", default=None)
    train.add_argument("--device", default=None)
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--steps", type=int, default=None)
    train.add_argument("--name", default=None)

    sweep = subparsers.add_parser("sweep", help="run a Cartesian grid")
    add_sweep_args(sweep)

    aggregate = subparsers.add_parser("aggregate", help="aggregate run.json files")
    add_aggregate_args(aggregate)

    plot = subparsers.add_parser("plot", help="plot a summary heatmap")
    add_plot_args(plot)
    return parser


def cmd_train(args: argparse.Namespace) -> int:
    config = load_config(args.config, overrides=args.set)
    if args.seed is not None:
        config.seed = args.seed
    if args.steps is not None:
        config.train.steps = args.steps
    if args.name is not None:
        config.name = args.name
    config.validate()
    summary = run_experiment(config, device=args.device, output_dir=args.output_dir)
    print(f"run directory: {summary['run_dir']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    if args.command == "train":
        return cmd_train(args)
    if args.command == "sweep":
        return run_sweep(args)
    if args.command == "aggregate":
        return run_aggregate(args)
    if args.command == "plot":
        return run_plot(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
