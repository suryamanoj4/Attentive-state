#!/usr/bin/env python3
"""Combine the full 60-run study and the 93-run follow-up into tidy tables.

Outputs:
  runs/combined_results/all_runs.csv
  runs/combined_results/summary.csv
  runs/combined_results/key_tables.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def load_runs(root: Path, experiment: str) -> list[dict]:
    rows: list[dict] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("run.json")):
        try:
            run = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not run.get("completed", False):
            continue
        cfg = run.get("config", {})
        model = cfg.get("model", {})
        task = cfg.get("task", {})
        mqar = task.get("mqar", {})
        group = task.get("group", {})
        params = run.get("param_counts", {})
        rel = path.parent.relative_to(root)
        source_group = rel.parts[0] if rel.parts else ""
        base = {
            "experiment": experiment,
            "source_group": source_group,
            "run_name": run.get("name", ""),
            "run_dir": str(rel),
            "seed": run.get("seed"),
            "task": task.get("name", ""),
            "group": group.get("group", ""),
            "state_type": model.get("state_type", ""),
            "readout": model.get("readout", ""),
            "n_slots": model.get("n_slots"),
            "slot_dim": model.get("slot_dim"),
            "block_size": model.get("block_size"),
            "d_model": model.get("d_model"),
            "n_layers": model.get("n_layers"),
            "readout_hidden": model.get("readout_hidden"),
            "readout_rank": model.get("readout_rank"),
            "readout_param_budget": model.get("readout_param_budget"),
            "cascade_init": model.get("cascade_init", "random"),
            "hybrid_gate_bias_init": model.get("hybrid_gate_bias_init", 0.0),
            "attention_output_gate": model.get("attention_output_gate", True),
            "attention_query_mode": model.get("attention_query_mode", "content"),
            "n_pairs": mqar.get("n_pairs"),
            "n_distractors": mqar.get("n_distractors"),
            "n_queries": mqar.get("n_queries"),
            "train_steps": cfg.get("train", {}).get("steps"),
            "params_total": params.get("total"),
            "params_readout": params.get("readout"),
            "params_recurrence": params.get("recurrence"),
            "params_encoder": params.get("encoder"),
            "duration_sec": run.get("duration_sec"),
            "finished_at": run.get("finished_at"),
            "git_hash": run.get("git_hash"),
        }
        for spec, metrics in run.get("final_metrics", {}).items():
            rows.append(
                {
                    **base,
                    "spec": spec,
                    "accuracy": metrics.get("accuracy"),
                    "loss": metrics.get("loss"),
                    "valid_tokens": metrics.get("valid_tokens"),
                    "attn_entropy": metrics.get("attn_entropy"),
                    "attn_max": metrics.get("attn_max"),
                    "slot_entropy": metrics.get("slot_entropy"),
                    "slot_max": metrics.get("slot_max"),
                }
            )
    return rows


def fmt_pct(value: float, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{100 * value:.{digits}f}"


def pct_count_table(
    df: pd.DataFrame,
    index_cols: list[str],
    specs: list[str],
    value: str = "accuracy",
) -> pd.DataFrame:
    grouped = (
        df.groupby(index_cols + ["spec"], dropna=False)[value]
        .agg(["mean", "count"])
        .reset_index()
    )
    means = grouped.pivot_table(index=index_cols, columns="spec", values="mean")
    counts = grouped.pivot_table(index=index_cols, columns="spec", values="count")
    out = pd.DataFrame(index=means.index)
    for spec in specs:
        if spec in means.columns:
            out[spec] = [
                f"{fmt_pct(means.loc[idx, spec])} ({int(counts.loc[idx, spec])})"
                if pd.notna(means.loc[idx, spec])
                else "—"
                for idx in means.index
            ]
        else:
            out[spec] = "—"
    return out.reset_index()


def first_values(df: pd.DataFrame, index_cols: list[str], cols: list[str]) -> pd.DataFrame:
    return df.groupby(index_cols, dropna=False)[cols].first().reset_index()


def md_table(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", default=str(REPO / "runs/server_102783/full_102783"))
    parser.add_argument(
        "--followup",
        default=str(REPO / "runs/server_followups_105433/followups_103960"),
    )
    parser.add_argument("--output", default=str(REPO / "runs/combined_results"))
    args = parser.parse_args()

    rows = load_runs(Path(args.full), "full_102783")
    rows += load_runs(Path(args.followup), "followup_103960")
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no completed runs found")

    df["finished_at"] = pd.to_datetime(df["finished_at"], errors="coerce", utc=True)
    df = df.sort_values("finished_at").drop_duplicates(
        ["experiment", "source_group", "run_name", "seed", "spec"], keep="last"
    )
    df["entropy"] = df["attn_entropy"].where(
        df["attn_entropy"].notna(), df["slot_entropy"]
    )
    df["wall_min"] = df["duration_sec"] / 60.0

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "all_runs.csv", index=False)

    group_cols = [
        "experiment",
        "source_group",
        "task",
        "group",
        "state_type",
        "readout",
        "n_slots",
        "slot_dim",
        "block_size",
        "d_model",
        "n_layers",
        "readout_rank",
        "readout_param_budget",
        "cascade_init",
        "hybrid_gate_bias_init",
        "attention_output_gate",
        "n_pairs",
        "n_distractors",
        "n_queries",
        "spec",
    ]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            count=("accuracy", "count"),
            mean_loss=("loss", "mean"),
            mean_duration_sec=("duration_sec", "mean"),
            params_total=("params_total", "first"),
            params_readout=("params_readout", "first"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "summary.csv", index=False)

    sections: list[str] = []
    sections.append("# Combined follow-up tables\n")
    n_runs = (
        df[["experiment", "source_group", "run_name", "seed"]]
        .drop_duplicates()
        .shape[0]
    )
    sections.append(
        f"Completed runs: {n_runs}, "
        f"{len(df)} metric rows, seeds {sorted(df['seed'].dropna().unique().tolist())}."
    )
    counts = (
        df[["experiment", "source_group", "run_name", "seed"]]
        .drop_duplicates()
        .groupby(["experiment", "source_group"], dropna=False)
        .size()
        .reset_index(name="runs")
    )
    sections.append("\n## Counts\n\n" + md_table(counts))

    # A5 full transitions
    lens = ["len1280", "len2560", "len5120"]
    a5 = df[
        (df["experiment"] == "full_102783")
        & (df["task"] == "a5")
        & (df["spec"].isin(lens))
    ]
    if not a5.empty:
        idx = ["state_type", "readout"]
        table = pct_count_table(a5, idx, lens)
        meta = (
            a5.groupby(idx, dropna=False)
            .agg(params_total=("params_total", "first"), wall_min=("wall_min", "mean"))
            .reset_index()
        )
        table = meta.merge(table, on=idx, how="right")
        table["wall_min"] = table["wall_min"].map(lambda v: f"{v:.0f}" if pd.notna(v) else "—")
        sections.append("\n## A5 full run: state vs readout\n\n" + md_table(table))

    # MQAR H1
    h1 = df[
        (df["experiment"] == "full_102783")
        & (df["source_group"] == "mqar_h1")
        & (df["readout"].isin(["gated_linear", "attention"]))
        & (df["state_type"].isin(["diagonal", "block_diagonal"]))
    ]
    if not h1.empty:
        idx = [
            "state_type",
            "readout",
            "readout_param_budget",
            "n_queries",
            "params_total",
        ]
        specs = ["train", "pairs8_dist0", "pairs16_dist0", "pairs32_dist0", "pairs64_dist0"]
        table = pct_count_table(h1, idx, specs)
        sections.append("\n## MQAR H1 (full run)\n\n" + md_table(table))

    # H2 follow-up on block-dense
    h2 = df[
        (df["experiment"] == "followup_103960")
        & (df["source_group"] == "h2_block_dense")
    ]
    if not h2.empty:
        idx = [
            "n_pairs",
            "n_distractors",
            "readout",
            "readout_param_budget",
            "n_queries",
            "params_total",
        ]
        specs = ["train", "pairs8_dist0", "pairs16_dist0", "pairs32_dist0", "pairs64_dist0"]
        table = pct_count_table(h2, idx, specs)
        sections.append("\n## MQAR H2 block-dense (follow-up)\n\n" + md_table(table))

    # H1/H2 readout gap summary
    gap_rows = []
    for label, frame in [
        (
            "H1 full",
            df[
                (df["experiment"] == "full_102783")
                & (df["source_group"] == "mqar_h1")
                & (df["state_type"].isin(["diagonal", "block_diagonal"]))
                & (df["readout"].isin(["gated_linear", "attention"]))
            ],
        ),
        (
            "H2 full diagonal",
            df[
                (df["experiment"] == "full_102783")
                & (df["source_group"] == "mqar_h2_compression")
                & (df["state_type"] == "diagonal")
                & (df["readout"].isin(["gated_linear", "attention"]))
            ],
        ),
        (
            "H2 follow-up block",
            df[
                (df["experiment"] == "followup_103960")
                & (df["source_group"] == "h2_block_dense")
                & (df["readout"].isin(["gated_linear", "attention"]))
            ],
        ),
    ]:
        if frame.empty:
            continue
        index_cols = ["n_pairs", "n_distractors", "seed", "spec"]
        group_cols = ["spec"]
        if frame["state_type"].nunique() > 1:
            index_cols = ["state_type"] + index_cols
            group_cols = ["state_type", "spec"]
        pivot = frame.pivot_table(
            index=index_cols,
            columns="readout",
            values="accuracy",
            aggfunc="mean",
        ).reset_index()
        if "attention" not in pivot.columns or "gated_linear" not in pivot.columns:
            continue
        pivot["gap"] = pivot["attention"] - pivot["gated_linear"]
        gap = (
            pivot.groupby(group_cols)["gap"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values(group_cols)
        )
        for _, row in gap.iterrows():
            gap_rows.append(
                {
                    "comparison": label,
                    "state": row.get("state_type", ""),
                    "spec": row["spec"],
                    "mean_gap_pp": f"{100 * row['mean']:.1f}",
                    "std_pp": f"{100 * row['std']:.1f}" if pd.notna(row["std"]) else "—",
                    "n": int(row["count"]),
                }
            )
    if gap_rows:
        sections.append("\n## Attention − gated gap (paired by condition/seed)\n\n" + md_table(pd.DataFrame(gap_rows)))

    # Rank
    rank = df[(df["experiment"] == "followup_103960") & (df["source_group"] == "rank")]
    if not rank.empty:
        idx = ["readout_rank", "params_total", "readout_param_budget"]
        specs = ["pairs8_dist0", "pairs16_dist0", "pairs32_dist0", "pairs64_dist0"]
        table = pct_count_table(rank, idx, specs)
        sections.append("\n## Rank sweep (block-dense MQAR, 16/16)\n\n" + md_table(table))

    # Slots
    slots = df[
        (df["experiment"] == "followup_103960")
        & (df["source_group"].str.startswith("slots_L"))
    ]
    if not slots.empty:
        idx = ["n_slots", "slot_dim", "readout", "params_total", "readout_param_budget"]
        specs = ["len1280", "len2560", "len5120"]
        table = pct_count_table(slots, idx, specs)
        entropy = (
            slots[slots["spec"] == "len5120"]
            .groupby(idx, dropna=False)["entropy"]
            .mean()
            .reset_index(name="entropy5120")
        )
        table = table.merge(entropy, on=idx, how="left")
        table["entropy5120"] = table["entropy5120"].map(
            lambda v: f"{v:.3f}" if pd.notna(v) else "—"
        )
        sections.append("\n## Slot ablation (A5 block-dense, width 128)\n\n" + md_table(table))

        # Per-seed len5120 values expose seed fragility.
        seed_rows = []
        for keys, frame in slots[slots["spec"] == "len5120"].groupby(
            ["n_slots", "slot_dim", "readout"], dropna=False
        ):
            row = {
                "n_slots": keys[0],
                "slot_dim": keys[1],
                "readout": keys[2],
            }
            for seed in [0, 1, 2]:
                vals = frame.loc[frame["seed"] == seed, "accuracy"]
                row[f"seed{seed}"] = fmt_pct(vals.iloc[0]) if len(vals) else "—"
            row["mean"] = fmt_pct(frame["accuracy"].mean())
            seed_rows.append(row)
        sections.append("\n### Slot per-seed at len5120\n\n" + md_table(pd.DataFrame(seed_rows)))

    # Cascade
    cascade = df[
        (df["state_type"] == "cascade_chain")
        & (df["readout"].isin(["linear", "attention"]))
        & (df["spec"].isin(lens))
    ]
    if not cascade.empty:
        idx = ["experiment", "cascade_init", "readout", "params_total"]
        table = pct_count_table(cascade, idx, lens)
        sections.append("\n## Cascade: initialization\n\n" + md_table(table))

        seed_rows = []
        for keys, frame in cascade[cascade["spec"] == "len5120"].groupby(
            ["experiment", "cascade_init", "readout"], dropna=False
        ):
            row = {
                "experiment": keys[0],
                "cascade_init": keys[1],
                "readout": keys[2],
            }
            for seed in [0, 1, 2]:
                vals = frame.loc[frame["seed"] == seed, "accuracy"]
                row[f"seed{seed}"] = fmt_pct(vals.iloc[0]) if len(vals) else "—"
            row["mean"] = fmt_pct(frame["accuracy"].mean())
            seed_rows.append(row)
        sections.append("\n### Cascade per-seed at len5120\n\n" + md_table(pd.DataFrame(seed_rows)))

    # Hybrid
    hybrid = df[
        (df["experiment"] == "followup_103960") & (df["source_group"] == "hybrid")
    ]
    if not hybrid.empty:
        idx = [
            "hybrid_gate_bias_init",
            "params_total",
            "readout_param_budget",
            "attention_output_gate",
        ]
        specs = ["pairs8_dist0", "pairs16_dist0", "pairs32_dist0", "pairs64_dist0"]
        table = pct_count_table(hybrid, idx, specs)
        sections.append("\n## Hybrid gate study\n\n" + md_table(table))

    # Efficiency: key full-run pairs and H1 pair
    eff_rows = []
    for keys, frame in df[
        (df["experiment"] == "full_102783")
        & (df["task"] == "a5")
        & (df["readout"].isin(["linear", "attention"]))
    ].groupby(["state_type", "readout"], dropna=False):
        acc = frame.loc[frame["spec"] == "len5120", "accuracy"]
        eff_rows.append(
            {
                "experiment": "A5 full",
                "condition": keys[0],
                "readout": keys[1],
                "params": int(frame["params_total"].iloc[0]),
                "wall_min": f"{frame['wall_min'].mean():.0f}",
                "metric": "len5120",
                "accuracy": fmt_pct(acc.mean() if len(acc) else float("nan")),
            }
        )
    for keys, frame in df[
        (df["experiment"] == "full_102783")
        & (df["source_group"] == "mqar_h1")
        & (df["state_type"] == "block_diagonal")
        & (df["readout"].isin(["gated_linear", "attention"]))
    ].groupby(["state_type", "readout"], dropna=False):
        acc = frame.loc[frame["spec"] == "pairs8_dist0", "accuracy"]
        eff_rows.append(
            {
                "experiment": "MQAR H1 block",
                "condition": keys[0],
                "readout": keys[1],
                "params": int(frame["params_total"].iloc[0]),
                "wall_min": f"{frame['wall_min'].mean():.0f}",
                "metric": "pairs8",
                "accuracy": fmt_pct(acc.mean() if len(acc) else float("nan")),
            }
        )
    if eff_rows:
        sections.append("\n## Efficiency snapshot\n\n" + md_table(pd.DataFrame(eff_rows)))

    (out_dir / "key_tables.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'all_runs.csv'}")
    print(f"wrote {out_dir / 'summary.csv'}")
    print(f"wrote {out_dir / 'key_tables.md'}")
    print(f"rows={len(df)} run_specs={df[['experiment','source_group','run_name','seed','spec']].drop_duplicates().shape[0]}")


if __name__ == "__main__":
    main()
