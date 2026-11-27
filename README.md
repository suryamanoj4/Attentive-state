# Readout vs. Recurrence: experiment code

Code for the paper-in-progress **“The Readout Is Not the Recurrence: What Attention
Over State Can and Cannot Buy.”** The research notes live in [`research/`](research/);
this directory is the runnable harness.

The claim under test is deliberately narrow:

> Attention over recurrent state slots can enlarge the **readout** function class,
> but it cannot change the **state-tracking** class of the underlying recurrence.
> The transition matrix is the lever, and readout vs. recurrence effects dissociate
> empirically.

The harness is built for tiny models (roughly 0.1–10M parameters) on synthetic
tasks. It is a mechanism study, not a scaling study.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/). Python 3.12 is pinned in
`.python-version`.

```bash
uv sync
```

This installs PyTorch, NumPy, pandas, matplotlib, and PyYAML into `.venv/`.
On the Linux GPU node, `uv sync` installs the standard Linux PyTorch wheel. If you
need a particular CUDA build, install it explicitly after syncing:

```bash
uv pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
```

On macOS, MPS is used automatically when available. Export
`PYTORCH_ENABLE_MPS_FALLBACK=1` if an op is missing on MPS; the CLI does this for
you at import time.

## Quick checks

```bash
uv run pytest
uv run ruff check .
make smoke
make pilot-h1    # MQAR readout pilot
make pilot-a5    # A5 transition pilot
```

A smoke run writes `runs/<name>_seed<seed>/` containing `config.yaml`,
`metrics.jsonl`, `run.json`, and checkpoints.

## Config ladder

| Config | Params | Purpose |
|---|---:|---|
| `configs/smoke_mqar.yaml` | 38.6k | Mac/CPU tests and readout pilots |
| `configs/mqar_base.yaml` | 154k | fast iteration on MQAR |
| `configs/mqar_large.yaml` | 940k | paper-scale MQAR on the GPU node |
| `configs/smoke_a5.yaml` | 33.7k | Mac/CPU tests and A5 learnability |
| `configs/a5_base.yaml` | 96.4k | fast A5 transition comparisons |
| `configs/a5_large.yaml` | 623k | paper-scale A5 on the GPU node |
| `configs/parity_base.yaml` | 10.5k | parity sanity check |
| `configs/copy_base.yaml` | 126k | copying pilot |
| `configs/followups/h2_block_dense.yaml` | 188k | H2 compression grid on the block-dense state |
| `configs/followups/rank_sweep.yaml` | 234k | H3 `input_linear` rank sweep |
| `configs/followups/a5_slot_ablation.yaml` | 100k | H4 slot/readout ablation at matched state width |
| `configs/followups/a5_cascade_init.yaml` | 135k | cascade leakage-initialisation study |
| `configs/followups/mqar_hybrid.yaml` | 213k | hybrid gate/output-gate study |

The `large` configs are still small by LM standards; the paper’s claim is
mechanistic, not a scaling claim. Use `base` for iteration and `large` for the
final controlled runs.

## Pilot status (not results)

These are one-seed, tiny-model, short-training sanity runs. They validate the
harness; they are not evidence for the paper. Details and commands are in
[`research/code-and-pilot-log.md`](research/code-and-pilot-log.md).

- **A5 is learnable with the current design.** A 33.7k-parameter smoke model with
  a learned initial state, constant gate, and sharp block transition reaches
  **99.2% at length 8** and **86.0% at length 16** after 20k steps (train 99.5%).
  Earlier input-gated variants were at ~15% at length 8 after 2k steps; the
  constant-gate/learned-initial-state run was already at 29.7% by 2k. This
  validates the design; it is not a controlled ablation or a final A5 result.
- **MQAR readout pilot (5k steps, one seed).** `linear`, `gated_linear`, and
  `mlp` sit around 25% on 4 pairs; **`attention` reaches 48.6%** with a 6.7k
  readout; the low-rank full `C_t H_t` readout (`input_linear`) also reaches
  48.6% with 8.7k readout parameters; the `hybrid` is worse than attention alone
  (35.9%) despite more parameters. At matched ~6.7k readout parameters,
  `gated_linear` is 25.0% and `attention` is 48.2%. This is a promising H1
  readout signal, not a result: one seed, 5k steps, no LR sweep.
- **A readout state leak was found and fixed.** The first harness fed the
  post-layer residual to the readout's input side; that residual contains a
  projection of the recurrent state, so even the gated-linear gate was
  state-dependent. The readout now sees the **token-encoder output only**, and
  `test_readout_input_is_token_encoding_only` enforces it. Pre-fix runs (`runs/`
  without `*_v2`) are contaminated and should not be cited.
- The fact that the full `C_t H_t` readout matches attention suggests a sharper
  TMLR question: does state-dependent routing add anything beyond a rich
  input-dependent linear map? The next run is a 3-seed, LR-swept, matched
  comparison with a capacity ladder.
- **Parity sanity.** The 10.5k-parameter parity config reaches 100% train and
  100% at every evaluation length from 40 to 5120 after 2k steps. This validates
  the group-task and length-generalisation path; A5 is the hard case.

## What is implemented

### Recurrences (`model.state_type`)

| Value | Transition | Purpose |
|---|---|---|
| `diagonal` | per-channel scalar gate | the TC⁰ / no-mixing baseline |
| `block_diagonal` | dense `block_size × block_size` blocks, no cross-slot mixing | the block-dense lever |
| `cascade_chain` | block lower-triangular, nearest-neighbour slot coupling | cheap hierarchical cascade |
| `cascade_full` | block lower-triangular, all earlier slots | full cascade |
| `full` | all slots couple to all slots | dense control, expensive |

The code uses source slots `j <= i` (block **lower**-triangular with destination
`i` and source `j`). The theory notes state the equivalent block **upper**
triangular convention; the two differ only by slot ordering/transpose and have the
same closure and stability proofs.

For `cascade_chain`, `cascade_full`, and `full`, `model.cascade_init: block_diagonal`
starts the cross-slot logits closed (a learnable scalar gate initialized at
`-10`) and lets training open leakage. This separates “the cascade can contain
the block-diagonal solution” from “the cascade finds it under gradient descent.”
The default `random` preserves the original behaviour.

The transition is input-dependent and normalised either row-wise with a softmax
(`transition_norm: softmax`, positive, row-stochastic) or row-wise L1 after `tanh`
(`transition_norm: signed_l1`, signed, stable). By default the state update is a
gated convex combination of the mixed previous state and an input write:

```text
H_t = g_t ⊙ (A_t H_{t-1}) + (1 − g_t) ⊙ W_t u_t
```

with `g_t ∈ (0, 1)` computed from the input. This keeps the forward pass bounded
while allowing `g_t ≈ 1` for memory and `g_t < 1` for writing. The initial state is
a learned parameter (`initial_state: learned`) rather than zeros: a zero start plus
an input-only gate cannot both initialise on token 0 and retain afterwards without
positional information.

**Group tasks use `gate_mode: constant` with `constant_gate: 1.0`**, so the
recurrence is the pure transition `H_t = A_t H_{t-1}` from a learned initial state.
This removes the input-gate decay and writing confounds and isolates the transition
matrix, which is the object of study. MQAR and copying keep the input gate. A
constant gate is a special case of the affine recurrence, not a different model
class.

### Readouts (`model.readout`)

| Value | Form | Tests |
|---|---|---|
| `linear` | `W H_t` | state-only linear baseline |
| `input_linear` | `C(u_t) H_t` (low-rank `C = A(u)B`) | the standard input-dependent linear readout |
| `gated_linear` | `W_out(g(u_t) ⊙ W_s H_t)` | strong input-dependent but state-independent mixer |
| `mlp` | MLP over the flattened state | nonlinear state-only readout |
| `attention` | softmax over state slots with `Q(u_t)`, `K(H_t)`, `V(H_t)` | content-based depth attention |
| `hybrid` | gate between `attention` and `gated_linear` | does attention add anything on top? |
| `slot_gated` | per-slot linear logits + input-only softmax slot gate | H4: can input-only slot selection replace state-content attention? |
| `slot_max` | per-slot linear logits + hard max over slots | H4: non-attention slot-selection control |

`input_linear` is the closest implementation of the `C_t H_t` family in the theory.
`gated_linear` is the important control: its weights depend on the input but not on
the state, so it is still “linear readout” in the sense of the separation argument.
`attention` is the readout whose mixing weights depend on the state itself.
`slot_gated` can select slots but its gate sees only the input; `slot_max` selects
by the largest per-slot logit. The pair distinguishes **state-content attention**
from **slot selection**. The hybrid gate bias is configurable via
`model.hybrid_gate_bias_init`; positive values favour attention, negative values
favour the gated-linear branch.

The readout's input-side features are the **token encoder output only**, not the
post-layer residual. The residual contains a projection of the recurrent state;
feeding it to the readout would make the `gated_linear` and `input_linear` weights
state-dependent through their input argument and would break the baseline. The
regression test `test_readout_input_is_token_encoding_only` enforces this.

### Tasks

- **MQAR** (`task.name: mqar`): multi-query associative recall. Keys and values are
  separate tokens; a causal input convolution is available to bind adjacent
  key/value tokens. Only query positions contribute to the loss.
- **A5 / S5 / parity** (`task.name: group`, `task.group.group: A5 | S5 | Z2`):
  cumulative group multiplication, token-tagging loss at every position. Train on
  short lengths, evaluate on longer lengths.
- **Copying** (`task.name: copy`): copy a random content sequence after a delimiter.
- **Character LM** is planned for Phase 4 and is not wired in yet.

### Training

- AdamW, warmup + cosine/linear/constant schedule, gradient clipping.
- Optional CUDA fp16 autocast + GradScaler (`train.amp: true`); fp32 by default.
- Deterministic eval batches, multiple seeds, parameter counts, throughput, and a
  `run.json` per run.
- Evaluation returns per-length (or per-MQAR-config) loss/accuracy, which is what the
  phase diagrams and the readout gap `Δ = attention − linear` are built from.

## Layout

```text
configs/                     YAML experiment configs
src/readout_recurrence/
  config.py                  dataclass configs, YAML, dotted overrides
  seeding.py                 RNG and determinism helpers
  data/                      task generators
    base.py                  Batch / Task interface / EvalSpec
    mqar.py                  multi-query associative recall
    group.py                 A5, S5, parity
    copy.py                  copying
  models/
    layers.py                token encoder, block transition, recurrent layer
    readouts.py              linear / input-linear / gated / MLP / attention / hybrid / slot controls
    model.py                 full model and parameter counting
  training/
    loop.py                  training loop, eval, logging, checkpoints
    metrics.py               masked loss/accuracy helpers
  analysis/
    aggregate.py             run.json -> tidy CSV / summary
    plots.py                 readout-gap heatmaps and learning curves
  cli.py                     `train`, `sweep`, `aggregate` subcommands
tests/                       task, model, causality, and smoke tests
```

## Running the finding-first matrix

The core quantity is the readout gap

```text
Δ(task, config) = metric(attention) − metric(gated_linear)
```

with higher metric better (accuracy). A uniform positive gap is only a start; the
paper is looking for a sign change, a compression threshold, a slot effect, or a
positive finite-width routing result. The experiment plan in
[`research/experiment-plan.md`](research/experiment-plan.md) defines H1–H5 and the
gates.

A small MQAR sweep:

```bash
uv run readout-recurrence sweep \
  --config configs/mqar_base.yaml \
  --grid model.readout=linear,input_linear,gated_linear,attention \
  --grid model.state_type=diagonal,block_diagonal \
  --seeds 0,1,2 \
  --output-dir runs/mqar_sweep
```

Add `--match-params` to pad the smaller readouts with a residual linear adapter
to the largest readout budget within each recurrence geometry. This is a
**parameter-count** control; for the linear readout the effective number of
state functionals is still `out_dim × state_dim`, so report both the budget and
the natural readout size. The primary contrast is `gated_linear` vs. `attention`
at matched non-redundant widths; use `input_linear` as the full/low-rank
`C_t H_t` upper bound.

On the 4-GPU node, shard the sweep and pin each shard to one GPU:

```bash
uv run readout-recurrence sweep ... --shard 0/4 --device cuda:0
uv run readout-recurrence sweep ... --shard 1/4 --device cuda:1
uv run readout-recurrence sweep ... --shard 2/4 --device cuda:2
uv run readout-recurrence sweep ... --shard 3/4 --device cuda:3
```

Aggregate the results:

```bash
uv run readout-recurrence aggregate \
  --runs runs/mqar_sweep \
  --output runs/mqar_sweep/aggregate
```

## Follow-up controls

The first full run exposed four issues that the follow-up configs isolate:

- **H2 was swept on the diagonal state**, where the readout gap is zero. Run it
  on the block-dense state instead:

  ```bash
  make followup-h2
  # or:
  uv run readout-recurrence sweep \
    --config configs/followups/h2_block_dense.yaml \
    --grid model.readout=gated_linear,attention \
    --grid task.mqar.n_pairs=8,16,32,64 \
    --grid task.mqar.n_distractors=0,16,64 \
    --seeds 0,1,2 --match-params
  ```

- **The rank-8 `input_linear` is not a full `C_t H_t` control.** Sweep the rank:

  ```bash
  make followup-rank
  # model.readout_rank=8,16,32,64
  ```

- **The A5 gain is likely slot selection.** Keep the state width at 128 and vary
  the slot partition, with attention and non-attention slot readouts:

  ```bash
  make followup-slots
  # L=1,d=128; L=2,d=64; L=4,d=32; L=8,d=16
  # readouts: attention, slot_gated, slot_max, gated_linear, linear
  ```

- **The cascade did not match block-dense.** Compare random leakage
  initialisation with a block-diagonal start that must learn to open:

  ```bash
  make followup-cascade
  ```

- **The hybrid underperformed attention alone.** Test gate bias and the attention
  branch output gate:

  ```bash
  make followup-hybrid
  ```

All follow-up sweeps use `--match-params` where the comparison is between
readout families. The rank sweep deliberately does not, because rank is the
independent variable.

## Design notes and gotchas

- **Use `--match-params` for readout-family comparisons.** It adds a residual
  linear adapter to the smaller readouts within each recurrence geometry. Every run
  still logs total and per-module parameter counts; report both the natural readout
  size and the padded budget. For the linear readout, padding parameters does not
  increase its effective number of state functionals (`out_dim × state_dim`), so
  the adapter is a parameter-count control, not a capacity control.
- **The softmax transition is positive and row-stochastic.** It can represent
  permutations in the sharp limit, which is what the group tasks need, but it cannot
  represent signed dense transitions. Use `transition_norm: signed_l1` for that
  control.
- **`cascade_full` is the expensive option.** The transition projection materialises
  `n_slots × n_slots` block logits per token. Use it with small `n_slots` and
  `slot_dim`, or lower `model.transition_chunk_size`.
- **No FlashAttention, no Triton.** The recurrence is a chunked Python loop over
  time. At the target scale this is fine; if a run is unexpectedly slow, profile the
  per-chunk projections before optimising the loop.
- **Causality is tested.** `tests/test_causality.py` checks that changing a future
  token never changes past logits for every state type and readout.

## Related documents

- [`research/paper-plan.md`](research/paper-plan.md) — thesis, contributions, venue.
- [`research/theory-notes.md`](research/theory-notes.md) — Propositions 1–3.
- [`research/experiment-plan.md`](research/experiment-plan.md) — H1–H5, gates, phases.
- [`research/README.md`](research/README.md) — entry point to the research notes.
