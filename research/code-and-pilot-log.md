# Code and pilot log

**Status:** first implementation pass complete; smoke and pilot runs executed on the
M5 MacBook Air (MPS). This is an engineering/validation log, not a results section.
Pilot numbers are one seed, tiny models, short training, and different parameter
counts; do not cite them as findings.

## What is implemented

The `uv` project lives at the repository root:

- `src/readout_recurrence/` — package.
- `configs/` — smoke and base YAML configs for MQAR, A5, parity, copying.
- `tests/` — task, model, causality, config, and end-to-end smoke tests.
- `runs/` — output directories (ignored by git).

The harness supports:

- **State types:** `diagonal`, `block_diagonal`, `cascade_chain`,
  `cascade_full`, `full`.
- **Transition normalisation:** row-wise softmax (positive, row-stochastic) and
  row-wise L1 after `tanh` (signed, stable).
- **Readouts:** state-only linear, low-rank input-dependent linear
  (`C_t H_t`), gated linear, MLP, depth attention over slots, and an
  attention + linear hybrid.
- **Tasks:** MQAR (separate key/value tokens, causal input convolution, only
  query positions in the loss), A5/S5/parity cumulative group multiplication,
  and copying.
- **Training/eval:** AdamW, warmup + cosine/linear/constant, gradient clipping,
  optional CUDA fp16, deterministic eval batches, per-spec evaluation,
  JSONL metrics, `run.json`, best/last checkpoints.
- **Sweeps/analysis:** Cartesian dotted-key grids, sharding for multiple GPUs,
  aggregation to tidy CSVs, and the readout gap
  `Δ = attention − baseline`, plus a heatmap helper.

### Design decisions that matter for the science

1. **Learned initial state.** A zero initial state plus a gated convex update has
   no position-free way to initialise on token 0 and then retain the state: the
   gate cannot know it is the first token. A learned initial state gives the model
   an identity-like starting representation without a BOS token. This is a
   constant affine bias, not a change to the recurrence class.
2. **Constant gate for group tasks.** For A5/S5/parity, the recurrence is run in
   `gate_mode: constant` with `constant_gate: 1.0`. The update is then a pure
   transition `H_t = A_t H_{t-1}` with a learned initial state. This isolates the
   transition matrix, which is the object of study, and removes the input-gate
   decay as a confound. MQAR and copying keep the input-dependent gate because
   they need to write and retain associations.
3. **Causal input convolution for MQAR.** Keys and values are separate tokens.
   A short causal convolution lets the value position see the preceding key, which
   is the binding mechanism. This mirrors Mamba-style short convolutions and is
   held fixed across readouts.
4. **Stable block transitions.** Transition blocks are row-normalised. The softmax
   variant can represent positive permutations in the sharp limit (enough for
   group actions); the signed-L1 variant allows signed permutation-like blocks and
   is the stronger control.
5. **Chunked recurrence.** Transition logits are computed in time chunks and the
   state update loops inside a chunk. This is pure PyTorch, no Triton or
   FlashAttention, and fits the target hardware.

## Config ladder

| Config | Params | Use |
|---|---:|---|
| `configs/smoke_mqar.yaml` | 38.6k | Mac/CPU tests, readout pilots |
| `configs/mqar_base.yaml` | 154k | fast MQAR iteration |
| `configs/mqar_large.yaml` | 940k | paper-scale MQAR on the GPU node |
| `configs/smoke_a5.yaml` | 33.7k | A5 learnability and tests |
| `configs/a5_base.yaml` | 96.4k | fast A5 transition comparisons |
| `configs/a5_large.yaml` | 623k | paper-scale A5 on the GPU node |
| `configs/parity_base.yaml` | 10.5k | parity sanity check |
| `configs/copy_base.yaml` | 126k | copying pilot |

## Commands used

Setup and checks:

```bash
uv sync
uv run pytest
uv run ruff check src tests
```

Smoke runs:

```bash
uv run readout-recurrence train --config configs/smoke_a5.yaml
uv run readout-recurrence train --config configs/smoke_mqar.yaml
```

A5 validation run (20k steps, smoke model):

```bash
uv run readout-recurrence train \
  --config configs/smoke_a5.yaml \
  --steps 20000 \
  --name smoke_a5_long \
  --set train.log_every=500 \
  --set train.eval_every=1000
```

MQAR readout pilot (one seed, 5k steps, smoke model):

```bash
uv run readout-recurrence sweep \
  --config configs/smoke_mqar.yaml \
  --grid model.readout=linear,gated_linear,input_linear,attention,mlp,hybrid \
  --seeds 0 \
  --output-dir runs/mqar_readout_sweep \
  --set train.steps=5000 \
  --set train.log_every=1000 \
  --set train.eval_every=1000 \
  --set train.eval_batches=4
```

Matched readout pilot (gated linear vs. attention, ~6.7k each):

```bash
uv run readout-recurrence sweep \
  --config configs/smoke_mqar.yaml \
  --grid model.readout=gated_linear,attention \
  --seeds 0 \
  --output-dir runs/mqar_matched_pilot \
  --match-params \
  --set train.steps=5000 \
  --set train.log_every=1000 \
  --set train.eval_every=1000 \
  --set train.eval_batches=4
```

Aggregate:

```bash
uv run readout-recurrence aggregate \
  --runs runs/mqar_readout_sweep \
  --output runs/mqar_readout_sweep/aggregate \
  --baseline gated_linear \
  --candidate attention
```

## Pilot 1 — A5 is learnable with the right state update

Smoke model: 33,724 parameters, `d_model=64`, `n_slots=2`, `slot_dim=32`,
`block_size=8`, one layer, `readout=linear`, `gate_mode=constant`, learned
initial state, `transition_temperature=0.5`.

| Step | Train acc | Len 8 acc | Len 16 acc |
|---:|---:|---:|---:|
| 1,000 | 20.9% | 15.4% | 8.5% |
| 2,000 | 32.1% | 29.7% | 15.9% |
| 5,000 | 93.8% | 89.0% | 65.6% |
| 10,000 | 98.4% | 97.0% | 77.4% |
| 20,000 | **99.5%** | **99.2%** | **86.0%** |

The curve is still improving at 20k steps, especially on length 16. The base A5
config uses 50k steps and the large config uses 50k steps, not the 2k-step smoke
number. The important validation is that the transition/initial-state design can
solve the task: earlier input-gated variants were at ~15% at length 8 after 2k
steps, while this run was at 29.7% by 2k and 99.2% at 20k. The training budget
also differs, so this is a design validation, not a controlled ablation of the
gate or initial state.

## Pilot 2 — MQAR readout comparison (corrected)

**Important correction.** The first version of the harness passed the post-layer
residual to the readout's input side. That residual contains a projection of the
recurrent state, so even the `gated_linear` gate was state-dependent through its
input argument. The first matched pilot was therefore flat by construction, not
by science. The fix: the readout's input features are the **token-encoder output
only**; only the attention keys/values (and the readout's state projection) see
the state. `test_readout_input_is_token_encoding_only` enforces this. All numbers
below are from the corrected harness.

Same recurrence, same data, one seed, 5,000 steps, smoke MQAR
(4 pairs + 2 distractors, 16 keys/values). Final eval:

| Readout | Total params | Readout params | Train acc | Pairs 4, no distractor | Pairs 8, no distractor |
|---|---:|---:|---:|---:|---:|
| `linear` | 32,912 | 1,040 | 25.6% | 25.4% | 10.9% |
| `gated_linear` | 36,528 | 4,656 | 25.2% | 25.0% | 12.9% |
| `input_linear` | 40,592 | 8,720 | 47.7% | 48.6% | 24.2% |
| `attention` | 38,578 | 6,706 | 46.5% | 48.6% | 25.0% |
| `mlp` | 34,480 | 2,608 | 24.2% | 25.0% | 11.5% |
| `hybrid` | 44,274 | 12,402 | 35.0% | 35.9% | 17.2% |

### What this does and does not say

- **Attention clearly helps retrieval at this scale.** It is ~48.6% on 4 pairs
  versus ~25% for the state-only linear, gated-linear, and MLP readouts. It does
  so with fewer readout parameters (6.7k) than the full input-dependent linear
  readout (8.7k). This is the direction the readout-separation argument predicts.
- **It does not show that attention is necessary.** The low-rank full
  `C_t H_t` readout (`input_linear`) matches attention (48.6%). The active
  ingredient may be rich input-dependent mixing rather than state-dependent
  softmax weights. This is a refined form of H3: the *diagonal* gated mixer is
  not sufficient, but the full input-dependent linear family can be.
- **The hybrid is worse than attention alone** (35.9% vs 48.6%) despite more
  parameters. The gated combination is not an ensemble win; it may interfere with
  optimization. Worth investigating, not a headline.
- **The ordering survives longer evaluation.** On 8 pairs, attention and
  `input_linear` are ~24–25%, versus ~11–13% for the state-only/gated readouts.
- **Still a pilot.** One seed, 5k steps, a tiny model, no LR sweep. The loss is
  still falling; the base/large configs use longer budgets.

## Pilot 3 — matched `gated_linear` vs. `attention`

The `--match-params` sweep adds a residual linear adapter to the smaller readout
until the budgets are within a fraction of a percent. Same recurrence, data,
seed, 5,000 steps:

| Readout | Readout params | Total params | Train acc | Pairs 4, no distractor | Pairs 8, no distractor |
|---|---:|---:|---:|---:|---:|
| `gated_linear` | 6,704 | 38,576 | 25.2% | 25.0% | 12.3% |
| `attention` | 6,706 | 38,578 | 45.9% | 48.2% | 24.8% |

At matched readout parameters, **attention wins by about 23 points on 4 pairs**.
The earlier flat result was the state leak. This is a clean readout signal for
the paper's direction (H1), but it is still one seed, 5k steps, and no LR sweep.

The fact that `input_linear` matches attention suggests the next question is not
"attention vs. gated linear" but "does state-dependent routing add anything
beyond a rich input-dependent linear map?" That is a better TMLR question than a
simple attention-is-better claim.

## Pilot 4 — parity sanity

`configs/parity_base.yaml` (10.5k parameters, one 2-wide block, constant gate,
learned initial state, gated-linear readout) reaches **100% train accuracy** and
**100% at every evaluation length from 40 to 5120** after 2,000 steps. This is a
sanity check that the group-task generator, the pure-transition recurrence, and
length-generalised evaluation work. It is not a paper result; A5 is the hard
state-tracking task.

## Immediate next experiments

1. **Matched H1/H2 on MQAR, 3 seeds.** Attention is ahead at 5k steps; test
   whether the gap survives an LR sweep and longer training. Add a full
   `C_t H_t` readout at matched rank and a capacity ladder. Sweep pairs/distractors
   for the compression threshold (H2).
2. **Investigate the hybrid failure.** It has more parameters but underperforms
   attention alone. Try initializing the branch gate near the attention branch,
   removing the output gate, or training the branches separately, then gating.
3. **A5 transition comparison.** `diagonal` vs `block_diagonal` vs
   `cascade_chain` with `linear` vs `attention` readout, 50k steps, 3 seeds.
   Prediction: diagonal + attention fails, block-dense + linear succeeds, and
   attention does not rescue diagonal.
4. **H5 finite-width routing.** Block-dense with many blocks and small per-block
   width; attention vs linear readout on A5. This is the most
   main-conference-friendly positive result if it appears.
5. **Cloud/GPU replication.** Reproduce the matched MQAR gap on the 4-GPU node and
   in a second environment (Colab/Kaggle) before trusting it.

## Caveats

- The smoke task is not the final task: 4 pairs / 16 values is small, and the
  model is 33k–44k parameters. The base and large configs are the next step.
- One seed, no LR sweep yet. SSM recall is known to be LR-sensitive; the pilot
  comparisons are not final until the LR is tuned per readout.
- The matched adapter controls readout **parameter count**, not effective linear
  rank. For the linear readout the number of state functionals is still
  `out_dim × state_dim`; report both the budget and the natural size.
- The A5 result is on a tiny model and short lengths; length generalisation to
  5120 is untested. The base run measures it.
- Pre-fix runs in `runs/` are contaminated by the readout state leak and should
  not be cited; the corrected runs are `*_v2`.
