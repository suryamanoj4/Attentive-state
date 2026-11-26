# Experiment Plan: Tiny-Scale Dissociation Study

**Goal:** test the paper's central prediction at a scale the available hardware can actually run:

> Depth attention over state slots improves content-addressable retrieval (MQAR, copying, NIAH) but does not rescue a diagonal recurrence on state tracking (`A5`/`S5`/parity); block-dense transitions improve state tracking but do not necessarily improve retrieval.

This is a **mechanism** study, not a scaling study. The models are 1–10M parameters and the datasets are synthetic, with a small character-level language-model sanity check. No 100M–1B pretraining.

---

## 0. Hardware reality

| Resource | Use | Constraints |
|---|---|---|
| 4×1080Ti/2080Ti (11GB each) | Main sweeps; 4 independent runs in parallel | 1080Ti: Pascal, fp32, no tensor cores. 2080Ti: Turing, fp16, no bf16, no FlashAttention. |
| M5 MacBook Air 24GB | Prototyping, MPS debugging, CPU correctness checks | Some ops fall back to CPU; no CUDA; no Triton. |
| Colab/Kaggle free (T4/P100 16GB) | Overflow runs, reproducibility, a second environment | Session limits; slow I/O. |

**What this rules out:** 100M–1B pretraining, large-scale architecture claims, long-context language modeling at scale. **What it does not rule out:** tiny synthetic models, mechanism claims, multiple seeds, parameter-matched controls.

---

## 1. Experimental discipline

These are non-negotiable; without them the results are confounded.

1. **Parameter matching.** Every readout variant must be compared at the same total parameter count (within ~1%). If the attention readout has more parameters, add a parameter-matched dense/MLP readout.
2. **Learning-rate sweep per architecture.** SSM recall success is confined to a narrow LR window (Okpekpe & Orvieto 2025). Sweep at least 5 LRs on a log grid (e.g., 1e-4 to 3e-3) for each architecture/readout; report the best and the sensitivity.
3. **Multiple seeds.** 3–5 seeds for the main claims; report mean ± std. Do not report a single seed.
4. **Same data, same steps, same batch size.** Match training tokens/steps across variants. If a variant is slower, do not give it more steps.
5. **No bf16.** 1080Ti: fp32. 2080Ti: fp16 with loss scaling. Do not use `torch.compile` on the old GPUs unless it is verified to help.
6. **Pure PyTorch.** No Triton, no FlashAttention. At this scale, a sequential loop is fine; a simple chunked scan is enough for speed.
7. **Log everything.** Params, FLOPs/token, throughput, peak memory, seed, LR, task config, git hash.

---

## 1.5 Finding-first matrix

The expected dissociation is a starting point, not the paper. The experiments should search for a **law** with sign changes, thresholds, or crossovers. Define the **readout gap**:

`Δ(task, config) = metric(attention readout) − metric(linear readout)`

where higher is better (accuracy, or negative BPC). Run the grid:

| Axis | Values |
|---|---|
| Task | MQAR, `A5`, parity, structure probes, copying |
| State type | diagonal, block-dense, cascade |
| State size / per-slot width | 64, 128, 256 |
| Number of slots `L` | 1, 2, 4, 8, 16 |
| Number of pairs / distractors (MQAR) | 4–64 / 0–4× |
| Readout | linear, dense, MLP, softmax, strong gated linear, softmax + linear gate |
| Seeds | 3–5 |

Plot `Δ` as a heatmap for each task. Look for:

- **Sign changes:** `Δ > 0` on MQAR and `Δ < 0` on `A5` (the trade-off hypothesis).
- **Thresholds:** `Δ > 0` only above a compression ratio `#pairs / state capacity` (the compression-threshold hypothesis).
- **Flatness:** a strong gated linear mixer gives `Δ ≈ 0` everywhere (linear-mixer sufficiency).
- **Slot effect:** `L` slots + linear readout matches `L` slots + attention (slots, not attention).
- **Positive routing:** block-dense with many blocks and small per-block width gives `Δ > 0` on `A5` (finite-width routing), even though the asymptotic TC⁰ class is unchanged.

Five hypotheses to test explicitly:

| ID | Hypothesis | Signature |
|---|---|---|
| H1 | Readout–tracking trade-off | `Δ_MQAR > 0`, `Δ_A5 < 0` |
| H2 | Compression threshold | `Δ` increases with `#pairs / state capacity`; zero below a threshold |
| H3 | Linear-mixer sufficiency | strong gated linear mixer has `Δ ≈ 0` vs. softmax |
| H4 | Slots, not attention | `L`-slot linear ≈ `L`-slot attention, both > 1-slot |
| H5 | Positive finite-width routing | block-dense + attention > block-dense + linear on `A5` when the readout is the bottleneck |

**Statistics.** Use paired seeds (same data order and init where possible), report mean ± 95% CI, and run a paired test for `Δ`. Do not interpret a single-seed difference.

**Surprise detectors.** A non-monotonic `Δ` (helps, then hurts as `L` grows), an interaction between state type and readout, or a task where the linear mixer wins are all more interesting than a uniform "attention helps recall." Log them.

---

## 2. Phase 1 — harness + MQAR

### 2.1 Harness

A single training script with switches for:
- state update: diagonal / block-dense / block upper-triangular cascade,
- readout: linear `C_t H_t` / dense `W H_t` / MLP / softmax depth attention / softmax + linear gate,
- `L` slots, per-slot width `d`, number of layers,
- task: MQAR / `A5` / parity / copying / structure probe / char LM.

The harness should run on CPU (for correctness), MPS (for debugging), and CUDA (for sweeps).

### 2.2 MQAR

- Sequence lengths: 256, 512, 1024.
- Key-value pairs: 4, 8, 16, 32, 64.
- Distractors: 0, 1×, 4× the number of pairs.
- Model: 2 layers, hidden 128, `L ∈ {2,4,8}`, `d ∈ {32,64}`, 1–5M params.
- Training: 10k–30k steps, batch 64, AdamW, LR sweep.
- Runtime: 10–30 min per run on a 2080Ti; 4 runs in parallel on the node.

**Prediction:** the attention readout beats the linear and dense readouts as the number of pairs/distractors grows; the gap is largest when the state is small relative to the number of pairs.

**Controls:** parameter-matched dense readout; MLP readout; linear scale mixer; attention with a shuffled/random query (to test whether content-based routing matters).

---

## 3. Phase 2 — state tracking

### 3.1 `A5` / `S5`

- Models: 1–2 layers, hidden 128, state 128–512; diagonal vs. block-dense (`b=4,8`) vs. block upper-triangular cascade.
- Readout: linear vs. depth attention, crossed with the state type.
- Training: 100k steps, batch 256, token-tagging loss (follow Merrill et al./SLiCE protocol).
- Evaluation: train length 3–20, test 20–5120; layers needed for >90%.
- Runtime: 1–3 h per run on a 2080Ti; 4 runs in parallel.

**Prediction:**
- Diagonal + attention readout: fails at long lengths (does not escape the TC⁰ limit).
- Block-dense + linear readout: solves `A5` with one layer (reproduces SLiCE/PD-SSM).
- Block-dense + attention readout: no significant gain over block-dense + linear; the transition is already doing the work.
- Cascade + linear readout: should match block-dense; whether it exceeds it is an open question.

**If the prediction fails** (attention readout rescues diagonal on `A5`), check for leakage: is the readout seeing future tokens, is the state carrying more information than intended, is the task length bounded? A genuine positive result would be surprising and worth investigating, but the default assumption is an implementation bug.

### 3.2 Parity / modular arithmetic

- Models: 1–2 layers, hidden 64–128.
- Training: length 3–40, test 40–256.
- Runtime: 1–2 h per run.

**Prediction:** same dissociation; parity is a finite-group task where the transition structure matters.

---

## 4. Phase 3 — structure probes and copying

### 4.1 Structure probes

Use the BD-LRU protocol:
- corrupted anagram detection,
- order-k Markov prediction,
- linear vs. random sequence classification.

Models: 2–4 layers, hidden 128–256, ~5–10M params; train on the probe datasets; compare readout types and state types.
Runtime: 2–4 h per run.

**Prediction:** block-dense/cascade improves structure probes; attention readout does not (or helps much less). This is the other half of the dissociation.

### 4.2 Copying

- Sequence lengths 64–1024; vocab 16–64.
- Models: 2 layers, hidden 128; state size varied.
- Runtime: 1–2 h per run.

**Prediction:** attention readout helps at fixed state size, but the model is still bounded by the state size (Repeat After Me, arXiv:2402.01032). The attention does not create an unbounded copy channel.

---

## 5. Phase 4 — character-level LM sanity check

This is a sanity check, not a scaling claim.

- Data: enwiki8 or WikiText-103 character-level.
- Model: 5–10M params, 4–6 layers, hidden 256–384.
- Training: 1–2 epochs (or a fixed token budget); AdamW; LR sweep on a small grid.
- Runtime: 1–2 days on 4 GPUs, or a few days on one 2080Ti.
- Metrics: bits-per-character; compare linear vs. attention readout at matched params; check that the model trains stably.

**Prediction:** the attention readout does not collapse; any gain is small. The point is to show the mechanism is not an artifact of synthetic data, not to claim a language-model win.

---

## 6. Phase 5 — ablations and analysis

| Ablation | Values | Question |
|---|---|---|
| Readout | linear / dense / MLP / softmax / softmax + linear gate | Is the gain from content-based selection? |
| `L` | 2, 4, 8, 16 | Does more slots help, or saturate? |
| Per-slot width `d` | 32, 64, 128 | Is the gain from slots or width? |
| State type | diagonal / block-dense / cascade | Which axis drives which task? |
| Cascade | none / nearest-neighbor / full upper-triangular | Does leakage help? |
| Block size `b` | 1, 2, 4, 8 | Where on the diagonal–dense ladder? |
| Depth mask | none / causal | Does slot order matter? |
| Slot embedding | none / learned bias / timescale init | Do slots need timescale semantics? |
| QKV sharing | shared / per-slot | Is the gain from parameters or routing? |
| Temperature | learned / fixed (0.1, 1, 10) | Is the softmax sharp or diffuse? |
| Query | content / constant / random | Does content-based routing matter? |

**Diagnostics:**
- Attention entropy over slots; fraction of tokens where the top slot is the slowest/fastest.
- Slot specialization: probe each slot's decay rate and content.
- Cascade utilization: performance drop when leakage blocks are zeroed after training.
- Readout rank: estimate the effective rank of the readout operator across tokens.

---

## 7. Gates

**Gate A (MQAR).** Attention readout beats the parameter-matched linear and dense readouts by more than seed noise as #pairs/distractors grow. If not, the empirical claim is weak; pivot to a pure theory/position paper or test a gated linear + softmax hybrid.

**Gate B (`A5`).** Attention readout does **not** rescue a diagonal recurrence; block-dense does. If the attention readout does rescue diagonal, investigate for leakage before interpreting.

**Gate C (dissociation and law).** The paper needs more than a uniform "attention helps recall." Pass if at least one of:
- **H1 trade-off:** `Δ_MQAR > 0` and `Δ_A5 < 0`.
- **H2 threshold:** `Δ` has a clear crossover as the state is compressed.
- **H3 linear-mixer sufficiency:** a strong gated linear mixer matches softmax everywhere (a contrarian finding).
- **H5 positive finite-width routing:** block-dense + attention beats block-dense + linear on `A5` when the readout is the bottleneck.

A uniform positive `Δ` on every task is interesting but weaker than a sign change or a threshold. If no dissociation and no law appears, the thesis is not supported at this scale; report honestly and consider a different readout (e.g., gated linear + softmax).

**Gate D (char LM).** The model trains stably and does not collapse; the attention readout is not catastrophically worse. If it collapses, debug the readout/normalization before drawing conclusions.

---

## 8. Metrics to report

- Task accuracy / exact match / BPC.
- Length generalization (train short, test long).
- Params, FLOPs/token, throughput (tokens/s), peak memory.
- Best LR and sensitivity (a small table of accuracy vs. LR).
- Attention entropy / slot specialization.
- Seed mean ± std.
- Wall-clock per run (for reproducibility).

---

## 9. Implementation notes

### 9.1 State updates

- **Diagonal:** `H_t = a_t ⊙ H_{t-1} + b_t x_t`; a sequential loop is fine at tiny scale; a `cumprod`/`cumsum` scan is easy for the diagonal case.
- **Block-dense:** `H_t = A_t H_{t-1} + B_t x_t` with `A_t` block-diagonal, dense `b×b` blocks; use a chunked loop or a simple scan.
- **Cascade:** block upper-triangular `A_t`; a sequential loop is fine for `L ≤ 8`; for speed, a chunked composition of small `Ld×Ld` matrices.
- **Stability:** row-wise L1 normalization over `[A_t, B_t]` (BD-LRU style); monitor the forward norm.

### 9.2 Readouts

- **Linear:** `y_t = C_t H_t`, `C_t = Linear(x_t)`.
- **Dense:** `y_t = W H_t` (flattened), parameter-matched.
- **MLP:** 2-layer MLP over the flattened state.
- **Depth attention:** `Q_t = x_t W_Q`, `K_t = H_t W_K`, `V_t = H_t W_V`, `y_t = softmax(Q_t K_t^T / τ) V_t`; optional per-slot bias, causal depth mask, and output gate initialized to zero.
- **Softmax + linear gate:** `y_t = g_t ⊙ (α_t V_t) + (1 − g_t) ⊙ (C_t H_t)`; test whether the combination beats both.
- **Keep `C_t` state-free.** Feed the readout's input-side features from the token encoder (or the current input) only, never from the post-layer residual. A residual contains a projection of the recurrent state; using it makes even the "linear" readout state-dependent through `C_t` and invalidates the separation. This was a real bug in the first harness; there is now a regression test for it.

### 9.3 Practicalities

- MPS: set `PYTORCH_ENABLE_MPS_FALLBACK=1`; use CPU for reference outputs.
- 1080Ti: fp32; 2080Ti: fp16 with `torch.cuda.amp`; no bf16.
- Run 4 independent configs on the 4 GPUs rather than data parallel; tiny models do not benefit from data parallel.
- Colab/Kaggle: use for overflow and to reproduce a subset in a second environment.
- Keep a `run.json` per experiment with config, seed, git hash, and metrics.

---

## 10. What to report regardless of outcome

1. The parameter-matched comparison table (params, state size, steps, throughput).
2. The LR sweep and the best LR per architecture.
3. The attention entropy / slot specialization analysis.
4. The cascade ablation.
5. The negative results: which tasks did not improve, and why (capacity, optimization, or inductive bias).
6. The theory note, even if the architecture does not win.

A well-controlled negative result on "depth attention over state" is valuable because the field is actively exploring DART, Depth-Attention, MoDA, MS-SSM, and BD-LRU, and the controlled comparison is missing.

---

## 11. Decision tree

```
Gate A (MQAR)
  ├─ attention readout ≤ linear/dense → empirical claim weak
  │     ├─ test gated linear + softmax hybrid
  │     └─ or pivot to theory/position paper
  └─ attention readout > baselines
        ↓
Gate B (A5)
  ├─ attention readout rescues diagonal → investigate leakage; if genuine, surprising positive result
  └─ attention readout does not rescue diagonal; block-dense does → expected
        ↓
Gate C (dissociation)
  ├─ retrieval↑ / structure↛, block-dense structure↑ → paper's thesis supported
  └─ no dissociation → thesis not supported at this scale; report honestly
        ↓
Gate D (char LM)
  ├─ stable, no collapse → include as sanity check
  └─ collapse → debug; do not over-interpret
```

The most likely outcomes:
- **A:** the attention readout helps MQAR but not `A5`; block-dense helps `A5` but not MQAR. The paper writes itself.
- **B:** a linear/MLP readout matches the attention; the attention is unnecessary. Report as a negative result with the theory.
- **C:** the attention readout helps both, or neither; the dissociation fails. Investigate why before writing.
- **D:** the cascade adds a small gain on top of block-dense; a supporting result, not the main claim.

---

## 12. Main-conference extension (conditional)

If the TMLR paper yields a strong law (H1, H2, H3, or H5), a main-conference version would need at least one of:

1. **A stronger theorem.** Finite-temperature separation; strict cascade-vs-block-diagonal separation; or a finite-width routing theorem that explains H5. This is the highest-value addition but the hardest.
2. **A 10–50M-parameter real-data study.** enwiki8/WikiText-103 first; a small FineWeb-Edu subset if the 4-GPU node can handle it. This tests whether the law survives natural data and a larger width.
3. **A hybrid experiment.** Put the depth-attention block inside a Mamba-2/Gated DeltaNet hybrid and measure the quality/cache Pareto frontier. This connects the finding to deployed architectures.
4. **A theory collaborator** who can sharpen the separation and the routing result.

Without one of these, a main-conference submission is unlikely. Decide only after the TMLR experiments; do not shape the TMLR paper around a speculative main-conference version.

### What would make the finding main-conference-worthy

- A **sign change**: attention helps recall and hurts tracking, with a mechanistic explanation.
- A **threshold**: attention helps only under severe state compression, with a measurable crossover.
- A **positive surprise**: attention helps finite-width tracking when the readout is the bottleneck, even though it cannot change the TC⁰ class.
- A **contrarian sufficiency**: a strong gated linear mixer matches softmax, undermining the attention-specific story.

A uniform "attention helps a bit on MQAR" is a TMLR result, not a main-conference result. Aim for a law.

### What to do if the law does not appear

- Publish the rigorous TMLR paper with the dissociation and the negative/weak result. That is a legitimate contribution.
- Do not inflate the claims or add a large-scale run that the hardware cannot support.
- Keep the code and the phase diagram public; a later collaborator or a larger compute budget can extend it.
