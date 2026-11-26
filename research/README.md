# Model Architecture Research: From State-Space Ideas to a Publishable Limits Paper

Working directory for the research that started from a "layered sandwich state + attention kernel" idea and converged on a smaller, more defensible paper:

> **The Readout Is Not the Recurrence: What Attention Over State Can and Cannot Buy**

The short version: attention over recurrent state slots can strictly enlarge the **readout** function class (content-based selection, winner-take-all, timescale routing), but it cannot increase the **state-tracking** capability of an affine recurrence. The transition matrix is the lever. The two effects dissociate empirically. That distinction is the paper.

## Documents

| File | Purpose |
|---|---|
| [`paper-plan.md`](paper-plan.md) | The finalized paper plan: title, thesis, contributions, outline, venue strategy, collaboration, risks. **Start here.** |
| [`theory-notes.md`](theory-notes.md) | Theorem statements and proof sketches: readout separation, composition/TC⁰ bound, block upper-triangular closure/stability, proof ladder. |
| [`experiment-plan.md`](experiment-plan.md) | Tiny-scale experiment matrix sized to the available hardware: MQAR, `A5`, structure probes, copying, char-LM sanity check, ablations, gates. |
| [`literature-survey.md`](literature-survey.md) | ~11k-word survey of RNNs, SSMs, CDEs, linear attention/delta rule, recall, multi-scale state, depth attention, expressivity, and parallelization. |
| [`idea-analysis.md`](idea-analysis.md) | The full analysis of the original architecture idea: four variants, prior-art map, what attention buys, the cascade, corrections to the informal claims. |
| [`code-and-pilot-log.md`](code-and-pilot-log.md) | The implementation status, design decisions that matter for the science, commands, and the first smoke/pilot runs (A5 learnability and the MQAR readout pilot). |

The downloaded source copies (`papers/`, `raw/`) are not tracked in git and have
been removed to keep the repository small. The survey cites primary sources by
arXiv ID; re-download a source if you need to re-check a specific claim.

## Chosen direction

**Paper:** *The Readout Is Not the Recurrence: What Attention Over State Can and Cannot Buy.*

**Core claims:**

1. **Separation.** Depth attention strictly enlarges the readout class at fixed state: a linear readout `C_t H_t` is input-dependent but state-independent; attention with `K_t = H_t W_K` can implement winner-take-all over slots, which no linear readout can do at the same state size.
2. **Limit.** The readout is a fixed-depth post-processing of the state; it cannot increase the state-tracking class beyond the state update. For diagonal SSMs, the whole model stays in TC⁰ and cannot solve `A5` for unbounded length. Block-dense transitions are the lever, not attention.
3. **Cascade (supporting).** Block upper-triangular input-dependent transitions are closed under multiplication (scan-compatible) and admit a row-wise L1 stability condition; they contain the block-diagonal model as a special case.
4. **Dissociation (empirical).** Depth attention helps MQAR/recall; it does not rescue a diagonal recurrence on `A5`; block-dense transitions help `A5` but do not necessarily help recall.
5. **Design principle.** Use content-based readout for retrieval and timescale routing; use block-dense/cascade transitions for state tracking. Do not expect one to substitute for the other.

**Finding-first.** The expected dissociation ("attention helps recall, not tracking") is probably true but may read as obvious. The experiments are designed to search for a stronger law: a **readout–tracking trade-off** (attention helps recall and hurts `A5`), a **compression threshold** (attention helps only when the state is over-compressed), **linear-mixer sufficiency** (a strong gated linear mixer matches softmax), **slots-not-attention** (the gain is from multiple slots), or **positive finite-width routing** (attention helps tracking when the readout is the bottleneck, even though it cannot change the asymptotic TC⁰ class). The last is the most main-conference-friendly result.

**Not claimed:** a new SOTA architecture, language-model scaling, maximal expressivity of the cascade, or a new circuit-complexity theorem (Proposition 2 is a corollary of known results).

## Compute reality

| Resource | Use | Constraints |
|---|---|---|
| 4×1080Ti/2080Ti (11GB each) | Main tiny-scale sweeps; 4 independent runs in parallel | 1080Ti: fp32, no tensor cores. 2080Ti: fp16, no bf16, no FlashAttention. |
| M5 MacBook Air 24GB | Prototyping, MPS debugging, CPU correctness | Some ops fall back to CPU; no CUDA/Triton. |
| Colab/Kaggle free (T4/P100 16GB) | Overflow, reproducibility, second environment | Session limits; slow I/O. |

**No 100M–1B pretraining.** The paper is a mechanism study; the models are 1–10M parameters plus a 5–10M char-LM sanity check. This is a deliberate constraint, not a limitation to apologize for.

## What not to target

- **A new SOTA architecture.** Without 100M–1B training, the claim is unsupportable and reviewers will reject it.
- **Language-model quality.** Same.
- **A pure survey.** Useful as a preprint/side artifact; not a peer-reviewed contribution on its own.
- **A pure philosophy paper.** The paper needs mathematical grounding; the three propositions provide it, even if the proofs are elementary.

## Venue strategy

- **Primary target: TMLR.** TMLR rewards correctness, clarity, significance, and reproducibility, not SOTA. Write the full paper for TMLR: proofs, phase diagram, controls, and a reproducible harness.
- **Conditional main-conference extension.** Only if the experiments produce a genuinely surprising law (a readout–tracking trade-off, a compression threshold, a positive finite-width routing result, or linear-mixer sufficiency). The extension would need a stronger theorem, a 10–50M real-data study, a hybrid experiment, or a theory collaborator. Decide after the TMLR experiments.
- **Workshops are optional**, for feedback only; they are not the goal.
- **Not:** a direct main-conference submission without a stronger theorem or 100M+ evidence.

## Collaboration

The main risk is the proofs. The plan is a theory co-author for Propositions 1–3 while the user handles framing, experiments, and writing. If no collaborator is available, submit the TMLR version with clearly labeled proof sketches and keep the claims modest; a workshop is a fallback only if the proofs cannot be completed.

## Status

- Literature survey: first pass complete.
- Idea analysis: first pass complete.
- Paper plan: finalized direction (limits paper), first pass complete. **Primary venue: TMLR; main conference conditional on a strong finding.**
- Theory notes: statements + proof sketches, proof ladder defined; stronger main-conference upgrades listed.
- Experiment plan: rewritten for the actual hardware; no large-scale phases; includes the finding-first matrix (H1–H5) and the conditional main-conference extension.
- Code: first implementation pass complete (`src/readout_recurrence/`, `configs/`, `tests/`, `Makefile`). MQAR, A5/S5/parity, copying, all five state types, six readouts, sweeps, aggregation, and plots are implemented. Tests pass.
- Pilot runs: A5 is learnable with the current transition/initial-state design (99.2% at length 8, 86.0% at length 16 after 20k tiny-model steps). On the corrected MQAR harness, attention reaches 48.6% on 4 pairs vs. 25.0% for matched gated linear; the full `C_t H_t` readout matches attention (48.6%), and the hybrid is worse (35.9%). One seed, 5k steps — a signal, not a result. See [`code-and-pilot-log.md`](code-and-pilot-log.md).
- A readout state leak was found and fixed: the readout input is now the token-encoder output only, not the post-layer residual. A regression test enforces it; pre-fix runs are contaminated.
- Next: (1) run the H1/H2 readout sweep on MQAR with 3 seeds and an LR sweep, including a full `C_t H_t` control, (2) run the A5 transition comparison at 50k steps, (3) test H5 finite-width routing, (4) investigate the hybrid failure, (5) tighten Proposition 1 and the scalar `L=2, d=1` proof.

## Code quickstart

The runnable project is at the repository root:

```bash
uv sync
uv run pytest
uv run readout-recurrence train --config configs/smoke_a5.yaml
uv run readout-recurrence train --config configs/smoke_mqar.yaml
make pilot-h1    # MQAR readout pilot
make pilot-a5    # A5 transition pilot
```

The README at the repository root documents the model, readouts, tasks, and
experiment commands. The design decisions that affect the science (learned
initial state, constant gate for group tasks, causal input convolution for MQAR,
row-normalised block transitions) are recorded in
[`code-and-pilot-log.md`](code-and-pilot-log.md).

## Suggested reading order

1. `paper-plan.md` — the thesis and the paper.
2. `theory-notes.md` — the three propositions and what still needs proof.
3. `experiment-plan.md` — the tiny-scale experiments and gates.
4. `idea-analysis.md` §1–4 for the original idea, §11 for the pivot to the limits paper.
5. `literature-survey.md` §1, §5, §7–8, §10 — the field map and the gap.

## Primary sources to read first

- SLiCE / BD-SLiCE: arXiv:2505.17761.
- BD-LRU / H-LRU: arXiv:2602.12021.
- MS-SSM: arXiv:2512.23824.
- DART: arXiv:2608.02032.
- Depth-Attention: arXiv:2606.05014.
- The Illusion of State: arXiv:2404.08819.
- Mamba-3: arXiv:2603.15569.
- How Many Heads Make an SSM: arXiv:2512.15115.
