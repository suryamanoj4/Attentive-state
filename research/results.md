# Results: The Readout Is Not the Recurrence

Combined results from the full 60-run study (`102783`) and the 93-run follow-up
matrix (`103960` + `105433`). All 153 runs completed; there are no failed runs.

## Data and provenance

| Item | Value |
|---|---|
| Full study | 60 runs, 4×1080 Ti, job `102783` |
| Follow-up matrix | 93 runs, jobs `103960` (timed out at 60) and `105433` (completed the remaining 33) |
| Total completed | **153 runs**, 3 seeds per cell where applicable |
| Combined tidy data | `runs/combined_results/all_runs.csv` (1,110 metric rows) |
| Aggregated data | `runs/combined_results/summary.csv` |
| Generated tables | `runs/combined_results/key_tables.md` |
| Follow-up archive | `runs/server_followups_105433/readout_followups_105433.tgz` |
| Follow-up run tree | `runs/server_followups_105433/followups_103960/` |
| Full-run tree | `runs/server_102783/full_102783/` |
| Reproduction script | `scripts/combine_results.py` |

All percentages below are mean accuracy over three seeds unless noted. `n=3`
means three seeds.

## 1. A5 state tracking: the transition and its initialization are the lever

Full-run A5, trained on lengths 3–20 and evaluated at long lengths:

| State | Readout | Params | Wall (min) | len1280 | len2560 | len5120 |
|---|---|---:|---:|---:|---:|---:|
| block-diagonal | attention | 100,480 | 32 | 100.0 | 100.0 | **87.9** |
| block-diagonal | linear | 100,412 | 29 | 71.5 | 53.1 | 40.0 |
| cascade | attention | 134,528 | 32 | 39.0 | 32.0 | 19.1 |
| cascade | linear | 134,460 | 30 | 47.1 | 40.9 | 35.4 |
| diagonal | attention | 66,432 | 12 | 1.7 | 1.7 | 1.7 |
| diagonal | linear | 66,364 | 9 | 1.7 | 1.7 | 1.7 |

Two things are unambiguous:

- **Diagonal fails at 1.7% regardless of the readout.** Attention cannot make a
  diagonal recurrence represent A5 composition. This is the clean negative
  result: the readout is not the recurrence.
- **Block-dense works, and the readout is the bottleneck there.** Attention
  lifts 40.0% → 87.9% at length 5120 for +8.8% wall time and +68 parameters
  (0.07% of the model). Once the state has the right transition, the readout
  class determines whether that capacity can be used.

The cascade is the interesting middle case: with random initialization,
attention is *worse* than linear (19.1% vs 35.4%). The follow-up explains why.

### Cascade initialization: block-diagonal init unlocks attention

| Experiment | Init | Readout | Params | len1280 | len2560 | len5120 |
|---|---|---|---:|---:|---:|---:|
| full | random | attention | 134,528 | 39.0 | 32.0 | 19.1 |
| follow-up | random | attention | 134,528 | 38.9 | 31.8 | 19.1 |
| follow-up | block-diagonal | attention | 134,529 | 68.2 | 67.7 | **66.7** |
| full | random | linear | 134,460 | 47.1 | 40.9 | 35.4 |

Per-seed at length 5120:

| Init | seed 0 | seed 1 | seed 2 | Mean |
|---|---:|---:|---:|---:|
| random | 50.5 | 2.4 | 4.5 | 19.1 |
| block-diagonal | 100.0 | 2.4 | 97.6 | 66.7 |

Block-diagonal initialization raises the mean from 19.1% to 66.7% and beats the
cascade linear baseline (35.4%). Two of three seeds solve the task; one still
collapses. This is strong evidence for an **initialization/optimization barrier**
rather than pure expressivity: the attention readout can work on the cascade, but
random leakage initialization often traps it in a bad solution. It is not fully
solved, because one seed still fails.

## 2. MQAR: a capacity-dependent attention advantage

### H1 full run: high readout budget, 16 queries

| State | Readout | Budget | Params | Train | p8 | p16 | p32 | p64 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| block-diagonal | attention | 66,624 | 225,156 | 25.6 | **37.6** | **26.1** | **12.1** | **5.1** |
| block-diagonal | gated linear | 66,624 | 221,056 | 14.4 | 23.6 | 14.0 | 4.6 | 1.8 |
| diagonal | attention | 66,624 | 191,108 | 5.7 | 12.4 | 5.7 | 2.7 | 1.5 |
| diagonal | gated linear | 66,624 | 187,008 | 6.2 | 12.5 | 6.2 | 2.9 | 1.6 |

Paired attention − gated gaps:

| State | p8 | p16 | p32 | p64 | train |
|---|---:|---:|---:|---:|---:|
| block-diagonal | **+13.9** | +12.1 | +7.5 | +3.3 | +11.2 |
| diagonal | −0.0 | −0.5 | −0.3 | −0.2 | −0.5 |

At this budget, block-dense attention beats gated linear by 13.9 points at 8
pairs, with only +4,100 attention-specific parameters (+1.9% of total model
parameters). The diagonal gap is zero: again, no readout rescues the state.

### H2 follow-up: smaller budget, 8 queries — no replication

| Train condition | Readout | Budget | Params | Train | p8 | p16 | p32 | p64 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8 pairs / 0 dist | attention | 24,708 | 192,388 | 11.4 | — | 4.4 | 2.3 | 1.4 |
| 8 pairs / 0 dist | gated | 24,708 | 192,384 | **32.2** | — | **12.0** | 3.4 | 1.7 |
| 8 pairs / 64 dist | attention | 24,708 | 192,388 | 10.6 | 10.5 | 3.5 | 2.2 | 1.4 |
| 8 pairs / 64 dist | gated | 24,708 | 192,384 | **26.6** | **26.8** | **8.6** | 2.3 | 1.5 |
| 32 pairs / 64 dist | attention | 24,708 | 192,388 | 4.7 | 10.0 | 7.2 | 4.6 | 2.9 |
| 32 pairs / 64 dist | gated | 24,708 | 192,384 | 4.9 | 8.6 | 7.1 | 4.8 | 3.4 |
| 64 pairs / 64 dist | attention | 24,708 | 192,388 | 1.8 | 1.3 | 1.9 | 1.8 | 1.6 |
| 64 pairs / 64 dist | gated | 24,708 | 192,384 | 1.9 | 1.3 | 1.9 | 1.7 | 1.6 |

Paired attention − gated gaps across the H2 block-dense grid:

| p8 | p16 | p32 | p64 | train |
|---:|---:|---:|---:|---:|
| **−3.5** | **−2.2** | −0.2 | −0.2 | **−6.1** |

At 64 pairs both methods are at chance (~1.56%). The H2 grid does **not**
replicate the H1 attention advantage. Attention underfits at low compression
(train 11.4% vs 32.2% at 8/0), and gated linear is better at 8 pairs.

**Critical confound.** H1 and H2 differ in two ways:

| | H1 block | H2 block |
|---|---:|---:|
| Queries per sequence | 16 | 8 |
| Readout budget | 66,624 | 24,708 |
| Attention readout params | 57,476 | 24,708 |
| Gated readout params | 53,376 | 24,704 |
| Adapter policy | both rank-128 | attention none, gated rank-16 |

H2 is therefore a tighter readout/supervision regime, and the `--match-params`
procedure gave the adapter to gated but not to attention. The H2 result cannot
be read as a clean compression-only refutation of H1. It is a boundary
condition: **the attention advantage is capacity- and supervision-dependent.**
A bridge run at H1's budget and query count is needed to isolate cause.

### H2 full diagonal grid: the null result

The full-run H2 diagonal grid shows attention − gated gaps between −0.5 and +0.2
points across p8/p16/p32/p64 and all distractor conditions (n = 6–12 paired
cells). On a diagonal state, the attention readout has no advantage at any
compression level tested.

## 3. Rank sweep: rank alone is not the lever

Block-dense MQAR, train 16/16, 16 queries, `input_linear` readout:

| Rank | Params | Budget | p8 | p16 | p32 | p64 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 234,304 | 0 | 9.8 | 5.8 | 2.9 | 1.7 |
| 32 | 433,984 | 0 | **16.1** | **9.5** | **4.8** | **2.1** |
| 64 | 700,224 | 0 | 12.2 | 6.1 | 3.3 | 1.5 |

Rank-32 is best; rank-64 is not better. All are far below H1 attention (37.6%)
and gated (23.6%) at p8, even though rank-64 has ~3× the total parameters of the
H1 attention model. **Rank alone does not close the gap.** The result is
consistent with the readout *form* (state-dependent routing) mattering, not just
linear rank or parameter count. Caveat: `input_linear` was not parameter-matched
to the attention readout; its parameters live in the input projection.

## 4. Slot ablation: native L=4 geometry is special

A5 block-dense, total state width fixed at 128, so `slot_dim` changes with L:

| L | slot_dim | Readout | Params | Budget | len1280 | len2560 | len5120 | Entropy @5120 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 128 | attention | 211,453 | 36,733 | 68.6 | 67.6 | 56.2 | 0.000 |
| 1 | 128 | slot-gated | 211,262 | 36,733 | 47.1 | 26.8 | 14.4 | 0.000 |
| 1 | 128 | slot-max | 211,388 | 36,733 | 68.2 | 60.3 | 42.1 | — |
| 2 | 64 | attention | 137,342 | 28,542 | 32.6 | 17.5 | 9.5 | 0.000 |
| 2 | 64 | slot-gated | 137,340 | 28,542 | 60.2 | 46.2 | 31.4 | 0.013 |
| 2 | 64 | slot-max | 137,336 | 28,542 | 11.5 | 6.6 | 4.1 | — |
| 4 | 32 | attention | 100,480 | 24,448 | 100.0 | 100.0 | **88.0** | 0.004 |
| 4 | 32 | slot-gated | 100,344 | 24,448 | 99.4 | 91.6 | 66.9 | 0.471 |
| 4 | 32 | slot-max | 100,336 | 24,448 | 11.5 | 6.5 | 4.1 | — |
| 8 | 16 | attention | 82,436 | 22,404 | 62.2 | 48.5 | 41.5 | 0.032 |
| 8 | 16 | slot-gated | 82,288 | 22,404 | 7.6 | 4.6 | 3.2 | 0.439 |
| 8 | 16 | slot-max | 82,272 | 22,404 | 40.4 | 22.7 | 12.0 | — |

Per-seed at length 5120:

| L | Readout | seed 0 | seed 1 | seed 2 | Mean |
|---:|---|---:|---:|---:|---:|
| 1 | attention | 96.5 | 69.4 | 2.7 | 56.2 |
| 1 | slot-gated | 22.1 | 2.4 | 18.6 | 14.4 |
| 1 | slot-max | 81.9 | 2.3 | 42.1 | 42.1 |
| 2 | attention | 16.3 | 2.4 | 9.8 | 9.5 |
| 2 | slot-gated | 20.6 | 71.3 | 2.4 | 31.4 |
| 2 | slot-max | 2.4 | 7.6 | 2.3 | 4.1 |
| 4 | attention | 100.0 | 64.1 | 100.0 | **88.0** |
| 4 | slot-gated | 71.5 | 43.7 | 85.4 | 66.9 |
| 4 | slot-max | 2.4 | 2.4 | 7.6 | 4.1 |
| 8 | attention | 22.0 | 2.4 | 100.0 | 41.5 |
| 8 | slot-gated | 4.7 | 2.4 | 2.4 | 3.2 |
| 8 | slot-max | 6.5 | 4.3 | 25.2 | 12.0 |

At the native L=4 geometry, the ordering is clean: **attention 88.0 >
slot-gated 66.9 > slot-max 4.1**. Attention and slot-gated are well above chance
on every seed; slot-max is near chance. Soft, state-content attention beats an
input-only slot gate and a hard max. Hard slot selection never works for L>1.

But the effect is not monotonic in L. At L=2, slot-gated (31.4) beats attention
(9.5). At L=8, attention is wildly seed-dependent (22 / 2 / 100), and slot-gated
collapses. L is confounded with `slot_dim` and the block geometry at fixed state
width, so this is **not** evidence that “more slots help.” It is evidence that
the readout must match the state geometry; L=4 is the architecture's native
geometry and the only one where all three attention seeds work.

## 5. Hybrid gate: a practical middle ground

MQAR, train 16/16, 16 queries, output gate off, readout budget 45,316:

| Gate bias init | Params | p8 | p16 | p32 | p64 |
|---:|---:|---:|---:|---:|---:|
| 0 | 212,996 | 26.0 | 18.0 | 8.3 | 3.1 |
| +2 (favor attention) | 212,996 | **34.0** | **25.2** | **12.4** | **3.6** |

Reference: H1 block attention is 37.6 / 26.1 / 12.1 / 5.1; gated is
23.6 / 14.0 / 4.6 / 1.8. Bias +2 recovers most of the attention advantage at
p8–p32 without committing to a single readout class. It remains below attention
at p64 (3.6 vs 5.1). The hybrid is a reasonable robust default, but it pays for
both branches and its gate is not logged, so branch usage is not yet measured.

## 6. Efficiency snapshot

| Condition | Readout | Params | Wall (min) | Metric | Accuracy |
|---|---|---:|---:|---|---:|
| A5 block | attention | 100,480 | 32 | len5120 | 87.9 |
| A5 block | linear | 100,412 | 29 | len5120 | 40.0 |
| A5 cascade random | attention | 134,528 | 32 | len5120 | 19.1 |
| A5 cascade random | linear | 134,460 | 30 | len5120 | 35.4 |
| A5 diagonal | attention | 66,432 | 12 | len5120 | 1.7 |
| A5 diagonal | linear | 66,364 | 9 | len5120 | 1.7 |
| MQAR H1 block | attention | 225,156 | 82 | p8 | 37.6 |
| MQAR H1 block | gated | 221,056 | 81 | p8 | 23.6 |

Attention over state is \(O(TL)\), not \(O(T^2)\): there is no sequence-length
attention and no KV cache over time. At the H1 budget, attention over gated costs
**+1.9% total parameters** and **+13.9 points** at 8 pairs, with essentially the
same wall time. The readout itself is 18% of per-token compute in the block case
and 11% in the cascade case; the transition is the expensive object (block-dense
costs \(b\times\) the diagonal transition). In the diagonal case the readout is
53% of per-token compute because the constant-gate recurrence is nearly free —
and it buys zero accuracy. That is the worst-case regime for the intervention.

Caveat: wall-clock numbers are end-to-end, tiny-scale, and unoptimized; the
readout attention is not fused and materializes K/V at every token. Treat them as
scaling indicators, not deployment throughput.

## 7. The complete picture

### What the full matrix says

1. **State tracking is set by the transition and its initialization.**
   Diagonal A5 is 1.7% with every readout. Block-dense is 87.9% with attention and
   40.0% with linear. Cascade is 19.1% with random init and 66.7% with
   block-diagonal init (two of three seeds near 100%). The readout cannot create
   tracking capacity; it can only use the capacity the state already has.

2. **Attention readout is a capacity-dependent upgrade, not a universal win.**
   At the H1 budget and 16 queries, block-dense attention beats gated linear by
   13.9 points at 8 pairs for +1.9% total parameters. At the H2 budget and 8
   queries, gated linear matches or beats attention, and attention underfits.
   The H2 comparison is confounded by budget, adapter policy, and query count,
   but it establishes a real boundary: a starved attention readout is worse than
   a strong gated linear readout.

3. **Rank is not the mechanism.**
   Rank-64 `input_linear` has ~3× the parameters of the H1 attention model and
   still reaches only 12.2% at p8 vs 37.6% for attention. Rank-32 is the best
   linear-rank condition (16.1%) but still far below attention. The state-
   dependent softmax routing, not raw rank, is doing the work.

4. **Slot geometry matters, and the gain is not “more slots.”**
   At L=4, attention (88.0) > slot-gated (66.9) > slot-max (4.1), all seeds above
   chance. At L=2, slot-gated beats attention; at L=8, attention is seed-fragile
   and slot-gated collapses. L is confounded with slot width and block geometry.
   The defensible claim is “attention beats cheaper selectors at the native
   geometry,” not “attention scales with slots.”

5. **Initialization can dominate the readout.**
   Cascade random init makes attention worse than linear (19.1 vs 35.4);
   block-diagonal init raises attention to 66.7. This is the strongest new
   mechanism result: a readout can be expressively sufficient and still fail
   because optimization never reaches the useful transition. One seed still
   fails, so the barrier is reduced, not eliminated.

6. **The hybrid is a useful compromise.**
   Gate bias +2 recovers most of the attention advantage (34.0 vs 26.0 at p8)
   without choosing one readout class. It is not a free replacement for
   attention, and it pays for both branches.

### Updated thesis

The original thesis survives, but it needs a capacity and initialization clause:

> **The readout is not the recurrence.** Attention over state can enlarge the
> readout class and exploit a state that already contains the information, but it
> cannot increase the state-tracking class. The transition and its initialization
> are the lever for tracking. Whether attention readout helps is conditional on
> readout capacity, query supervision, slot geometry, and whether the transition
> can be optimized at all.

The clean dissociation is:

- **Diagonal A5:** readout is irrelevant; state is the bottleneck.
- **Block-dense A5:** state is sufficient; readout is the bottleneck; attention
  is a cheap, large win.
- **Cascade A5:** state is expressive enough, but random initialization hides it;
  initialization is the bottleneck; attention can win after block-diagonal init.
- **MQAR H1:** readout capacity is sufficient; attention wins.
- **MQAR H2:** readout capacity is tight; gated linear wins; attention underfits.
- **Slot L=4:** geometry matches the native state; attention wins.
- **Slot L=2/L=8:** geometry/optimization mismatch; cheaper selectors or
  attention instability dominate.

### What this means for the paper

- **TMLR is still the right primary target.** The paper now has a stronger
  mechanism story: a clean diagonal negative, a block-dense positive, a cascade
  initialization result, a slot-geometry boundary, and an honest capacity
  boundary on MQAR.
- **The cascade-init result is the most novel positive.** It separates
  expressivity from optimization and gives a concrete intervention
  (block-diagonal leakage initialization) with a large mean gain.
- **The L=4 slot result is the cleanest readout-class result.** Attention beats
  both an input-only slot gate and a hard max at the native geometry, all seeds.
- **The H2 result must be framed as a boundary, not a refutation.** The bridge
  run is required before claiming a general attention advantage. Without it, the
  MQAR claim is “attention helps at high readout budget and 16 queries; gated
  linear is better at low budget and 8 queries.”
- **A main-conference extension is still conditional.** The missing ingredient is
  a law: a measured capacity threshold (budget/query/slot geometry) that predicts
  when attention wins. The H2 bridge is the first step toward that law. The
  cascade-init and slot-geometry results could become the other axes.

### Recommended next experiments

1. **H2 bridge / capacity curve.** Fix the H2 task at 16 pairs / 16 distractors
   and `n_queries=16`; vary readout budget at 24.7k, 32k, and 66.6k; compare
   attention and gated with the same adapter policy (both adapted or neither).
   This isolates capacity from query supervision and adapter asymmetry.
2. **Cascade seed robustness.** Re-run the failing cascade seed with the same
   block-diagonal init, and vary the leakage-gate initialization / learning rate.
   Determine whether the 2.4% seed is a bad initialization or an optimization
   failure that needs a different schedule.
3. **Slot geometry control.** At fixed `slot_dim`, vary L; or at fixed L=4, vary
   `slot_dim`, to separate slot count from slot width and block structure.
   Compare attention, slot-gated, and a matched-compute hybrid.
4. **Hybrid gate diagnostics.** Log the learned gate values and branch usage.
   The bias +2 result is promising, but we do not yet know whether the model uses
   attention, gated linear, or both at convergence.

No new runs have been started for these; they are proposals for the next
iteration.

## 8. Caveats

- Three seeds per cell; several slot cells are seed-sensitive, so means alone are
  misleading. Per-seed tables are included.
- The slot ablation varies L and `slot_dim` together at fixed state width; it is
  not a pure slot-count manipulation.
- H2 differs from H1 in readout budget, adapter policy, and `n_queries`; it is a
  boundary condition, not a clean replication.
- Wall-clock numbers are end-to-end, tiny-scale, and unoptimized; they indicate
  scaling, not deployment throughput.
- No real language-model or 100M+ experiment. The claims are mechanism claims at
  tiny scale.
- The theory notes still need proof review; the empirical claims above do not
  depend on the unproven parts, but the paper's separation argument does.
