# Paper Plan: The Readout Is Not the Recurrence

## Title

**The Readout Is Not the Recurrence: What Attention Over State Can and Cannot Buy**

Alternative titles:
- *Depth Attention Is a Readout, Not a Recurrence: Limits of Attention Over Recurrent State*
- *What Attention Over State Buys (and What It Doesn't): A Separation Between Readout and Recurrence*

## One-sentence thesis

Attention over recurrent state slots can strictly enlarge the **readout** function class — enabling content-based timescale selection and winner-take-all — but it cannot increase the **state-tracking** capability of an affine recurrence; the transition matrix is the lever, and the two effects dissociate empirically.

## Why this paper

The field is converging on multiple forms of "attention over state": DART over Mamba-2 chunk states, Depth-Attention/MoDA over transformer depth, MS-SSM's scale mixer over multi-resolution state, BD-LRU's block-dense channel mixing. The informal intuition — "if we attend over the state, the model becomes more expressive with fewer layers" — conflates two different claims:

1. **Readout expressivity:** which memory slot does the token read?
2. **Recurrence expressivity:** what can the state compute over time?

The first is real and useful for retrieval. The second is governed by the transition structure, not the readout. No paper we found states this distinction cleanly and tests it. That is the gap.

## Contributions

1. **A conceptual distinction** between readout expressivity and recurrence expressivity, with a precise definition of the standard readout family (`C_t H_t`) and the depth-attention readout (`softmax(q_t K(H_t)^T) V(H_t)`).
2. **Proposition 1 (separation):** depth attention strictly enlarges the readout class at fixed state; a linear readout `C_t H_t` with input-only weights cannot implement a winner-take-all selection for all states. Elementary proof, with the baseline family defined precisely.
3. **Proposition 2 (limit):** the readout is a fixed-depth post-processing of the state; it cannot increase the state-tracking class beyond the state update. For diagonal SSMs, the whole model stays in TC⁰ and cannot solve `A5` for unbounded length. Corollary of known circuit-complexity results, applied to this readout.
4. **Proposition 3 (cascade):** block upper-triangular input-dependent transitions are closed under multiplication (scan-compatible) and admit a row-wise L1 stability condition; they contain the block-diagonal model as a special case. Supporting result for the architecture variant, not the main claim.
5. **Empirical dissociation:** depth attention helps MQAR/associative recall and does not rescue a diagonal recurrence on `A5`; block-dense transitions help `A5` but do not necessarily improve recall. The two axes are separable in controlled tiny-scale experiments.
6. **A design principle:** use content-based readout for retrieval and timescale routing; use block-dense/cascade transitions for state tracking. Do not expect one to substitute for the other.

## What the paper is not

- Not a new SOTA architecture.
- Not a language-model scaling paper.
- Not a claim that the cascade is maximally expressive (that is a conjecture).
- Not a claim that depth attention is useless (it is useful for retrieval).
- Not a new circuit-complexity theorem (Proposition 2 is a corollary).

## Theory

See `theory-notes.md` for statements, proof sketches, assumptions, and the proof ladder.

The theory section should be short and honest:
- Proposition 1: separation at fixed state. Include the scalar construction and the general `L`-slot version.
- Proposition 2: composition bound + the diagonal/TC⁰ corollary. Cite Merrill et al. and Barrington; do not reprove TC⁰.
- Proposition 3: closure + stability + contains block-diagonal. One page.
- A "what we do not prove" paragraph: strict separation, maximal expressivity, finite-temperature gap. This honesty will help with reviewers.

## Experiments

See `experiment-plan.md` for the full matrix. The core is:

| Task | What it tests | Prediction |
|---|---|---|
| MQAR | associative recall | attention readout > linear/dense readout as #pairs/distractors grow |
| `A5`/`S5`/parity | state tracking | attention readout does **not** rescue a diagonal recurrence; block-dense does |
| Structure probes (anagram, order-k Markov, linear vs. random) | retrieval vs. structure | attention readout helps retrieval, not structure; block-dense helps structure |
| Copying | verbatim recall vs. state size | attention readout helps at fixed state; still bounded by state size |
| enwiki8/WikiText-103 char LM (5–10M params) | real-data sanity | no collapse; mechanism survives on natural data |

The experiments are deliberately tiny because the claim is mechanistic, not scale-dependent. Report multiple seeds, parameter-matched controls, and LR sweeps.

## Finding-first: designing for a surprising result

The expected dissociation ("attention helps recall, not state tracking") is probably true, but on its own it may read as obvious. To make the paper TMLR-strong and potentially main-conference-worthy, the experiments should search for one of these:

1. **Readout–tracking trade-off.** Depth attention improves MQAR but *degrades* `A5` relative to a linear readout on a block-dense state. A mechanism that helps one capability and hurts another is a real finding.
2. **Compression threshold.** Depth attention helps only when the state is over-compressed (many pairs relative to state size); below a threshold, a linear readout is as good. A phase diagram with a crossover is a clean law.
3. **Linear-mixer sufficiency.** A strong gated linear mixer matches softmax attention everywhere; the softmax is unnecessary. Contrarian and useful.
4. **Slots, not attention.** The gain comes from having multiple slots/timescales, not from attention; a linear readout over the same slots captures it. Isolates the mechanism.
5. **Positive finite-width routing.** Depth attention *does* help state tracking when the readout is the bottleneck (many blocks, small per-block width), even though it cannot change the asymptotic TC⁰ class. This gives a nuanced theory: **asymptotic class vs. finite-width routing**. It is the most main-conference-friendly result because it is positive and nontrivial.

The experiment plan adds a **finding-first matrix**: state size × number of slots × number of pairs × readout type × state type, with the readout gap `attention − linear` plotted as a heatmap per task. Look for sign changes, thresholds, and crossovers. Test #1, #2, and #5 explicitly.

The single most important control remains a **strong, parameter-matched linear mixer** (per-slot, input-dependent, gated), not a weak one. If a strong linear mixer matches softmax everywhere, that is the paper's finding.

## Paper outline

1. **Introduction**
   - The convergence on "attention over state" (DART, Depth-Attention, MoDA, MS-SSM, BD-LRU).
   - The conflation of readout and recurrence.
   - The two claims; the thesis; the contributions.
2. **Background and related work**
   - SSMs, linear attention, delta rule, block-dense transitions (brief).
   - Attention over state: DART, Depth-Attention, MoDA, Attention Residuals, MS-SSM, MS-GLA, BD-LRU/H-LRU.
   - Expressivity: TC⁰, Illusion of State, BD-SLiCE, PD-SSM.
3. **Setup and definitions**
   - State update, linear readout, depth-attention readout.
   - What "readout expressivity" and "recurrence expressivity" mean.
4. **Theory**
   - Proposition 1 (separation).
   - Proposition 2 (composition bound / TC⁰ corollary).
   - Proposition 3 (cascade closure/stability).
   - What we do not prove.
5. **Experiments**
   - MQAR: attention vs. linear/dense/MLP readout.
   - `A5`: diagonal vs. block-dense vs. cascade, with and without attention readout.
   - Structure probes: the dissociation.
   - Char LM: sanity check.
   - Ablations: `L`, causal depth mask, shared QKV, linear mixer, temperature.
6. **Discussion**
   - Design principle: readout for retrieval, transition for tracking.
   - Implications for hybrids: where to place depth attention, where to place block-dense layers.
   - Limitations: tiny scale, synthetic tasks, finite-temperature gap, no strict separation proof.
7. **Conclusion**
8. **Appendix**
   - Proofs and assumptions.
   - Extra ablations, seeds, hardware.
   - A note on the block upper-triangular conjecture.

## Venue strategy

- **Primary target: TMLR.** TMLR reviews correctness, clarity, significance, and reproducibility, not SOTA. The full proofs, the controlled dissociation, the phase diagram, and the reproducible tiny-scale harness are a good fit. Write for TMLR from the start; do not water the paper down for a workshop.
- **Conditional main-conference extension.** Only if the experiments produce a genuinely surprising law (a trade-off, a compression threshold, a positive finite-width routing result, or linear-mixer sufficiency). The extension would need at least one of: a stronger theorem, a 10–50M real-data study on the available 4-GPU node, a hybrid experiment, or a theory collaborator. Decide after the TMLR experiments, not before.
- **Workshops are optional.** Use them for feedback only if the timing is convenient; a workshop version is not the goal and should not replace the TMLR submission.
- **Do not target a main conference directly** without either a stronger theorem or 100M+ evidence. Neither is feasible on the available hardware, and a premature submission wastes a review cycle.
- **Preprint:** post to arXiv once the TMLR submission is ready; use feedback to strengthen the revision.

## Collaboration

The main risk is the proofs. The plan:

- **You:** conceptual framing, experiment design, implementation, analysis, writing.
- **Theory co-author:** tighten Proposition 1 (finite-temperature version), Proposition 2 (precision/closure conditions), and Proposition 3 (stability). A math/ML theory student or postdoc is the right fit.
- **How to find one:** post a short preprint/note with the theorem statements; reach out to authors of the expressivity papers (Merrill, Sarrof/Hahn, Cirone/Walker, Terzić) with a specific question; ask your network for a theory collaborator. A concrete theorem statement is the best recruiting tool.
- If no collaborator is available, aim for the TMLR version with clearly labeled proof sketches, and keep the claims modest. A workshop is a fallback only if the proofs cannot be completed.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| "The separation is trivial." | Frame it as a conceptual distinction + empirical dissociation, not a deep theorem. Add a quantitative separation (margin/communication complexity) if possible. |
| "The TC⁰ result is known." | Cite it as a tool; the contribution is the application to depth attention and the design implication. |
| "The experiments are too small." | The claim is mechanistic; include multiple seeds, parameter-matched controls, and a real-data sanity check. Discuss scale as future work. |
| "The cascade is incremental vs. BD-SLiCE." | Keep it as a supporting result, not the main claim. The main claim is the readout/recurrence distinction. |
| "The negative result is not publishable." | Pair it with a positive design principle and a constructive separation. TMLR/workshops accept rigorous analysis. |
| "No theory collaborator." | TMLR version with clearly labeled proof sketches and modest claims; a workshop is a fallback only if the proofs cannot be completed. Be explicit about what is proved vs. conjectured. |

## Compute reality

- 4×1080Ti/2080Ti: tiny synthetic sweeps; 4 independent runs; fp32 on 1080Ti, fp16 on 2080Ti; no bf16, no FlashAttention.
- M5 MacBook Air 24GB: prototyping, MPS debugging, CPU correctness checks.
- Colab/Kaggle free: overflow and reproducibility runs.
- No 100M–1B pretraining. The paper does not need it.

## What "done" looks like

- A TMLR submission (full paper, not a workshop note) with:
  - the three propositions and proofs/sketches,
  - the MQAR/A5/structure-probe dissociation,
  - the phase diagram / finding-first results (trade-off, threshold, or positive finite-width routing),
  - the char-LM sanity check,
  - the design principle,
  - a candid limitations section.
- A public code release with the tiny experimental harness, so others can reproduce and extend.
- A clear answer to the question: **does attention over state buy state tracking? No. Does it buy retrieval? Sometimes. Here is when.**

## Open questions

1. Is there a **finite-temperature** separation, or only the `τ→0` limit? (Theory.)
2. Is the cascade **strictly** more expressive than block-diagonal at fixed block size? (Theory; optional.)
3. Does the empirical dissociation hold at 10M–50M parameters, or only at 1–5M? (Experiments; feasible on the 4-GPU node.)
4. Does a **gated linear + softmax** readout beat both pure linear and pure softmax? (Experiments; cheap.)
5. Does depth attention help when the slots are **arbitrary heads** (Mamba-2 heads, BD-LRU blocks) rather than timescales? If not, the timescale semantics are doing the work. (Experiments; a useful negative result.)
