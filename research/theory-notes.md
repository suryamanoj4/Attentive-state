# Theory Notes: Readout vs. Recurrence

**Status:** theorem statements and proof sketches for the limits paper. This is a scaffold, not a formal appendix. Each proposition lists its assumptions, a proof sketch, what still needs formalization, and its relation to prior work.

**Purpose.** The paper's mathematical core is narrow and elementary on purpose:

1. Depth attention strictly enlarges the **readout** function class at fixed state.
2. It does **not** enlarge the **state-tracking** class of the underlying recurrence.
3. A block upper-triangular cascade is scan-compatible and admits a simple stability condition.

The first is the conceptual separation; the second is the limit; the third supports the architecture variant. We are **not** proving maximal expressivity of the cascade — that is a conjecture and, if pursued, a separate paper.

---

## 1. Setup and notation

- Input `x_t ∈ R^{d_x}`, state `H_t ∈ R^{L×d}` (`L` slots, per-slot width `d`), output `y_t ∈ R^{d_y}`.
- State update: `H_t = F_t(H_{t-1}, x_t)`. We consider affine updates
  `H_t = A_t H_{t-1} + B_t x_t`
  with `A_t ∈ R^{Ld×Ld}`, `B_t ∈ R^{Ld×d_x}`. `A_t` may be input-dependent.
- Readout: `y_t = R_t(H_t, x_t)`.
- **Linear readout family:** `R_lin(H, x) = C(x) H`, with `C(x) ∈ R^{d_y×L}`. This covers the standard SSM readout `C_t H_t`, where `C_t` is a linear function of `x_t`. The weights depend on the input but **not on the state**.
- **Depth-attention readout:** `R_att(H, x) = softmax(q(x) K(H)^T / τ) V(H)`, with `q(x) ∈ R^{d_k}`, `K(H) = H W_K ∈ R^{L×d_k}`, `V(H) = H W_V ∈ R^{L×d_v}`. The softmax is over the `L` slots. `τ > 0` is a temperature.
- We write `H_i ∈ R^d` for the `i`-th slot (row of `H`).

The two readouts are compared **at the same state** `H` and the same input `x`. This is a statement about the readout function class, not about the sequence-level class. That distinction is the paper's point.

---

## 2. Proposition 1 — depth attention strictly enlarges the readout class

**Proposition 1 (separation).** Let `L ≥ 2`, `d ≥ 1`. There exists a function `f: R^{L×d} → R^d` and a depth-attention readout `R_att` such that `R_att(H, x) = f(H)` for all `H` and all `x`, but for every linear readout `R_lin(H, x) = C(x) H` there exists an `H` with `R_lin(H, x) ≠ f(H)`.

**Proof sketch (scalar case, `L=2`, `d=1`).**
Let `H = (h_1, h_2)` and `f(H) = max(h_1, h_2)`. Fix `x`; then `C(x) = (c_1, c_2)` is fixed. Suppose `c_1 h_1 + c_2 h_2 = max(h_1, h_2)` for all `H`.

- On the region `h_1 > h_2`: `c_1 h_1 + c_2 h_2 = h_1` for all such `H`. Rearranging, `(c_1 − 1) h_1 + c_2 h_2 = 0`. Since this holds on an open set, `c_1 = 1` and `c_2 = 0`.
- On the region `h_2 > h_1`: `c_1 h_1 + c_2 h_2 = h_2`, which forces `c_1 = 0` and `c_2 = 1`.
- Contradiction. Hence no fixed linear readout equals `max` everywhere.

For the attention side, take `q(x) = 1`, `W_K = W_V = I`, and let `τ → 0`. Then
`softmax([h_1/τ, h_2/τ]) · (h_1, h_2) → max(h_1, h_2)`
(up to ties, which are measure-zero). With a hard-attention variant (argmax instead of softmax), the equality is exact.

**Generalization.** For `L > 2`, `d > 1`, fix a query direction `q` (for example `q = e_1`) and define the score `s_i = <q, H_i>`. Let `f(H) = H_{i*}` with `i* = argmax_i s_i`. A linear readout computes a fixed linear combination of the slots; attention with `K_i = H_i`, `V_i = H_i`, fixed `q`, and `τ→0` selects the argmax slot. The same open-set/linearity argument applies. A cleaner generalization is to a **selection** function with a one-hot output; the proof is the same.

**What needs formalization.**
- Define the attention readout with a temperature parameter and either (a) state the separation in the `τ→0` limit, or (b) prove a finite-temperature separation with an explicit gap. Option (b) is stronger and avoids the limit; it requires a target function that is not in the linear span for any finite `τ`, e.g. a hard selection with a margin.
- Handle ties and measure-zero sets; use an open set where the argmax is unique.
- Make the input-dependence explicit: `C(x)` is fixed for each `x`, so the argument applies pointwise in `x`. The target `f` is independent of `x`.

**Relation to prior work.** The observation that a linear readout is state-independent in its weights is implicit in the SSM literature; the explicit separation is elementary. The paper's novelty is the **use** of this separation to explain the empirical dissociation, not the algebra. Cite Ghodsi (2025, arXiv:2512.15115) for the interaction-rank view and the kernel view of attention.

**Caveat.** This is a separation at **fixed state**, not a sequence-level separation. It is stated against the standard readout family `C_t H_t` with `C_t` computed from the current input (e.g., `C_t = Linear(x_t)`), not against an arbitrary nonlinear function of the entire past. If `C_t` were allowed to be an arbitrary function of the past, it could in principle encode state information implicitly, and the separation would not be meaningful. A linear readout with a nonlinear feature map of `H` could also compute `max`; the baseline is specifically the standard linear-in-`H` readout. State this clearly, and define the baseline family in the paper.

---

## 3. Proposition 2 — the readout cannot increase the state-tracking class

**Proposition 2 (composition bound).** Let `C` be a circuit class closed under composition with fixed-depth, poly-precision TC⁰ circuits. If the state sequence `(H_t)` is computable in `C` from `(x_t)`, and the readout `R_t` is a fixed-depth TC⁰ circuit, then the output sequence `(y_t)` is computable in `C`. In particular, if the state update is a diagonal SSM (TC⁰), adding a depth-attention readout keeps the model in TC⁰; it cannot solve `A5` for unbounded length (assuming TC⁰ ≠ NC¹).

**Proof sketch.**
- The state update for a diagonal SSM is in TC⁰ (Merrill, Petty, Sabharwal 2024, arXiv:2404.08819; Chen et al. 2024, arXiv:2412.06148).
- The readout is a fixed-depth circuit: matrix multiplies, a softmax over `L` slots, and a value mix. With poly-precision and bounded logits, softmax is TC⁰-computable; matrix multiplication of poly-size matrices is TC⁰.
- Composition of TC⁰ functions is TC⁰. Therefore `(H_t, y_t)` is TC⁰-computable.
- `A5` is NC¹-complete (Barrington 1989); if TC⁰ ≠ NC¹, no TC⁰ model solves `A5` for unbounded length. Merrill et al. use exactly this argument for S4/Mamba.

**General version.** The proposition does not depend on the specific class being TC⁰. For any state update in class `C`, a fixed-depth readout in a class `D` gives an output in `C ⋄ D` (composition). If `D ⊆ C` and `C` is closed under composition, the readout does not enlarge the class. For block-dense input-dependent transitions, the state update may not be TC⁰ (iterated products of general matrices are not TC⁰); the readout still does not enlarge **that** class. The clean statement is:

> The readout is a post-processing of the state; it cannot increase the state-tracking capability beyond the state update.

**What needs formalization.**
- The exact precision assumptions (log-precision vs. poly-precision) and the softmax approximation argument.
- The closure conditions on `C`.
- The `A5` reduction; cite Barrington and Merrill et al. rather than reproving it.
- The block-dense case: state clearly that the class may be larger than TC⁰ and that the readout still does not enlarge it.

**Relation to prior work.** This is a corollary of known circuit-complexity results, not a new theorem. The paper's contribution is applying it to the depth-attention readout and drawing the design implication. Be explicit about this in the paper; reviewers will otherwise say "this is just Merrill et al."

---

## 4. Proposition 3 — block upper-triangular cascade: closure and stability

**Proposition 3a (closure).** Let `A, B ∈ R^{Ld×Ld}` be block upper-triangular with the same `L`-block partition (i.e., `A_{ij} = 0` and `B_{ij} = 0` for `i > j`). Then `AB` is block upper-triangular.

(The implementation uses the lower-triangular convention `A_{ij} = 0` for `j > i`, with destination `i` and source `j`. This is the transpose/slot-reversal of the statement and has the same proof.)

**Proof.** For `i > j`,
`(AB)_{ij} = Σ_k A_{ik} B_{kj}`.
If `k ≤ j`, then `i > j ≥ k`, so `A_{ik} = 0`. If `k > j`, then `B_{kj} = 0`. Every term is zero. ∎

**Proposition 3b (stability).** Consider `H_t = A_t H_{t-1} + B_t x_t` with `A_t` block upper-triangular. If `||A_t||_∞ ≤ 1` for all `t` and `||B_t x_t||_∞ ≤ b` for all `t`, then `||H_t||_∞ ≤ ||H_0||_∞ + t b`; if `||A_t||_∞ ≤ 1 − δ` for some `δ > 0`, then `||H_t||_∞ ≤ (1 − δ)^t ||H_0||_∞ + b/δ`. A sufficient condition for `||A_t||_∞ ≤ 1` is row-wise L1 normalization over the full block row, including the leakage blocks and the input gate:
`Σ_j |(A_t)_{ij}| + Σ_j |(B_t)_{ij}| ≤ 1` for every row `i`.

**Proof.** The infinity-norm is submultiplicative: `||A_t H_{t-1}||_∞ ≤ ||A_t||_∞ ||H_{t-1}||_∞`. Induction gives the bound. The row-wise L1 condition implies `||A_t||_∞ ≤ 1` by definition. ∎

**Proposition 3c (contains block-diagonal).** Setting the off-diagonal (leakage) blocks to zero recovers a block-diagonal model. Hence the cascade inherits any function computable by the block-diagonal model with the same diagonal blocks.

**Proof.** Trivial by substitution. ∎

**What needs formalization.**
- The exact joint normalization for input-dependent diagonal blocks (BD-LRU style) extended to the full block row. The Gershgorin/∞-norm argument is enough for a sufficient condition; a tight condition is more work.
- The **strict separation** conjecture: is there a function computable by a 2-block cascade but not by a block-diagonal model with the same block size? This is plausible but not proved. Do **not** claim it in the paper unless a proof or a strong empirical separation is included.
- The **maximal expressivity** conjecture: if the diagonal blocks are dense and input-dependent, does the cascade inherit maximal probabilistic expressivity? Likely yes (it contains a maximally expressive block-diagonal sub-model), but the readout may not isolate the sub-model. This is a conjecture; keep it out of the main claims.

**Relation to prior work.** Block-diagonal closure/stability is in BD-LRU (arXiv:2602.12021) and BD-SLiCE (arXiv:2505.17761). The block upper-triangular extension is the new bit. The closure proof is one line; the stability condition is a standard ∞-norm argument. The contribution is the **combination** and its use in the architecture, not deep new mathematics.

---

## 5. What we are not claiming

- We are **not** claiming the block upper-triangular cascade is maximally expressive. That is a conjecture.
- We are **not** claiming the cascade is strictly more expressive than block-diagonal at fixed block size. That is an open question.
- We are **not** claiming depth attention is useless. It is useful for content-addressable retrieval.
- We are **not** claiming the tiny experiments settle language-model quality. They test a mechanism, not a scale frontier.
- We are **not** claiming a new circuit-complexity result. Proposition 2 is a corollary of known results.

### What a stronger (main-conference) version would need

These are optional and harder; the TMLR paper can stand without them.

1. **A finite-temperature separation.** Replace the `τ→0` limit with an explicit margin: a target function and a `τ` such that the attention readout approximates it within `ε`, while every input-only linear readout has error at least `δ`. This is the cleanest upgrade to Proposition 1.
2. **A strict cascade separation.** Exhibit a function computable by a 2-block upper-triangular cascade but not by a block-diagonal model with the same block size and parameter budget. This would turn Proposition 3 from a supporting lemma into a real expressivity result.
3. **A finite-width routing theorem.** Explain the positive finite-width routing hypothesis (H5): when the state is block-dense and the linear readout is the bottleneck, attention can route among blocks and improve tracking even though the asymptotic class is unchanged. This is the most main-conference-friendly theory result because it is positive and nuanced.
4. **A non-corollary composition/lower bound.** A result that is not just "the readout is post-processing" but characterizes a class of tasks where readout and recurrence trade off quantitatively.

None of these are required for TMLR. They are the difference between "a clean analysis" and "a new theoretical insight."

---

## 6. Proof ladder

Do these in order; each is self-contained.

1. **Separation (Proposition 1).** Write the scalar `L=2` proof cleanly; generalize to `L>2`; decide between the `τ→0` limit and a finite-temperature gap. This is the most important proof because it is the conceptual core.
2. **Composition bound (Proposition 2).** Write the reduction carefully; cite Merrill et al. and Barrington; state the general class version and the diagonal corollary. Do not reprove TC⁰.
3. **Cascade closure/stability (Proposition 3).** Write the one-line closure proof and the ∞-norm stability proof; connect to BD-LRU's joint normalization.
4. **Optional: strict separation / maximal expressivity.** Only if the first three are solid and there is time. This is the hardest part and the most likely to be incremental.
5. **Appendix.** Assumptions, edge cases (ties, finite precision, temperature), and the relation to prior work.

If step 1 or 2 stalls, submit the TMLR paper with the empirical dissociation + clearly labeled proof sketches and keep the claims modest. A workshop is a fallback only if the proofs cannot be completed; TMLR would require the full proofs for the strongest claims.

---

## 7. References

- Merrill, Petty, Sabharwal. *The Illusion of State in State-Space Models.* ICML 2024. arXiv:2404.08819.
- Merrill & Sabharwal. *The Parallelism Tradeoff: Limitations of Log-Precision Transformers.* 2022. arXiv:2207.00729.
- Barrington. *Bounded-Width Polynomial-Size Branching Programs Recognize Exactly Those Languages in NC¹.* 1989.
- Cirone et al. *Theoretical Foundations of Deep Selective State-Space Models.* 2024. arXiv:2402.19047.
- Walker et al. *Structured Linear CDEs.* NeurIPS 2025. arXiv:2505.17761.
- Dubinin, Orvieto, Effenberger. *Improved State Mixing in Higher-order and Block Diagonal Linear Recurrent Networks.* 2026. arXiv:2602.12021.
- Ghodsi. *How Many Heads Make an SSM?* 2025. arXiv:2512.15115.
- Brandoit et al. *Length-Independent State Tracking Under a Parallel Scan.* 2026. arXiv:2609.32447.
