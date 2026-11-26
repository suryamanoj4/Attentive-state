# Idea Analysis: A Depth-Attentive Multi-Timescale State

This document takes the idea as stated in the conversation, separates its parts, maps each part onto prior art, states the strongest defensible version, and lists what would falsify it.

---

## 1. The idea, stated precisely

The informal version has four ingredients:

1. **Split the recurrent state into layers/slots.** Instead of one monolithic state `h_t ∈ R^d` (or one matrix state `H_t ∈ R^{d×d}`), keep `L` slots `H_t ∈ R^{L×d}`. Slots are intended to have different timescales: shallow slots update fast, deep slots update slowly ("diminishing update rates").
2. **Make the slots convolved.** Apply a convolution across the slot dimension (and/or across time) so that neighboring slots mix locally.
3. **Put attention inside the kernel.** Instead of fixed convolution weights, compute attention weights over a small window — either over time positions or over state slots — and slide the kernel.
4. **Get global attention from local patches.** The hope is that sliding the attentive kernel over all positions yields "complete intertwined attention for the whole known state", and that this gives more capability per parameter/layer than a monolithic SSM.

There are two different axes hidden here, and they must be separated:

- **Time axis:** a local attentive kernel over tokens. A kernel of width `K` has receptive field `K`; stacking `n` layers gives `n(K−1)+1`. One layer is local, not global. This is the SASA / local-attention / dynamic-convolution axis. It is well studied and is **not** a route to global attention in one layer.
- **Depth/state axis:** a kernel over the `L` state slots. All `L` slots exist at time `t`, so one depth-wise kernel **does** see the whole state. This is the Depth-Attention / MoDA axis, applied to recurrent state instead of transformer layers. This is the interesting and less explored axis.

The phrase "complete attention of the whole state by doing small attentions in patches" is true only for the depth axis (or for a very large time kernel / many stacked layers). The analysis below assumes the depth axis is the intended one; the time-axis version is treated as a separate, already-known primitive.

---

## 2. Prior-art positioning

### 2.1 The diagonal-vs-dense result is the strongest existing argument for "less depth"

- Diagonal SSMs (S4D, Mamba's per-channel `A`) are not maximally expressive. Input-dependent dense / block-diagonal / DPLR / sparse transitions are (Cirone et al. 2024, arXiv:2402.19047; Walker et al. SLiCE, NeurIPS 2025, arXiv:2505.17761).
- **BD-SLiCE** uses block-diagonal transitions with dense `b×b` blocks. It solves the `A5` state-tracking benchmark with **one layer**; diagonal models need a number of layers that grows with sequence length. Its parameter count is `O(d_h Σ b_j²)` and its scan cost is `O(n d_h Σ b_j²)`.
- **PD-SSM** (Terzić et al. 2025, arXiv:2509.22284) can emulate any `N`-state finite-state automaton with one layer of dimension `N`, at a scan cost linear in `N`.
- **BD-LRU** (Dubinin et al. 2026, arXiv:2602.12021) shows that block-dense state mixing trains and improves structure probes at ~210M parameters, but its language-model perplexity gains are modest and DeltaNet/Mamba-2 still have lower perplexity at the same scale.

**Consequence.** If the goal is state tracking / formal-language expressivity, the literature already has a one-layer answer: block-dense (or PD-style) transitions. Attention over the state is orthogonal to that. The user's "less layers" claim is strongest on **state-tracking/structure** tasks, and weakest on **language-model perplexity**.

### 2.2 Multi-scale state is already crowded

- **MS-SSM** (Karami et al. 2025, arXiv:2512.23824) decomposes the input into `S+1` wavelet scales, runs an independent SSM per scale, and merges with an **input-dependent linear scale mixer** `E_t = Linear(x_t)`. It matches effective state size to baselines and reports gains on LRA, hierarchical reasoning, time series, and images. Its ablation found a data-dependent linear mixer beat a softmax mixer.
- **MS-GLA** (Dev et al. 2026, arXiv:2609.35664) distributes GLA heads across temporal resolutions and uses a learnable input-dependent fusion layer; reports up to 18.9% recall-intensive gains and 9.5% lower perplexity than GLA at matched params.
- **Harmonic** (Nyoma 2026, arXiv:2606.24650) stacks three SSM levels with progressively slower decays, passes prediction errors between levels, and reports large long-context gains at 28M–1B. Its ablation says the **timescale hierarchy**, not the prediction-error signal, is the load-bearing component. It is an independent preprint; treat the magnitudes as provisional.
- **Clockwork RNN** (2014) is the ancestor; **MELODI** (2024, arXiv:2410.03156) already uses "sandwich architecture" for a short-term/long-term memory hierarchy.

**Consequence.** "Layers with different decay rates" is not novel. The possible novelty is (a) the **cascade** between layers (slow layers receive from fast layers), and (b) the **content-based attention readout** across layers. Both need to be compared against MS-SSM's linear mixer and Harmonic's sum.

### 2.3 Depth attention is already a proven primitive in transformers

- **Attention Residuals** (Kimi Team 2026, arXiv:2603.15031) replaces the residual sum with a softmax over depth.
- **MoDA** (Zhu et al. 2026, arXiv:2603.15619) lets each attention head attend to depth KV pairs from preceding layers; +2.11% average downstream at 1.5B for 3.7% extra FLOPs.
- **Depth-Attention** (Zeng et al. 2026, arXiv:2606.05014) has the current query attend along depth over earlier layers' keys at the same position, mixes values into the value that self-attention reads, adds **no parameters**, and improves over vanilla by up to 2.3 accuracy points at 1.5B/3B, beating Attention Residuals and DenseFormer.

**Consequence.** "Attention over depth" is not a new idea in general. The novel question is whether it helps when the depth dimension is a **recurrent state with timescale semantics**, rather than a stack of transformer layers. That is a narrower and more defensible claim.

### 2.4 Attention over recurrent states is DART

- **DART** (Qian et al. 2026, arXiv:2608.02032) retains Mamba-2's chunk state contributions as `ΔH_[c]`, decodes token-conditioned keys and values from them, and runs state-memory attention over historical chunks, with a zero-initialized gated residual. It improves associative recall and retrieval and cuts the inference cache by 75% at chunk 256 / state 128.
- **Sessa** (2026, arXiv:2604.18580) puts attention inside a recurrent feedback path and proves power-law memory tails; it needs a triangular solve, not a plain scan.
- **Cross-Layer Value Routing** (Cerruti et al. 2026, arXiv:2607.07953) finds that routing a delta-rule layer's write **value** into the hidden stream gives a small validation-loss gain, while routing the write **error** does not.

**Consequence.** DART is the closest published system. The difference is the memory axis: DART attends over **time-chunk states** (`O(T/S)` memories), while the proposed idea attends over **depth/timescale slots** (`O(L)` memories at the current timestep). DART buys exact-ish retrieval of past chunks; the proposed idea buys cheap selection among timescales. They are complementary, not the same.

### 2.5 The honest gap

The intersection that is least covered:

> A recurrent state split into a small number of timescale slots, with a **block upper-triangular input-dependent transition** (so slots cascade and the scan is preserved), read out by a **softmax attention across slots** with depth positional embeddings and a causal depth mask.

Each component is published. The combination is not, as far as this survey found. The contribution would be the combination plus a controlled study, not a new primitive.

---

## 3. Formal variants

Let `x_t ∈ R^{d_x}` be the input, `H_t ∈ R^{L×d}` the state, `L` the number of slots, `d` the per-slot width. A generic linear state update is

```
H_t = A_t H_{t-1} + B_t x_t
```

where `A_t ∈ R^{Ld×Ld}` is input-dependent, and `B_t ∈ R^{Ld×d_x}`. The readout is

```
y_t = Readout(H_t, x_t)
```

The design choices are (i) the block structure of `A_t`, (ii) the readout, and (iii) whether the readout feeds back into the update.

### Variant A — readout-only depth attention (scan-safe)

**State update.** `A_t` is **block upper-triangular** with `L` diagonal blocks `A_t^{ℓ} ∈ R^{d×d}` and superdiagonal leakage blocks `C_t^{ℓ} ∈ R^{d×d}`:

```
A_t = [[A_t^1, C_t^1, 0, ...],
       [0,     A_t^2, C_t^2, ...],
       ...
       [0, ...,       A_t^L]]
```

Each diagonal block is dense and input-dependent (BD-LRU/BD-SLiCE style). The leakage blocks encode "fast slot feeds slow slot". The matrix is block upper-triangular, hence closed under multiplication; a parallel scan preserves the structure. Cost per composition is `O(L d^3)` for dense blocks (or `O(L d^2)` if the diagonal blocks are diagonal; `O(L b^3)` for block size `b`).

**Readout.** For each token:

```
Q_t = x_t W_Q                    ∈ R^{1×d_k}
K_t = H_t W_K                    ∈ R^{L×d_k}
V_t = H_t W_V                    ∈ R^{L×d_v}
α_t = softmax(Q_t K_t^T / √d_k + p)   ∈ R^{L}
y_t = α_t V_t + D x_t
```

where `p ∈ R^L` is a learned per-slot (timescale) bias or positional embedding, and `D` is a feedthrough. A causal depth mask is optional: if the slots are updated in parallel, no mask is needed; if they are updated in a cascade within the timestep, a mask (shallow → deep) prevents using an un-updated slot.

**Properties.**
- The recurrence is linear/affine and scan-compatible; the attention is a pointwise nonlinearity after the scan.
- The readout is a **content-based, input-dependent mixture** of slots. It can select one timescale per token (winner-take-all) or blend several.
- Parameter count: `O(L d²)` for the block transitions plus `O(L d d_k + L d d_v)` for the readout (or `O(d²)` if Q/K/V are shared across slots). This is comparable to, not necessarily smaller than, a dense readout.
- Compute: `O(L² d + L d²)` per token for the readout. With `L = 4–16`, the `L d²` projection term dominates; the `L²` term is small.
- It does **not** change the state-tracking class beyond the underlying block-dense recurrence. The readout is a function of the state, not a new state transition.

**What it tests.** H1: does content-based selection among timescale slots improve recall/structure at matched parameters over a dense/linear readout?

### Variant B — depth convolution + attention readout

Same as A, but before the attention, apply a short causal depthwise convolution across slots:

```
H'_t = DepthConv1D(H_t; kernel=3, causal=True)
```

Then `K_t = H'_t W_K`, `V_t = H'_t W_V`.

**Properties.** The conv is a fixed local mixing of slots; it can be folded into the `W_K/W_V` projections if it is linear and time-invariant, so it adds little expressive power beyond a banded projection. Its value is inductive bias: neighboring timescales should interact locally before the global softmax. The conv is cheap (`O(L d)` per token). This is the closest analogue of SASA/Involution, but along depth rather than time.

**What it tests.** H3: does local depth mixing before attention help beyond the attention alone? The control is Variant A with the same parameter count.

### Variant C — attention-coupled update (scan-breaking)

The state update itself uses a cross-slot attention:

```
H_t = A_t H_{t-1} + B_t x_t + γ · AttnSlot(H_{t-1}, x_t)
```

or the decay/write gates are computed from a cross-slot attention.

**Properties.** The state at time `t` now depends nonlinearly on `H_{t-1}` through a softmax, so the recurrence is nonlinear and a plain parallel scan does not apply. Options:
- **Chunked approximation**: within a chunk, use the previous chunk's state as a fixed context for the attention; across chunks, use the linear recurrence. This is a common trick and keeps most of the parallelism.
- **Triangular solve**: if the attention weights form a strictly lower-triangular matrix over time (as in Sessa), the update is a linear solve `(I − B) y = f`, which can be done with a triangular solve and has a recurrence form for inference.
- **Fixed-point iteration**: iterate the update to a fixed point; the number of iterations is a compute knob (Fixed-point RNNs).

**What it tests.** H4: does coupling attention into the update buy state-tracking/expressivity beyond a readout-only design? This is the expensive variant and should be attempted only after A/B show a signal.

### Variant D — delta-rule sandwich

Each slot is a matrix-valued fast-weight memory with a delta-rule update (DeltaNet/Gated DeltaNet), and a depth attention reads/writes across slots. This combines:
- block-dense transitions (BD-SLiCE),
- error-correcting writes (DeltaNet),
- multi-timescale slots (MS-SSM/Clockwork),
- content-based readout (the proposed attention).

**Properties.** The most expressive and the most expensive. It is essentially a "hierarchical fast-weight memory". It risks reinventing DART + Gated DeltaNet with extra steps. It is a later-stage variant, not the first experiment.

---

## 4. What the attention readout buys — and what it does not

### 4.1 It buys a nonlinear, content-dependent selection

A standard readout is `y_t = C_t H_t` with `C_t = Linear(x_t)`. This is **input-dependent but state-independent**: the mixing weights are a function of the current token, not of the state contents. The attention readout computes `α_t = softmax(q_t K_t^T)` with `K_t = H_t W_K`, so the mixing weights depend on **both** the input and the state. This is a bilinear interaction, and it can implement:
- **winner-take-all** over slots (select the slowest/fastest timescale for this token),
- **content-addressable retrieval** from the slot memory (query the slot whose content matches),
- **soft routing** that a linear `C_t` cannot express with the same parameter count.

This is a genuine addition to the function class at fixed `L,d`, because softmax is nonlinear in the state. It is the strongest technical argument for the idea.

### 4.2 It does not necessarily buy fewer parameters

A dense readout `W ∈ R^{d_out×Ld}` has `O(L d d_out)` parameters. A per-slot attention readout has `O(L d²)` parameters for Q/K/V plus `O(L d d_out)` for the output. A shared-QKV attention readout has `O(d²)` parameters, which can be **fewer** than a dense readout when `d_out ≈ d` and `L` is not tiny. But the attention also adds compute (`L² d`), and if the baseline readout is a simple linear `C_t` (which is `O(L d)` parameters), the attention is **more** expensive. The "less params" claim is only true relative to a dense readout, not relative to a linear one.

### 4.3 It does not give state tracking by itself

The state `H_t` is still updated by an affine map. The attention readout is a function applied after the state; it cannot compute a non-TC⁰ transition. For `A5`/`S5`/permutation composition, the transition matrix must be non-diagonal and input-dependent (or the recurrence must be nonlinear). BD-SLiCE/PD-SSM already solve these with one layer. A depth-attention readout on top of a diagonal Mamba state will not fix Mamba's state-tracking failure; a depth-attention readout on top of a block-dense state is redundant for that task.

**Where it can help:** associative recall, copy/retrieval, structure probes, long-context selection, and language modeling — tasks where the bottleneck is *which* memory slot to read, not *whether* the state can represent a group element.

### 4.4 It does not give global sequence attention

The attention is over `L` slots at the current timestep. Long-range information still flows through the recurrence. The model is still `O(1)` memory at inference (plus the `L×d` state), and still has the same fading-memory/compression limits. DART's chunk attention is the design that adds a growing memory; the proposed depth attention does not.

### 4.5 It is a form of "mixture of timescales"

MS-SSM's scale mixer is the linear analogue. The difference is softmax vs. linear. MS-SSM's ablation found linear better in their setting; that is a warning, not a verdict. A **gated linear + softmax** mixture (e.g., `α_t` from a softmax but with a learned gate that can also pass a linear combination) may be the best of both. The experiment should include:
- linear mixer (MS-SSM style),
- softmax attention,
- softmax + linear gate,
- dense readout,
- MLP readout.

### 4.6 Interaction-rank framing

"How Many Heads Make an SSM?" (Ghodsi 2025, arXiv:2512.15115) gives a useful language: a linear SSM's lag operators span a `k`-dimensional interaction space; a multi-head factorized model needs `H ≥ k` heads to match it. A depth-attention readout with `L` slots can be viewed as increasing the number of independent value maps that can be mixed per token. The precise claim to test is:

> At fixed state size and parameter count, does depth attention increase the effective interaction rank (or the number of distinguishable lag operators) relative to a linear readout?

This is measurable: estimate the rank of the empirical readout operator across tokens, or compare the number of distinct impulse responses the model can realize.

---

## 5. The cascade: block upper-triangular transitions

### 5.1 Why it is scan-compatible

Block upper-triangular matrices are closed under multiplication. If `A_t` is block upper-triangular for every `t`, then any product `A_{t} A_{t-1} ... A_{s}` is block upper-triangular. A parallel scan over the recurrence composes these products, so the structure is preserved. This is the same argument BD-SLiCE uses for block-diagonal matrices, extended to the superdiagonal.

Cost: if each block is `b×b`, composing two block upper-triangular matrices costs `O(L b^3)` (or `O(L² b^3)` if the leakage blocks are full and the product is dense in the upper triangle; in practice, a banded cascade with only nearest-neighbor leakage costs `O(L b^3)`). With `L=8`, `b=16`, this is manageable.

### 5.2 Stability

BD-LRU's joint gate normalization (row-wise L1 over `[A_t, a_0,t]`) guarantees a bounded forward pass for block-diagonal transitions. For a block upper-triangular cascade, the analogous condition is row-wise L1 over the **full block row**, including the leakage blocks and the input gate. A sufficient condition: for every block row `ℓ`, the sum of the absolute values of the entries in `[A_t^{ℓ}, C_t^{ℓ}, B_t^{ℓ}]` is ≤ 1. This gives a Gershgorin-style bound on the eigenvalues and a bounded `||H_t||_∞`. Proving this and testing it is a concrete, tractable contribution.

### 5.3 Expressivity

The BD-SLiCE theorem covers block-diagonal `A`. A block upper-triangular `A` is not block-diagonal, so the theorem does not directly apply. Intuition: if the diagonal blocks are dense and input-dependent, the model already contains a maximally expressive block-diagonal sub-model; the leakage blocks can only add expressivity, not remove it. But "contains a maximally expressive sub-model" is not the same as "is maximally expressive" — the readout may not be able to isolate the sub-model's state. A proof would likely show that the cascade is a triangular linear system whose diagonal blocks are maximally expressive, and that the readout can be chosen to ignore the leakage; this is plausible but needs care.

A more interesting question: is the cascade **strictly** more expressive than block-diagonal at fixed block size? The leakage blocks add cross-timescale interactions that a block-diagonal model can only emulate by widening the blocks or adding layers. If the answer is yes, that is a genuine theoretical contribution.

### 5.4 Relationship to H-LRU

H-LRU uses a companion matrix per coordinate: it mixes **past values of the same coordinate**, not different slots. The cascade mixes **different slots at the same timestep**. They are different axes: H-LRU is temporal state mixing, the cascade is channel/depth state mixing. BD-LRU is instantaneous channel mixing within a block; the cascade is instantaneous channel mixing **across** blocks. The four combinations (temporal × channel, within × across) form a useful design table.

---

## 6. Naming and the MELODI collision

MELODI (2024) already uses "sandwich architecture" for a short-term multi-layer recurrent compression plus a long-term memory layer. If we use "sandwich state", we should:
- cite MELODI,
- define our meaning precisely (a block upper-triangular cascade of timescale slots with a depth-attention readout),
- avoid claiming the word as novel.

A more descriptive name: **Cascaded Timescale State (CTS)** or **Depth-Attentive State Space Model (DA-SSM)**. "Sandwich" is fine informally but weak for a paper.

---

## 7. Corrections to the informal claims

1. **"BD-SLiCE achieves max expressiveness by a small batch of the dense matrix at a time."**
   Not quite. BD-SLiCE is a **block-diagonal** transition: the state is partitioned into blocks, and each block has a dense `b×b` matrix. It is not a batch of dense matrices applied to the whole state. The maximal-expressivity result is asymptotic in the largest block size. At fixed width, block size is a compute–expressivity trade-off.

2. **"In SSMs the architecture itself is already a convolution once unrolled."**
   True for time-invariant SSMs (S4/S4D/DSS), which unroll to a long convolution. False for selective SSMs (Mamba/S6, Mamba-2, Mamba-3, Gated DeltaNet), which are time-varying and use scans/chunkwise algorithms. Mamba also has a short depthwise convolution on the input; that is a local feature extractor, not the recurrence.

3. **"A small attention kernel sliding over all positions gives complete attention of the whole state."**
   For a time kernel, no: the receptive field is `K` per layer, `n(K−1)+1` after `n` layers. For a depth kernel over `L` slots, yes: all slots are visible at once. The claim is true only on the depth axis.

4. **"Attention over the state gives state tracking."**
   No, not if the state update is affine. The transition matrix is the lever. Attention over the state can improve *which* memory is read, but it does not change the recurrence's ability to compose permutations.

5. **"Less layers because the state is richer."**
   Task-dependent. For state tracking, block-dense transitions already give one-layer solutions; attention is not needed. For language modeling, BD-LRU's block-dense state gives modest perplexity gains and larger structure-probe gains; the depth savings are not established. For parameter efficiency, fixed-point/looped models are the stronger existing answer.

6. **"Attention is the mechanism that captures relativeness in the state."**
   It captures **content-based selection** among state slots. "Relativeness" is vague; the precise mechanism is a query-key interaction between the current token and the slot contents, followed by a softmax. That is a useful inductive bias, but it is not the same as relational reasoning or state tracking.

---

## 8. The strongest defensible version

**Hypothesis.** At a fixed parameter budget, depth, and state size, a recurrent block whose state is split into `L` timescale slots and read out by a **content-based attention across slots** will outperform:
- the same block with a linear/dense readout,
- a block-diagonal model with the same state size (BD-LRU/BD-SLiCE),
- a multi-scale model with a linear scale mixer (MS-SSM),
on associative recall and structure-probe tasks, **without** a growing KV cache;
and it will match or slightly improve language-model perplexity at the same compute.

**Why it could be true.** The attention readout adds a nonlinear, state-dependent selection that a linear mixer cannot express. It lets the model route each token to the appropriate timescale (fast local syntax vs. slow global semantics) without widening the state or adding layers. The cascade gives the slots a principled ordering and cross-timescale leakage, and remains scan-compatible.

**Why it could be false.**
- A dense/MLP readout may capture the same selection with more parameters but better optimization.
- A linear scale mixer may be enough (MS-SSM's ablation).
- The gain may be limited to synthetic recall and vanish at language-model scale.
- The optimization brittleness of SSMs may swamp the architectural gain (Revisiting associative recall).
- DART-style chunk attention may dominate for the tasks where depth attention helps.
- The `L²` attention and the per-slot projections may cost more than they return at small `L`.

**What would falsify it.**
- No improvement over a parameter-matched dense readout on MQAR/NIAH at 100M scale.
- Improvement only when the attention adds parameters/compute (i.e., the baseline is under-parameterized).
- No improvement over MS-SSM's linear mixer or BD-LRU's block-dense state on structure probes.
- The attention weights collapse to a single slot or become uniform (no routing).
- A fixed-point/looped baseline matches it with fewer parameters.

---

## 9. Design recommendations

If we build this, start with the smallest falsifiable version:

1. **Variant A (readout-only)**, `L = 4` or `8`, per-slot width `d = 64` or `128`, dense diagonal blocks, nearest-neighbor cascade, shared Q/K/V across slots (to keep parameter count honest), learned per-slot bias, no depth mask initially (slots updated in parallel).
2. **Baselines**: (a) same block with a linear `C_t` readout; (b) same block with a dense `W` readout; (c) BD-LRU with the same state size; (d) MS-SSM-style linear scale mixer; (e) Mamba-2; (f) Gated DeltaNet; (g) a small Transformer; (h) if code is available, DART/Depth-Attention.
3. **Tasks first**: MQAR (varying pairs and distractors), `A5`/`S5`/parity/mod-arithmetic, copying, NIAH, and a structure probe (anagram, order-k Markov, linear vs. random). These are cheap and discriminate.
4. **Then, if compute were available, language modeling**: 100–200M parameters, 10–20B tokens on FineWeb-Edu, matched tokens and compute. Sweep learning rate per architecture (critical). Report perplexity, NIAH, and structure probes, not just perplexity. **This is not part of the chosen plan** because the available hardware cannot support it; the chosen paper is a tiny-scale mechanism study (see `paper-plan.md`).
5. **Ablations**: readout type, depth mask, per-slot embeddings, shared vs. per-slot QKV, cascade on/off, block size, `L`, gate normalization.
6. **Diagnostics**: attention entropy over slots; which slot is read at which token position; whether slots specialize to timescales (probe the decay rates); whether the model uses the cascade (ablate the leakage blocks).
7. **Only then** consider Variant C (attention-coupled update) or D (delta-rule sandwich).

The single most important control: **a parameter-matched dense readout and a linear scale mixer**. If the attention readout cannot beat those, the idea is not doing what it claims.

---

## 10. Open theoretical questions

1. Are block upper-triangular input-dependent transitions maximally expressive? Under what conditions on the diagonal blocks and the leakage?
2. Is the cascade strictly more expressive than block-diagonal at fixed block size? If so, what is the smallest separating function?
3. What is the interaction rank of a depth-attention readout as a function of `L`, `d`, and the number of heads? Does it exceed a dense readout at matched parameters?
4. What is the finite-precision state-tracking ceiling of a block upper-triangular affine recurrence? Does the cascade help or hurt the contraction/separation trade-off?
5. Can the cascade + attention be fused into a single kernel that keeps the scan and the softmax in one pass? What is the roofline?
6. Does content-based readout help when the state slots are **not** timescales but arbitrary heads (e.g., Mamba-2 heads, BD-LRU blocks)? If it only helps with timescale semantics, that is a useful negative result.
7. How does depth attention interact with the hybrid placement (attention layers + recurrent layers)? Does it reduce the number of full-attention layers needed?

---

## 11. Chosen direction: from architecture proposal to limits paper

The architecture analysis above is the background. The finalized research direction is narrower, because it must fit the available compute (4×1080Ti/2080Ti, an M5 MacBook Air, and free Colab/Kaggle) and because the architecture space is already crowded.

### 11.1 Why pivot

- **Compute:** a new architecture claim needs 100M–1B training to be credible. The available hardware cannot support it. A mechanism claim with 1–10M-parameter models can be tested rigorously on the available hardware.
- **Crowding:** BD-SLiCE/PD-SSM already give one-layer state tracking; MS-SSM/MS-GLA/Harmonic already cover multi-scale state; DART already covers attention over recurrent chunk states; Depth-Attention/MoDA already cover attention over depth. A new block would be judged against all of them, at scale.
- **The most defensible contribution is conceptual:** the field conflates *readout expressivity* (which memory does the token read?) with *recurrence expressivity* (what can the state compute over time?). The distinction is clean, useful, and testable at tiny scale.

### 11.2 The paper

**Title:** *The Readout Is Not the Recurrence: What Attention Over State Can and Cannot Buy.*

**Claims:**
1. Depth attention strictly enlarges the readout class at fixed state (separation).
2. It cannot increase the state-tracking class beyond the state update (composition/TC⁰ bound).
3. A block upper-triangular cascade is scan-compatible and stable under a row-wise condition (supporting result).
4. Empirically, the two axes dissociate: attention readout helps MQAR/recall, not `A5`; block-dense helps `A5`, not necessarily recall.
5. Design principle: readout for retrieval, transition for tracking.

### 11.3 What is kept from the original idea

- The multi-slot state and the timescale semantics.
- The block upper-triangular cascade as the state structure.
- The depth-attention readout and its analysis (separation, content-based selection).
- The empirical comparison against linear/dense/MLP readouts, BD-LRU, MS-SSM, Mamba-2, and Gated DeltaNet.

### 11.4 What is dropped

- The claim that the new block is a SOTA architecture.
- The 100M–1B language-modeling phase.
- The claim that attention over state gives state tracking.
- The claim that the cascade is maximally expressive (it remains a conjecture).

### 11.5 Where the finalized details live

- `paper-plan.md` — thesis, contributions, outline, venue, collaboration, risks.
- `theory-notes.md` — the three propositions, proof sketches, assumptions, proof ladder.
- `experiment-plan.md` — the tiny-scale experiment matrix, hardware notes, gates.

### 11.6 Remaining open questions

1. Can Proposition 1 be strengthened to a finite-temperature separation, or is the `τ→0` limit acceptable?
2. Is the cascade strictly more expressive than block-diagonal at fixed block size? (Optional; a conjecture.)
3. Does the dissociation hold at 10–50M parameters, or only at 1–5M?
4. Does a gated linear + softmax readout beat both pure linear and pure softmax?
5. Does depth attention help when the slots are arbitrary heads rather than timescales?
