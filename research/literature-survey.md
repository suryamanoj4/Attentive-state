# Literature Base: State, Memory, and Attention in Sequence Models

**Status:** working survey, last updated 2026-11-23.
**Purpose:** establish the technical base for evaluating a proposed architecture in which the recurrent state is split into multiple layers/slots, updated with a linear cascade, and read out (or updated) with a small attention kernel across those slots.

**Scope.** RNNs and linear RNNs; state-space models (SSMs); neural controlled differential equations (CDEs) and signatures; linear attention and the delta rule; associative recall and memory capacity; multi-scale/hierarchical state; convolution–attention hybrids; attention over depth and over recurrent states; parallelization and kernels; expressivity and state-tracking theory. The survey covers work up to November 2026. Papers from 2025–2026 are frequently preprints; venue labels are best-effort and should be re-checked before formal citation.

**How the documents fit together**

| File | What it is |
|---|---|
| `literature-survey.md` | This document. The map of the field, organized by thread, with a relevance note per thread. |
| `paper-plan.md` | The finalized paper plan: thesis, contributions, outline, venue, collaboration, risks. |
| `theory-notes.md` | Theorem statements and proof sketches for the limits paper. |
| `experiment-plan.md` | A falsifiable experimental programme sized to the available hardware: benchmarks, baselines, ablations, metrics, and go/no-go criteria. |
| `idea-analysis.md` | A precise restatement of the original architecture idea, its formal variants, what is novel, what is already covered, and the main failure modes. |
| `raw/` | Downloaded source texts and metadata used to write the survey. |

---

## 0. Caveats that matter for reading the literature

1. **Asymptotic expressivity ≠ finite-size efficiency.** Theorems of the form "architecture X is maximally expressive" usually mean universal approximation as width/depth → ∞, or "maximal probabilistic expressivity" over a parameter distribution. They do not imply that X is better than Y at 100M parameters and 10B tokens. The practical question — more capability per parameter and per layer at a fixed budget — is empirical and is usually decided by optimization, data, and inductive bias rather than by the universal-approximation class.

2. **"State tracking" and "language modeling" reward different things.** Diagonal SSMs can be competitive language models while failing formal state tracking; delta-rule models can be excellent at retrieval while performing at chance on structure probes; LSTMs can be very expressive while being weaker language models. A single benchmark cannot adjudicate the design.

3. **Parallel training is a property of the update, not of the architecture's name.** A linear (affine) recurrence in the state admits a parallel scan. A nonlinear recurrence generally does not. Attention inside the state update is therefore a different computational object from attention in the readout; the two must be analysed separately.

4. **"Convolution" means several different things.** A time-invariant SSM unrolls to a long convolution. A selective SSM does not (it is time-varying). A short depthwise convolution on inputs (as in Mamba) is a local feature extractor. A convolution across state slots is a fixed linear mixing of the state. These are not interchangeable.

5. **The word "sandwich" is already taken.** MELODI (Chen et al., 2024, arXiv:2410.03156) uses "sandwich architecture" for a short-term multi-layer recurrent compression plus a long-term memory layer. If we use "sandwich state", we should define it explicitly and distinguish it.

---

## 1. Executive summary: twelve findings that bear on the idea

1. **The diagonal-vs-dense expressivity result is real and well established.** Diagonal state transitions (S4D, Mamba's per-channel A) are not maximally expressive; input-dependent dense, block-diagonal, diagonal-plus-low-rank, sparse, and Walsh–Hadamard transitions are. The cleanest references are Cirone et al. (2024, arXiv:2402.19047), Walker et al. (SLiCE, NeurIPS 2025, arXiv:2505.17761), and Terzić et al. (PD-SSM, arXiv:2509.22284). The practical corollary is that **the transition matrix, not attention over the state, is the primary lever for state-tracking expressivity.**

2. **Block-diagonal is the cheapest maximally expressive structure known so far.** BD-SLiCE uses dense `b × b` blocks on the diagonal, with cost `O(d_h Σ b_j²)` and maximal probabilistic expressivity when the largest block grows. It solves the `A5` state-tracking benchmark with **one layer**, while diagonal models need a number of stacked layers that grows with sequence length. This directly supports the "less depth via richer state mixing" intuition — but the mechanism is block-dense transitions, not attention over the state.

3. **"Maximal expressivity" is asymptotic and does not settle the finite-width trade-off.** For a fixed hidden size, smaller blocks are faster but less expressive; larger blocks are more expressive but cost more. BD-SLiCE's own appendix studies this trade-off. BD-LRU (Dubinin et al., arXiv:2602.12021) finds that **moderate block sizes (2–8) scale best** in language modeling; the largest block is not automatically the best.

4. **Attention over the state is not the same as state tracking.** In an affine recurrence, putting a softmax attention *readout* over state slots adds a nonlinear, content-dependent selection at each timestep. It does **not** change the class of state transitions, so it does not by itself give the model the ability to compose permutations outside TC⁰. To change the state-tracking class, the *update* must be non-diagonal and input-dependent (or nonlinear); the readout is not enough. See Merrill et al. (2024, arXiv:2404.08819), especially §5, and the finite-precision argument in Brandoit et al. (2026, arXiv:2609.32447).

5. **A local attention kernel over time does not give global attention in one layer.** A kernel of width `K` has receptive field `K`; stacking `n` layers gives `n(K−1)+1`. To cover a sequence of length `T` you need about `T/K` layers, or one global layer. A kernel over **state slots/layers** is different: all `L` slots exist simultaneously at time `t`, so one depth-wise kernel sees the whole state. This distinction is central and is often blurred in informal descriptions.

6. **The "SSM is already a convolution" statement holds only for time-invariant SSMs.** S4/S4D/DSS unroll to a long convolution and can be trained with FFTs. Selective SSMs (Mamba/S6, Mamba-2, Mamba-3, Gated DeltaNet) are time-varying; they use scans or chunkwise algorithms, not a single global convolution. Mamba also has a *short* depthwise convolution on the input, which is a different object from the recurrence.

7. **The closest prior art to "attention over state layers" is no longer hypothetical.** DART (2026, arXiv:2608.02032) performs attention over **chunk-level recurrent states** retained from a Mamba-2 scan; Depth-Attention (2026, arXiv:2606.05014) and Mixture-of-Depths Attention (2026, arXiv:2603.15619) perform attention over **depth** (layer outputs) in transformers; Attention Residuals (Kimi Team, 2026, arXiv:2603.15031) replaces the residual sum with a softmax over depth. The specific combination "block-diagonal/multi-timescale SSM state + depth attention readout" is not, to our knowledge, published as such, but each component is.

8. **Multi-scale state is a crowded area.** Clockwork RNN (2014), MS-SSM (2025, arXiv:2512.23824), MS-GLA (2026, arXiv:2609.35664), Harmonic (2026, arXiv:2606.24650), and MELODI (2024) all split state or computation across timescales/resolutions. MS-SSM already uses an **input-dependent scale mixer**; its ablation found a linear mixer better than a softmax one in their setting. Harmonic uses three SSM levels with slow/fast decays and gets large long-context gains, but it is an independent preprint with a narrow evaluation.

9. **Depth attention is a real, current research direction in transformers.** Depth-Attention reports +2.3 accuracy points at 1.5B/3B with **no extra parameters** and no extra KV-cache state, by reusing the attention query/key and mixing values along depth. MoDA reports +2.11% average downstream at 1.5B for 3.7% extra FLOPs. This is evidence that depth-wise selection is a useful inductive bias, independent of SSMs.

10. **Attention over recurrent states is also a real direction.** DART shows that Mamba-2's chunk states can be treated as a compressed KV cache and attended over, improving associative recall and retrieval while cutting the inference cache (75% savings at chunk 256 / state 128). Sessa (2026, arXiv:2604.18580) puts attention *inside* a recurrent feedback path and proves power-law memory tails, at the cost of a triangular solve rather than a plain scan.

11. **The binding constraint is often optimization, not raw expressivity.** "Revisiting associative recall in modern recurrent models" (Okpekpe & Orvieto, 2025, arXiv:2508.19029) finds that SSM success on MQAR/copying is confined to a narrow learning-rate window, that SSMs favor width over depth, and that one-layer Mamba can learn recall with well-tuned optimization. BD-LRU needed a new **joint gate normalization** before block-dense state mixing would train at all. Any new architecture must be compared under a fair LR/optimizer sweep, or the result is confounded.

12. **Parameter-efficient depth has alternatives that do not require attention over state.** Fixed-point RNNs (Movahedi et al., NeurIPS 2025, arXiv:2503.10799) implement dense linear RNNs as fixed points of parallelizable diagonal RNNs; a few iterations interpolate from diagonal to dense and reach state-of-the-art on `A5`/`S5` without stacking many layers. Looped/weight-tied models (Huang et al., 2026, arXiv:2610.23036) reuse physical blocks across logical depth and can match deeper feedforward models with far fewer resident parameters. If the goal is "more capability per parameter/layer", these are direct competitors to a new state-attention block.

**Bottom line.** The idea has a real kernel: *make the state multi-slot, keep the recurrence linear/scan-friendly, and use a small content-based attention across slots to select the right timescale or memory*. But the strongest evidence in the literature says (a) richness of the transition matrix, not attention over the state, is what buys state-tracking expressivity; (b) multi-scale state and depth attention are both active, partially solved areas; and (c) the finite-budget benefit must be demonstrated against block-dense recurrences, multi-scale SSMs, depth-attention transformers, and fixed-point/looped models, not only against Mamba.

---

## 2. Foundations I: from RNNs to linear recurrent networks

### 2.1 The classical recurrent stack

- **LSTM** (Hochreiter & Schmidhuber, 1997) introduced gated memory cells and constant error carousels; it is the canonical proof that a recurrent state can carry information over long spans. It is nonlinear in the recurrence, hence sequential at training time.
- **GRU** (Cho et al., 2014, arXiv:1406.1078) simplified the gating (update and reset gates) and became a standard seq2seq building block.
- **Clockwork RNN** (Koutnik et al., 2014, arXiv:1402.3511) partitioned the hidden state into modules updated at exponentially different periods (every step, every 2, every 4, …). This is the direct ancestor of "layers with diminishing update rates". It is not parallel-in-time and predates the modern scan-based SSM.
- **Hierarchical multiscale RNNs** (Chung et al., 2017) and **phased LSTM** (Neil et al., 2016) are further multi-timescale variants. The lesson is old: separating fast and slow state prevents local noise from overwriting long-term memory.

### 2.2 Linear recurrent units and the modern revival

- **LRU** (Orvieto et al., ICML 2023, arXiv:2303.06349) showed that a carefully parameterized *linear* diagonal recurrence, with complex eigenvalues, stable exponential parameterization, a ring initialization, and forward normalization, matches S4/S4D/S5 on Long Range Arena. The key ablations: removing the nonlinearity from the recurrence helps; diagonalizing the recurrence preserves expressivity while enabling scans; eigenvalue magnitude controls memory; normalization is essential. This is the cleanest statement of why linear recurrences are a good design point.
- **minGRU / minLSTM** (Feng et al., 2024, arXiv:2410.01201) remove the hidden-to-hidden nonlinearity and show that gated linear recurrences can be trained in parallel and remain competitive. They are a useful minimal baseline.
- **HGRN / HGRN2** (Qin et al., 2023/2024, arXiv:2311.04823, arXiv:2404.07904) add hierarchical gating and state expansion to linear RNNs; HGRN2 is a strong, simple gated linear baseline.
- **xLSTM** (Beck et al., 2024, arXiv:2405.04517) deliberately keeps a nonlinear recurrent path: sLSTM (scalar memory with exponential gating) and mLSTM (matrix memory with covariance updates), stacked in residual blocks. It is the main modern argument that *some* nonlinearity in the recurrence is worth the loss of scan parallelism. It is also a strong baseline on state tracking.

**Relevance.** The user's "layers with diminishing update rates" is a Clockwork/LRU-style idea. The modern way to instantiate it is as a block-diagonal or block-triangular linear recurrence with per-slot decay rates, trained by scan. The novelty cannot be the multi-timescale decay itself; it must be the coupling/readout.

### 2.3 Gated linear attention and the delta rule

The linear-attention family is the second half of the modern recurrent story.

- **Linear attention** (Katharopoulos et al., ICML 2020, arXiv:2006.16236) replaces softmax with a feature map and uses the associativity of matrix multiplication to compute `(QKᵀ)V` as `Q(KᵀV)`, giving linear time. Its recurrent form is a matrix state `S_t = S_{t−1} + v_t k_tᵀ`, read by `S_t q_t`.
- **Fast-weight programmers** (Schmidhuber, 1992; Schlag et al., ICML 2021, arXiv:2102.11174) interpret the matrix state as fast weights updated by outer products; **DeltaNet** (Schlag et al., 2021, arXiv:2106.06295) replaces the additive write with the delta rule: `S_t = S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`, i.e. an error-correcting write that first removes the current prediction. This is online gradient descent on a reconstruction loss.
- **Gated DeltaNet** (Yang et al., ICLR 2025, arXiv:2412.06464) adds a scalar decay gate, combining forgetting with the delta rule. It is the current strong baseline for recall-intensive language modeling.
- **Gated DeltaNet-2** (Hatamizadeh et al., 2026, arXiv:2605.22791) decouples the erase and write directions, giving finer control over what is overwritten.
- **DeltaProduct** (Siems et al., 2025, arXiv:2502.10297) replaces the single delta update with a product of Householder reflections, which gives the state richer orthogonal/negative-eigenvalue dynamics and substantially improves state tracking (`A5`, `S5`, regular languages) at a modest language-modeling cost.
- **RWKV-7 "Goose"** (Peng et al., 2025, arXiv:2503.14456) uses a vector-gated delta-rule-like state evolution (`wkv` bonus) and reports strong state tracking and long-context behavior.
- **Kimi Delta Attention / Kimi Linear** (Kimi Team, 2025, arXiv:2510.26692) refines the delta rule with channel-wise decay and a hardware-efficient chunkwise WY/UT-transform algorithm; the Kimi Linear model is a 3:1 KDA-to-MLA hybrid with strong long-context results.
- **TTT layers** (Sun et al., 2024, arXiv:2407.04620) make the state the weights of a small model trained by self-supervision at each step; **Titans** (Behrouz et al., 2024/2025, arXiv:2501.00663) add a long-term memory module with surprise-based updates; **MesaNet** (2025, arXiv:2506.05233) solves a local test-time regression problem. These are "state as an online learner" variants.
- **RunningTensor** (Herranz-Celotti et al., 2026, arXiv:2609.12814) lifts the matrix state to an order-`o` tensor, updated by a rank-1 outer product and read by contracting `o−1` queries. Order 2 recovers linear attention; order 3 improves MQAR. This is the "state as a higher-order tensor" axis, distinct from the "state as layers" axis.

**Relevance.** The delta rule is already an attention-like, content-dependent write into a matrix state. If the goal is "attention *in* the state", DeltaNet is the strongest existing answer: it is a fast-weight memory with error-correcting writes, a chunkwise parallel algorithm, and a well-understood capacity story. A new architecture that adds attention across state slots must explain what it adds beyond (a) a delta-rule write and (b) a content-based readout. The most plausible answer is **structure across timescales/slots**, not another write rule.

### 2.4 A compact comparison

| Model family | State shape | Update | Readout | Parallel training | Main strength | Main weakness |
|---|---|---|---|---|---|---|
| LSTM/GRU | vector | nonlinear gated | linear | no (sequential) | state tracking, small data | slow, hard to scale |
| S4/S4D/DSS | vector (diagonal) | linear, time-invariant | linear | FFT / scan | long-range, efficient | not maximally expressive; fading memory |
| Mamba/S6 | vector (diagonal, input-dependent Δ,B,C) | selective linear | linear | scan | strong LM baseline | diagonal limits state tracking |
| Mamba-2/SSD | matrix (scalar decay per head) | selective linear | linear | chunked scan / attention dual | GPU-efficient, strong LM | scalar decay, limited cross-channel mixing |
| Mamba-3 | complex / matrix, MIMO | selective, complex rotation | linear | chunked scan | state tracking + LM Pareto | new, complex kernels |
| DeltaNet/Gated DeltaNet | matrix (fast weights) | delta rule (+ decay) | linear | chunkwise WY | associative recall, editing | capacity/rank limits; optimization brittleness |
| DeltaProduct | matrix (Householder product) | product of reflections | linear | chunkwise | state tracking | LM perplexity cost at fixed budget |
| BD-LRU / BD-SLiCE | block-diagonal, dense blocks | linear, input-dependent | linear | block scan | expressivity per parameter; state tracking | stability/normalization; modest LM perplexity gain |
| Transformer | — (KV cache) | — | softmax attention | fully parallel | recall, in-context learning | quadratic; growing cache |
| DART | Mamba-2 state + chunk memories | Mamba-2 + attention over chunks | linear + SMA | chunked | recall without full KV cache | still grows with number of chunks |
| MS-SSM / MS-GLA / Harmonic | multi-resolution / multi-level | per-scale SSM/GLA | linear scale mixer | parallel per scale | multi-scale structure | not content-addressable across scales |
## 3. Foundations II: continuous-time models, CDEs, and signatures

The CDE line is the most mathematically mature version of "state as a controlled dynamical system". It is directly relevant because SLiCE (the paper behind BD-SLiCE) is a CDE framework, and because the EWS paper maps Mamba and Mamba-2 into a CDE form.

### 3.1 Neural ODEs and Neural CDEs

- **Neural ODE** (Chen et al., NeurIPS 2018, arXiv:1806.07366) defines the hidden state by `dh/dt = f_θ(h, t)` and backpropagates through an ODE solver (or the adjoint). It is a continuous-depth model; the state is a deterministic function of the initial condition and time.
- **Neural CDE** (Kidger et al., NeurIPS 2020, arXiv:2005.08926) makes the vector field depend on the *derivative of a control path*: `dz_t = f_θ(z_t) dX_t`, where `X` is an interpolation of the observed time series. This is the continuous-time analogue of an RNN and handles irregular sampling, missing values, and variable-length sequences naturally. The hidden state is driven by the path, not just by time.
- **Log-ODE / Neural RDE** (Morrill et al., 2020, arXiv:2009.08295) summarizes local path increments by their log-signature and integrates an ODE over larger steps. This reduces the effective sequence length and gives large speed-ups on long series (up to 17k observations in the original paper). **Log-NCDE** (Walker et al., ICML 2024) is a stronger variant that outperforms NCDEs, NRDEs, LRU, S5, and Mamba on several multivariate time-series datasets.
- **Signatory** (Kidger & Lyons, ICLR 2021, arXiv:2001.00706) provides differentiable, GPU-capable signature/log-signature transforms; it is the practical substrate for signature-based models.

**Why it matters.** A CDE is a linear recurrence in the state if `f_θ` is linear in the state; the path dependence can be input-dependent without breaking linearity in the state. This is exactly the structure SLiCE exploits. The CDE view also gives a principled way to think about "layers" as iterated integrals / truncation depth: higher truncation depth is nonlinear in the path *within a single layer*, which is an alternative to stacking.

### 3.2 SLiCE: structured linear CDEs

**SLiCE** (Walker, Yang, Cirone, Salvi, Lyons; NeurIPS 2025, arXiv:2505.17761) is the key unification.

- It reframes SSMs as **linear neural CDEs (LNCDEs)**: `dh_t = Σ_i A^i_θ h_t dω^i_t`, with input-dependent transition matrices `A^i_θ`.
- It proves that **diagonal** `A` is not maximally expressive, while **dense**, **block-diagonal**, **sparse**, **Walsh–Hadamard**, and **diagonal-plus-low-rank (DPLR)** structures are (under stated conditions).
- The block-diagonal variant, **BD-SLiCE**, has parameters `O(d_h Σ_j b_j²)` and recurrent/scan cost `O(n d_h Σ_j b_j²)`, and is the only maximally expressive variant that strictly reduces both parameter count and cost relative to dense.
- Empirically, BD-SLiCE solves `A5` with **one layer**, generalizes to 8× the training length on regular languages, and matches Log-NCDE on six UEA time-series datasets with ~20× lower training time.
- The paper also introduces a **diagonal–dense** variant (D-DE-SLiCE): a large diagonal part for cheap chunkwise processing plus a small dense block for expressivity. This is a practical template for "most of the state is cheap; a small part is expressive".

**What the user should take from it.** The user's description ("BD-SLICE achieves max expressiveness by a small batch of the dense matrix at a time") is slightly off: BD-SLiCE is a **block-diagonal** transition, with each block dense; it is not a batch of dense matrices applied to the whole state. The important facts are (a) the blocks are dense and input-dependent, (b) the structure is closed under multiplication, so scan works, and (c) maximal expressivity is asymptotic in the largest block size. At fixed width, block size is a compute–expressivity knob.

### 3.3 EWS: SSMs as depth-one CDEs

**Exponentially Weighted Signature (EWS)** (Bloch, Walker, Mouterde, Morley, Cohen, Lyons; 2026, arXiv:2603.19198) computes iterated integrals of a path with each increment weighted by the matrix exponential of a learnable generator over elapsed time. It proves a modified Chen identity, enabling a parallel scan, and shows that **at depth one the EWS is an SSM**: linear time-invariant SSMs, Mamba channels, and Mamba-2 heads can be mapped to it in closed form. Higher truncation depths are nonlinear in the path within one layer. Learned clocks are necessary for state tracking; at depth one it matches or exceeds competing SSMs with far fewer parameters on regression/forecasting.

**Why it matters.** This is a concrete existence proof that "one layer with richer internal state" can substitute for stacking, and that the CDE/signature language gives a principled way to design that internal state. It is closer to the user's "more expressivity per layer" goal than most SSM work.

---

## 4. Foundations III: attention, local kernels, and convolution hybrids

### 4.1 The attention baseline and its kernel view

- **Transformer** (Vaswani et al., NeurIPS 2017, arXiv:1706.03762): softmax attention over all previous tokens; quadratic in sequence length; growing KV cache at inference.
- **Linear attention** (Katharopoulos et al., 2020, arXiv:2006.16236): replace `softmax(qᵀk)` with `φ(q)ᵀφ(k)`, use associativity to get `O(T)` compute and a matrix state. This is the bridge between attention and recurrent models.
- **Performer** (Choromanski et al., 2020, arXiv:2009.14794) and **Linformer** (Wang et al., 2020, arXiv:2006.04768) are the other canonical linear/approximate attention families.
- **FlashAttention** (Dao et al., 2022, arXiv:2205.14135) is the IO-aware exact attention kernel that made long-context attention practical; it is an implementation layer, not an architectural change.
- **The kernel view**: attention is a Nadaraya–Watson-style kernel smoother; the softmax weights are input-dependent. A single-head attention layer is rank-limited in its interaction operator (see §5.4), which is why multiple heads matter.

### 4.2 Local attention, dynamic convolution, and involution

This is the literature most directly relevant to "attention inside a small kernel".

- **Stand-Alone Self-Attention (SASA)** (Ramachandran et al., NeurIPS 2019, arXiv:1906.05909) replaces convolution with local self-attention over a small spatial window. It is the clean statement that a local attention kernel can stand alone as a primitive.
- **Involution** (Li et al., ICCV 2021, arXiv:2103.06255) inverts the inherence of convolution: the kernel is *location-specific* and generated from the input, rather than shared across locations. This is a dynamic, content-adaptive kernel with `O(HW)` rather than `O(HWC·C')` parameters.
- **Dynamic Convolution** (Chen et al., CVPR 2020, arXiv:1912.03458) computes attention weights over a small set of convolution kernels and mixes them per input. It is "attention over kernels", not "attention inside a kernel", but it is the same design instinct.
- **"On the Connection between Local Attention and Dynamic Depth-wise Convolution"** (Han et al., ICLR 2022, arXiv:2106.04263) proves a formal connection: local attention with a small window is closely related to dynamic depthwise convolution. This is the single most useful reference for the claim "attention in a sliding kernel is a dynamic convolution".
- **Mega** (Ma et al., 2022, arXiv:2209.10655) combines an exponential moving average (a linear recurrence) with a gated attention block in the same layer. It is an early, strong hybrid of "state" and "attention" and predates the current SSM wave.
- **Hyena** (Poli et al., NeurIPS 2023, arXiv:2302.10866) stacks long implicit convolutions with gating and achieves sub-quadratic long-context language modeling. It is the canonical "long convolution instead of attention" architecture.
- **Longformer** (Beltagy et al., 2020, arXiv:2004.05150) and **BigBird** (Zaheer et al., 2020, arXiv:2007.14062) are the sparse/local+global attention baselines. They establish the receptive-field arithmetic: local windows plus a few global tokens.

**Relevance.** "Attention inside a convolution kernel" is not new as a primitive. The open questions are (a) whether it is applied along **time** or along **state depth**, (b) whether the kernel is content-adaptive in both the query and the key, and (c) whether it is cheap enough to replace a standard short convolution. The literature already contains the primitive; the novelty must come from where and how it is used.

### 4.3 What a local kernel can and cannot do

A single local attention layer with window `K` has receptive field `K`. Stacking `n` layers gives `n(K−1)+1`. Therefore:

- One local kernel does **not** give "complete attention over the whole sequence"; it gives local mixing. Global information still has to travel through the recurrence or through many layers.
- One local kernel over `L` **state slots** does see the whole state, because all slots are present at the same timestep. This is the correct sense in which a depth-wise kernel can give "complete attention over the state".
- If the state slots are updated in a cascade within a timestep, the depth kernel may need a causal mask over slots (shallow → deep) to avoid using a slot that has not been updated yet. If the slots are updated in parallel, no such mask is needed, but the order/timescale semantics must be injected some other way (e.g., per-slot positional embeddings or learned timescale embeddings).

---

## 5. Expressivity and state tracking: what the theory actually says

### 5.1 The TC⁰ boundary

- **Merrill & Sabharwal, "The Parallelism Tradeoff"** (2022, arXiv:2207.00729) show that log-precision transformers are in uniform TC⁰; this is the baseline complexity class.
- **"The Illusion of State in State-Space Models"** (Merrill, Petty, Sabharwal; ICML 2024, arXiv:2404.08819) shows that S4, Mamba/S6, and related SSMs are also in TC⁰, and therefore cannot solve problems like permutation composition (`S5`) or track chess state under certain notation, despite their recurrent form. The paper's §5 gives the two escapes:
  1. **Nonlinearity inside the recurrence** (an "RNN-SSM"): expressive, but breaks the scan.
  2. **Fully input-dependent transition matrices** (an "IDS4"): expressive and still parallelizable, because iterated products of general matrices are not in TC⁰.
- **"The Computational Limits of State-Space Models and Mamba via the Lens of Circuit Complexity"** (Chen et al., 2024, arXiv:2412.06148) independently proves that Mamba/SSMs with poly-precision and constant depth are in DLOGTIME-uniform TC⁰.
- **"The Expressive Capacity of State Space Models: A Formal Language Perspective"** (Sarrof, Veitsman, Hahn; 2024, arXiv:2405.17394) gives a finer picture: SSMs and transformers have overlapping but distinct strengths; SSMs can implement length-generalizing solutions to some star-free state-tracking problems and can model bounded hierarchical structure without a stack, but a design choice in current SSMs limits them. It also identifies PARITY as hard for the standard SSM form.

**Implication.** A softmax readout over state slots does not move the model out of TC⁰ if the state update remains affine. To get state tracking, the *transition* must be input-dependent and non-diagonal (block-dense is enough), or the recurrence must be nonlinear. This is why BD-SLiCE and PD-SSM solve `A5` with one layer while Mamba needs depth that grows with sequence length.

### 5.2 The diagonal-to-dense hierarchy

- **Cirone et al., "Theoretical Foundations of Deep Selective State-Space Models"** (2024, arXiv:2402.19047) is the theory paper behind SLiCE. It shows that diagonal transition matrices are fundamentally less expressive than dense ones, and frames SSMs as linear CDEs. It is the source of the "maximal probabilistic expressivity" notion.
- **SLiCE** (2025) extends this to block-diagonal, sparse, Walsh–Hadamard, and DPLR structures, and proves maximal expressivity for them.
- **PD-SSM** (Terzić et al., 2025, arXiv:2509.22284) parametrizes `A = P D` with `P` a column one-hot matrix and `D` a complex diagonal. The scan cost is linear in state size, the model is BIBO-stable, and it can emulate any `N`-state finite-state automaton with **one layer of dimension N** and a linear readout. It outperforms a wide set of SSMs on FSA emulation, and beats NCDEs on multivariate time-series classification.
- **Fixed-point RNNs** (Movahedi et al., NeurIPS 2025, arXiv:2503.10799) take a different route: they represent a dense linear RNN as the fixed point of a parallelizable diagonal RNN. A few fixed-point iterations interpolate from diagonal to dense; `A5`/`S5` results are state-of-the-art, and the number of iterations adapts to task difficulty. This is a direct competitor to "more layers" and to "more state structure".
- **BD-LRU / H-LRU** (Dubinin, Orvieto, Effenberger; 2026, arXiv:2602.12021) make the block-dense idea practical for language modeling. BD-LRU uses dense `m × m` blocks and mixes channels at the same timestep; H-LRU uses a companion matrix and mixes each channel with its own past. They introduce **joint selective gate normalization** (row-wise L1 over state and input gates) to keep the forward pass bounded for arbitrary block/window size. Empirically: moderate block sizes (2–8) scale best; BD-LRU is best on permutation composition; H-LRU is more parameter-efficient on compression; language-model perplexity gains are modest because perplexity mostly rewards retrieval, not state tracking.

**Implication.** There is a clear "structured transition" ladder: diagonal → diagonal-plus-low-rank → block-diagonal → sparse/Walsh–Hadamard → dense. The user's "layers" can be mapped onto block-diagonal blocks; a linear cascade between layers makes the transition **block upper-triangular**, which is a different (and less studied) point on the ladder. That is a potentially interesting contribution: block upper-triangular input-dependent transitions are closed under multiplication (so scan works), and they encode a timescale hierarchy with leakage, but they are not covered by the BD-SLiCE theorem as stated.

### 5.3 Finite precision and length-independent tracking

- **"Length-Independent State Tracking Under a Parallel Scan"** (Brandoit et al., 2026, arXiv:2609.32447) formalizes tracking at finite precision. It shows that length-independent tracking requires two competing dynamics in one map: **contraction** to suppress numerical perturbations and **separation** to keep states distinct. Affine recurrences, with a single rate at each step, can realize at most definite automata. The paper introduces a **Neural Finite-State Machine (NFSM)**, a nonaffine but scan-compatible RNN, which learns exact transition tables and stays correct at all tested lengths.
- **"Rethinking State Tracking in Recurrent Models Through Error Control Dynamics"** (2026, arXiv:2605.07755) argues that exact affine tracking cannot correct state error, and studies state-dependent error correction and accumulated error over rollouts. It compares S4, Mamba, negative Mamba, Mamba-3, AUSSM, DeltaNet, DeltaProduct, Gated DeltaNet, KDA, Gated DeltaNet-2, LSTM, GRU, and more.

**Implication.** A purely linear/affine state update has a finite-precision ceiling that is independent of how clever the readout is. If the target is robust long-horizon state tracking, the update needs a contraction/separation mechanism (nonlinear or state-dependent), and a scan-compatible design is nontrivial. This is a serious constraint on the "linear cascade + attention readout" plan.

### 5.4 Depth vs. width vs. precision

- **"On the Expressive Power and Limitations of Multi-Layer SSMs"** (Zubić et al., 2026, arXiv:2604.14501) proves a depth hierarchy for `K`-function composition: an `L`-layer SSM solving `(L+3)`-function composition must satisfy `d²p = Ω(N/L³)` (state dimension `d`, precision `p`, input length `N`). Conversely, `K`-function composition is solved by a `(K+1)`-layer generalized SSM with `d=1` and `p=Θ(log N)`. Width and precision are not interchangeable under exact step-preserving simulation in the base affine model, but become interchangeable with input-interleaved chain-of-thought.
- **"How Many Heads Make an SSM?"** (Ghodsi, 2025, arXiv:2512.15115) defines an **interaction rank** `k = dim span{W(τ)}` for a linear SSM's lag operators. A multi-head factorized (attention-style) model needs `H ≥ k` heads to represent the SSM exactly, and `H = k` suffices. It also proves a **gradient highway** result: attention has distance-independent gradient paths, while stable linear dynamics attenuate gradients with distance.

**Implication.** "More capability per layer" is not a single quantity. There are at least three resources: state dimension/rank, precision, and depth. A depth-attention readout can increase the effective interaction rank at a given state size (by mixing slots with input-dependent weights), but it does not substitute for precision or for a non-affine update. The interaction-rank framing is a good way to state the hypothesis precisely: *does depth attention increase the effective interaction rank per parameter?*

### 5.5 What this means for the user's "less layers" claim

The literature supports a narrower claim than "attention over state reduces the number of layers needed":

- **For state tracking / formal languages**, the number of layers needed is set mainly by the transition structure. Block-dense or PD-style transitions solve `A5` in one layer; diagonal models need depth growing with length. Attention over the state is not the lever.
- **For language modeling / recall**, depth helps but is often less efficient than width or state size; SSMs can favor width over depth due to optimization brittleness. A depth-attention readout may improve recall at a given depth, but it is competing with simply increasing state size or adding a delta-rule memory.
- **For parameter efficiency at fixed depth**, fixed-point/looped models and block-dense transitions are the strongest existing answers. A new block must beat them at matched parameters, tokens, and compute.

---

## 6. Recall, memory capacity, and associative retrieval

### 6.1 The MQAR benchmark and the recall gap

- **Zoology** (Arora et al., 2023, arXiv:2312.04927) identifies the **multi-query associative recall (MQAR)** task and shows a perplexity/recall gap between attention and gated-convolution/SSM models at matched scale. It also gives a theoretical account of gated convolution capacity.
- **Based** (Arora et al., ICML 2024, arXiv:2402.18668) makes the memory–recall trade-off explicit: recurrent state is `O(1)` memory but loses recall; attention has perfect recall but `O(T)` memory. Based uses a **second-order Taylor feature map** for linear attention plus a small sliding-window exact attention, and is a strong recall-oriented baseline.
- **Hedgehog** (Zhang et al., ICML 2024, arXiv:2402.04347) learns a feature map that mimics softmax attention, improving linear-attention recall without quadratic cost.
- **"Repeat After Me"** (Jelassi et al., 2024, arXiv:2402.01032) proves that transformers can copy inputs of exponential length (with hashing/positional tricks) while state-space models cannot copy beyond their memory size. This is a clean capacity separation, not just an empirical gap.
- **"Understanding Input Selectivity in Mamba"** (Huang et al., ICML 2025, arXiv:2506.11891) proves that Mamba's S6 layer can represent Haar wavelet projections (an edge over S4D for discontinuous functions), shows how selectivity counteracts memory decay, and gives analytical MQAR solutions for Mamba, Mamba-2, and S4D.
- **"Revisiting associative recall in modern recurrent models"** (Okpekpe & Orvieto, 2025, arXiv:2508.19029) is the optimization counterpoint: SSMs' MQAR/copying success is confined to a narrow LR window; outside it, accuracy collapses. SSMs favor width over depth; one-layer Mamba can learn MQAR with a well-tuned LR, but the dynamics do not look like induction heads. The differentiator may be learnability, not expressivity.
- **DART** (2026, arXiv:2608.02032) shows that attention over Mamba-2 chunk states substantially improves associative recall and retrieval at a fraction of the token-level KV cache.
- **Fan Duality** (2026, arXiv:2604.07716) and **Echo** (2026, arXiv:2605.06997) are 2026 attempts at `O(1)` decode memory with stronger associative recall, using duality/Koopman operators. They are worth tracking as competitors to any "state attention" design.

### 6.2 Capacity and rank

- **RunningTensor** (2026, arXiv:2609.12814): vector state is `O(W)`, matrix state is `O(W²)`, order-`o` tensor state is `O(W^o)`; order 3 improves MQAR over linear attention and SSMs. This is the "increase the order of the state" axis.
- **"How Many Heads Make an SSM?"** (2025, arXiv:2512.15115): the interaction rank of a linear SSM's lag operators is the number of independent impulse-response matrices; a multi-head attention model needs that many heads to match it. This gives a principled reason why a single monolithic state/readout is rank-limited, and why multiple slots/heads can help.
- **MIMOMamba** (2026): generalizes SSD to MIMO with matrix-polynomial parameterization; cross-channel mixing inside the state, and a matrix-valued attention dual. It reports better RMSE than Mamba-2/Mamba-3/PD-SSM on a rotational-dynamics benchmark with ~1/3 the Transformer's parameters. This is the "state as a matrix with cross-channel dynamics" axis.
- **"Expanding Attention"** (2026, arXiv:2610.20637): a truncated Taylor expansion of the exponential kernel collapses the causal history into exactly `n` recurrent states, each updated by a rank-one outer product; the optimum is second order (`n=3`) at a given error, and the states are additive across chunks (parallel scan). This is a concrete "multiple recurrent states" design with a clear capacity–cost trade-off.

### 6.3 Hybrids and the current frontier

- **Griffin / Hawk** (De et al., 2024, arXiv:2402.19427) mix gated linear recurrences with local attention; Griffin matches or beats Mamba/Transformer at matched scale, and Hawk is pure recurrent.
- **RecurrentGemma** (Botev et al., 2024, arXiv:2404.07839) is a Griffin-based open model with local attention + recurrent blocks; it demonstrates the hybrid at 2B/7B scale.
- **Jamba** (Lieber et al., 2024), **Samba** (Ren et al., 2024), **Zamba** (Glorioso et al., 2024), **Hymba** (Dong et al., 2024), **MiniMax-01** (2025), **Nemotron-H**, **Falcon-H1**, **Qwen3-Next**, and **Kimi Linear** are the 2024–2026 production hybrids. The consistent pattern: a small number of full-attention layers plus many recurrent/SSM layers, with attention handling exact recall and the recurrent state handling cheap long-context compression.
- **Priming** (2026, arXiv:2605.08301) is a controlled comparison of SSM layer types at scale under identical conditions (GKA > GDN > Mamba-2 in the reported hierarchy), and shows that a compressed fading memory is more useful than a larger sliding window of verbatim tokens when complementing attention. It also demonstrates that hybrid design can be turned from a pretraining problem into a knowledge-transfer problem.
- **"What Attention Recalls and Recurrence Controls in Hybrid Language Models"** (2026, arXiv:2609.04434) uses split-prefill and state-swap interventions to show that the KV side supplies the value and the recurrent side supplies the language/format. This is a useful causal decomposition for thinking about what a state-attention block should contribute.

**Relevance.** The field has largely converged on hybrids for production, not pure SSMs. A new block that claims to replace depth or attention must show where it sits in the hybrid Pareto frontier. The strongest framing may be "a better recurrent block for hybrids" rather than "a replacement for attention".
## 7. Multi-scale and hierarchical state

### 7.1 The classical roots

- **Clockwork RNN** (2014, arXiv:1402.3511): hidden units partitioned into modules with periods `1, 2, 4, ...`; slow modules are updated less often. This is the original "diminishing update rate" state.
- **Hierarchical multiscale RNN** (Chung et al., 2017) and **phased LSTM** (Neil et al., 2016) are close relatives. The recurring lesson: separating timescales helps long-range memory, but hard periodicity is brittle and non-differentiable.

### 7.2 MS-SSM: multi-resolution SSM

**MS-SSM** (Karami, Behrouz, Zhong, Pascanu, Mirrokni; 2025, arXiv:2512.23824) decomposes the input with a stationary wavelet transform (causal dilated depthwise convolutions at scales `2^s`) into `S+1` resolution streams, runs an independent SSM per stream (plus the original), and merges them with an **input-dependent scale mixer**:

```
z_t = E_t y_t,   E_t = Linear_E(x_t),   y_t ∈ R^{S+2}
```

Key details:
- The effective state size is `(S+2)N`, and they match it to the baseline state size for fair comparison.
- Lower-resolution (coarser) SSMs are initialized with eigenvalues closer to 1 (longer memory); higher-resolution ones with faster decay.
- The scale mixer is a **linear** input-dependent projection. Their appendix ablation compares a data-independent linear mixer, a data-dependent linear mixer, and a data-dependent **softmax** mixer; the data-dependent linear mixer won in their setting.
- They report gains on LRA, hierarchical reasoning, time-series classification, and image recognition.

**Relevance.** MS-SSM is the closest published architecture to "multiple state layers with different timescales + a learned cross-scale mixer". The user's proposal differs in (a) using a **cascade** (slow layers receive from fast layers) rather than independent per-scale SSMs, and (b) using a **softmax attention** across slots rather than a linear mixer. Both differences are testable, and MS-SSM's ablation suggests the softmax may not be automatically better.

### 7.3 MS-GLA: multi-scale gated linear attention

**MS-GLA** (Dev, Sankar, Varma; 2026, arXiv:2609.35664) distributes **groups of GLA heads** across temporal resolutions by pooling tokens for coarser groups, then uses a learnable input-dependent fusion layer to recombine head-group outputs. It reports up to 18.9% improvement on recall-intensive tasks and 9.5% lower average perplexity than GLA at matched parameter counts. It explicitly adapts MS-SSM's multi-resolution idea to gated linear attention.

**Relevance.** This is the "multi-scale state" idea implemented in the linear-attention family, with a fusion layer. It is another direct prior-art point: multi-scale heads + input-dependent fusion is already published (as a preprint) and reports strong recall gains.

### 7.4 Harmonic: hierarchical SSM with prediction-error signals

**Harmonic** (Nyoma, 2026, arXiv:2606.24650) stacks three SSM levels with progressively slower timescales:

```
h_ℓ(t) = A_ℓ(t) ⊙ h_ℓ(t−1) + b_ℓ(t),   ℓ = 1,2,3
```

- `A_ℓ(t)` is a data-dependent decay constrained by initialization to a level-specific timescale range `[τ_min, τ_max]`, with `τ_1 ≪ τ_2 ≪ τ_3`.
- Each level predicts the level below via a linear map `P_ℓ`; the **prediction error** `e_ℓ = h_{ℓ−1} − P_ℓ(h_ℓ)` is normalized and passed to level `ℓ+1` instead of the raw state.
- Outputs are summed and passed through one linear head; training is end-to-end with next-token prediction.
- Results: 28M-parameter model, enwiki8, equal token budgets; +1.4% at 1K, +6.7% at 8K, +11.4% at 32K vs. a Transformer; 0.7–1.8% better than Mamba; trains at 64K where both baselines OOM on an 80GB H100. Replicates on WikiText-103. At 1B scale, replacing TinyLlama's attention layers with HarmonicBlock removes the RoPE length limit.
- Ablation: passing raw states instead of prediction errors is indistinguishable; **the timescale hierarchy is the load-bearing component**. The prediction-error story is attractive but not empirically supported in their setup.

**Caveats.** Independent researcher, narrow evaluation (enwiki8/WikiText-103, 28M–1B, mostly equal-token comparisons), no peer review as of this writing. Treat the magnitude of the gains as provisional. The architectural lesson — multi-timescale decay + sum/merge — is consistent with MS-SSM and Clockwork.

**Relevance.** Harmonic is essentially the "sandwich state" without cross-layer attention. It is the strongest evidence that a small number of timescale levels can produce large long-context gains. It also suggests that the *hierarchy* matters more than the specific inter-level signal. A depth-attention readout would be a natural addition: instead of summing the levels, let the token attend over them.

### 7.5 MELODI and the "sandwich" naming

**MELODI** (Chen, Hutchins, Jansen, Zhmoginov, Racz, Andersen; Google DeepMind, 2024, arXiv:2410.03156) uses a **sandwich architecture** for long-context memory:
- Short-term memory: recurrent compression of context windows across multiple transformer layers (e.g., 512 tokens → 128 memory tokens), giving smooth window-to-window transitions.
- Long-term memory: further compression within a single middle layer, aggregated across windows.
- The long-term compression layer is inserted between short-term recurrent layers; the whole thing operates on short context windows.
- Compared to a Memorizing Transformer with 64K KV pairs, it is better on long-context datasets with an 8× smaller memory footprint.

**Relevance.** The word "sandwich" is already used for a layered short-term/long-term memory. MELODI is not a multi-timescale SSM, but it is prior art for "layers with different memory roles + compression". If we use "sandwich state", we should cite MELODI and define our meaning.

### 7.6 Higher-order and block-diagonal state mixing

**BD-LRU / H-LRU** (2026, arXiv:2602.12021) is the most directly relevant paper for "state mixing" as a design axis:
- **BD-LRU**: partition the state into `N` blocks of size `m`; each block has a dense `m × m` input-dependent transition; dense channel mixing within a block, no cross-block mixing.
- **H-LRU**: each coordinate has an `m`-th order recurrence, implemented as a companion matrix; temporal mixing of past values, with only `m+1` gates per block.
- **Joint selective gate normalization**: row-wise L1 normalization over `[A_t, a_0,t]` guarantees a bounded forward pass for any block/window size. Without it, block-dense models do not train well.
- Results: moderate block sizes (2–8) scale best; BD-LRU is best on permutation composition and the only model that solves it with scarce data; H-LRU is more parameter-efficient on compression; language-model perplexity gains over diagonal are modest, because perplexity mostly rewards retrieval, not state tracking. DeltaNet/Mamba-2 have lower perplexity than BD-LRU/H-LRU at the same scale, but are at or below chance on structure probes (anagram, order-k Markov, linear vs. random). BD-LRU/H-LRU are above chance on all three structure probes at essentially the same perplexity as Mamba-2.
- The paper explicitly frames the trade-off: **channel mixing for state tracking, temporal mixing for compression**.

**Relevance.** This is the strongest empirical support for the user's "less depth via richer state" intuition — but the mechanism is block-dense mixing, not attention. It also gives the right evaluation philosophy: report both perplexity/retrieval and structure probes, because they reward different things.

---

## 8. Attention over depth, over layers, and over recurrent states

This is the most directly relevant section for the proposed architecture. The literature splits into three families.

### 8.1 Attention over depth in transformers

- **Attention Residuals** (Kimi Team, 2026, arXiv:2603.15031): replaces the residual sum with a learned softmax over depth, so each layer can selectively read earlier layer outputs. It is a depth-attention scheme operating on hidden states.
- **Mixture-of-Depths Attention (MoDA)** (Zhu et al., ByteDance Seed, 2026, arXiv:2603.15619): each attention head attends to sequence KV pairs at the current layer **and** depth KV pairs from preceding layers. It uses a hardware-efficient non-contiguous memory access algorithm and reports 97.3% of FlashAttention-2 efficiency at 64K. At 1.5B parameters: +0.2 average perplexity across 10 validation benchmarks, +2.11% average on 10 downstream tasks, 3.7% FLOPs overhead. Post-norm works better than pre-norm with MoDA.
- **Depth-Attention** (Zeng et al., 2026, arXiv:2606.05014): before a layer attends over the sequence, its query attends **along depth** over the keys of earlier layers at the same position and mixes their values into the value that self-attention then reads. It reuses the standard Q/K/V cache slots, adds **no parameters**, and introduces no persistent inference state beyond the standard KV cache. On Qwen3-style 1.5B/3B models it improves over vanilla by up to 2.3 accuracy points and beats mHC, Attention Residuals, and DenseFormer. It also works in looped transformers. A sparse strided subset of earlier layers is used in practice.
- **Cross-Layer Attention (CLA)** and **YOCO** are KV-cache-sharing schemes (not softmax over depth) that are adjacent prior art for "layers sharing state"; they are useful for the efficiency argument but not for the expressivity argument.

**Relevance.** Depth attention is a proven, cheap inductive bias in transformers: it improves quality with negligible parameter/FLOP overhead by letting a layer select among earlier layers. The user's proposal is the SSM analogue: let a token select among **state slots** rather than layer outputs. The difference is that the state slots are recurrent memories with different timescales, not hidden activations. That difference is the potential contribution.

### 8.2 Attention over recurrent states

- **DART** (Qian et al., 2026, arXiv:2608.02032): keeps the **chunk state contributions** `ΔH_[c]` produced by the Mamba-2 chunked scan as a memory stack; decodes token-conditioned keys and values from them; runs **state-memory attention (SMA)** over the resulting KV pairs; adds the retrieved output to the Mamba-2 output through a zero-initialized gated residual. Key details:
  - The Mamba-2 state is a compressed KV cache; Mamba-2 reads values via `C_t H_t` but does not decode token-conditioned keys. DART adds the key side.
  - For a chunk `c`, value `V_{t,c} = C_t ΔH_[c]`, key `K_{t,c} = ΔH_[c] E_t`, logit `ℓ_{t,c} = Q_t K_{t,c} / √N`, softmax over historical chunks, gated residual.
  - The number of chunk memories grows with sequence length / chunk size, so the inference cache is smaller than token-level attention but not `O(1)`. At chunk 256 / state 128, it saves 75% of the cache vs. token attention.
  - It substantially improves associative recall and retrieval vs. Mamba-2 while preserving LM quality.
- **Sessa** (2026, arXiv:2604.18580): places attention **inside a recurrent feedback path**. It uses a forward causal attention and a feedback attention whose weights form a strictly lower-triangular matrix `B_fb`; the output solves `(I − B_fb) y = f`. This gives multi-hop routing, power-law memory tails `O(ℓ^{−β})`, and flexible selective retrieval. The dense full-prefix formulation is quadratic in `T`; the recurrence form is linear per step. It is a concrete example of "attention in the state update" that is not a plain scan.
- **Attamba** (2024) and **RAT** (2025) perform attention over chunk-level representations produced by SSMs/recurrence. They are the "chunk-level attention" branch that DART cites.
- **Infini-attention** (Munkhdalai et al., 2024) combines local causal attention with a compressive memory (a linear attention state) in one block; it is an early "attention + compressed state in the same layer" design.
- **Compressive Transformer** (Rae et al., 2019), **Memorizing Transformers** (Wu et al., 2022), **Recurrent Memory Transformer** (Bulatov et al., 2022), and **Elastic Memory** (Song et al., 2026, arXiv:2602.11212) are the memory-compression lineage. Elastic Memory uses HiPPO-style online function approximation for a fixed-size memory and polynomial sampling for retrieval; it beats Memorizing Transformer with 16× less memory and Melodi at all memory sizes in their experiments.

**Relevance.** DART is the closest published system to "attention over state". It differs from the user's idea in one crucial way: DART attends over **time-chunk states** (a growing memory of past chunks), while the user proposes attention over **depth/timescale slots** (a fixed `L`-slot state at the current timestep). DART's memory is `O(T/S)`; the user's is `O(L)`. DART's advantage is exact-ish retrieval of past chunks; the user's advantage would be cheap selection among timescales. The two are complementary and could be combined, but they are not the same claim.

### 8.3 Cross-layer routing in linear attention

**"Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing"** (Cerruti et al., ETH Zurich, 2026, arXiv:2607.07953) is a careful 350M/15B-token comparison of DeltaNet, Gated DeltaNet, Kimi Delta Attention, and Gated DeltaNet-2, plus two cross-layer routing variants:
- **CLER (Cross-Layer Error Residuals)**: forward a lower layer's delta-rule write **error** into the next layer's value target. It does **not** improve over matched baselines.
- **CLER-H**: route the same error into the shared hidden stream. Still not a clear win.
- **CLVR (Cross-Layer Value Routing)**: route the layer's **write value** into the aligned hidden stream. It gives a small reduction in final validation loss for both DeltaNet and Gated DeltaNet, while preserving linear-time structure.
- The paper also notes that Attention Residuals and MoDA-style depth attention replace the residual sum / add depth pathways, and that their benefit did not transfer cleanly across the two linear-memory hosts they studied.

**Relevance.** This is a warning: naive cross-layer routing of recurrent-state signals often does not help. The useful signal was the write **value**, not the write error, and the gain was modest. A depth-attention design must be compared against this baseline and must explain why attention is better than a simple routed value.

---

## 9. Parallelization and kernels: what breaks when attention enters

### 9.1 The scan and its variants

- **Associative scan** (Blelloch, 1990; Martin & Cundy, 2018; Smith et al., S5, 2022): a linear recurrence `h_t = a_t h_{t−1} + b_t` is an associative operation, so it can be computed in `O(log T)` depth. This is the basis of S5 and many linear RNNs.
- **Chunkwise / blockwise algorithms**: split the sequence into chunks, compute within-chunk in parallel, and carry a chunk-level state. Mamba-2's SSD algorithm, Gated DeltaNet's WY representation, Kimi Delta Attention's UT transform, and DeltaProduct's chunkwise algorithm are all variants. They avoid materializing all intermediate matrices and keep I/O manageable.
- **Block-diagonal scans**: because block-diagonal matrices are closed under multiplication, the scan preserves the block structure; cost per composition is `O(d_h Σ b_j²)` rather than `O(d_h³)`. BD-LRU provides PyTorch and Triton implementations that are competitive with diagonal scans for moderate block sizes.
- **Fixed-point iterations**: instead of a scan, iterate a parallelizable diagonal RNN to a fixed point. The number of iterations is a compute–expressivity knob; empirically it adapts to task difficulty. This is a different parallelization strategy from scan and is directly relevant to "less depth".

### 9.2 What breaks with attention

| Where attention is placed | Recurrence in state | Parallel training | Notes |
|---|---|---|---|
| Input pre-filter (short conv / local attention) | linear | yes | standard; equivalent to a dynamic conv / feature extractor |
| Readout across state slots | linear | yes | attention is pointwise after the scan; cheapest option |
| Attention over stored past states (DART) | linear | yes (chunked) | memory grows with `T/S`; attention over chunks |
| Attention inside the state update (Sessa) | nonlinear | no plain scan; triangular solve / recurrence | can be chunked; full-prefix form is quadratic |
| Attention replacing the recurrence (Transformer) | none | yes | quadratic; growing KV cache |

The key engineering fact: **a readout-only attention preserves the scan**, because the state at time `t` does not depend on the attention output. The attention can be applied after the scan in a separate pass, or fused into the readout kernel. An update-coupled attention makes the state at time `t` depend nonlinearly on the state at time `t`, so the scan no longer applies; one needs a triangular solve (Sessa), a chunked approximation, or a fixed-point iteration.

### 9.3 Cost accounting for a depth-attention readout

Let the state be `H_t ∈ R^{L×d}`, with `L` slots and per-slot dimension `d`.

- **Linear readout**: `y_t = C_t H_t`, with `C_t ∈ R^{1×L}`. Cost `O(L d)` per token; params `O(L d)` (or `O(L d)` for a shared `C` plus input projection).
- **Dense readout**: `y_t = W H_t` with `W ∈ R^{d_out×Ld}`. Cost `O(L d d_out)`; params `O(L d d_out)`.
- **Depth attention readout**: `Q_t = x_t W_Q`, `K_t = H_t W_K`, `V_t = H_t W_V`; `y_t = softmax(Q_t K_tᵀ/√d) V_t`. Cost `O(L² d + L d²)`; params `O(L d² + d²)` if per-slot projections, or `O(d²)` if shared across slots.
- If `L` is small (4–16) and `d` is large (256–1024), the attention cost is dominated by the projections (`O(L d²)`), not the `L²` term. The parameter count is comparable to a dense readout; the win, if any, is inductive bias, not parameter savings.
- If the attention uses a **shared** Q/K/V projection across slots, the parameter count can be much smaller than a dense readout, but the slots become exchangeable unless a depth positional/timescale embedding is added.

---

## 10. Synthesis: the prior-art map and the actual gap

### 10.1 The two axes that matter

The proposed idea mixes two independent design choices:

**Axis A — state structure**
1. Single vector/matrix state (Mamba, DeltaNet).
2. Multi-head / block-diagonal state with dense intra-block mixing (BD-LRU, BD-SLiCE).
3. Multi-timescale layers with different decay (Clockwork, MS-SSM, Harmonic).
4. Block upper-triangular cascade (the user's "sandwich").
5. Higher-order tensor state (RunningTensor) or higher-order recurrence (H-LRU).

**Axis B — readout / inter-slot interaction**
1. Linear readout `C_t H_t` (Mamba, BD-LRU).
2. Input-dependent linear scale mixer (MS-SSM).
3. Softmax attention over slots (the user's depth attention).
4. Attention over time-chunk states (DART).
5. Attention over transformer depth (Depth-Attention, MoDA, Attention Residuals).
6. Cross-layer value routing (CLVR).

The literature has extensively explored A1–A3 and B1–B2, and is beginning to explore B3–B5 in adjacent settings. The **intersection of A3/A4 with B3** — a multi-timescale, block-structured recurrent state read out by softmax attention across its slots — is the least populated cell. That is the honest novelty claim.

### 10.2 What is already covered

| Claim | Prior art | Status |
|---|---|---|
| "Diagonal SSMs are less expressive; block-dense/BD transitions are maximally expressive" | Cirone 2024; SLiCE 2025; PD-SSM 2025; BD-LRU 2026 | Established |
| "A block-diagonal model can solve A5 with one layer" | SLiCE 2025; PD-SSM 2025 | Established |
| "Multi-timescale state with slow/fast layers helps long context" | Clockwork 2014; MS-SSM 2025; Harmonic 2026; MS-GLA 2026 | Established (with varying rigor) |
| "An input-dependent linear mixer across scales works" | MS-SSM 2025 | Established |
| "Attention over recurrent chunk states improves recall" | DART 2026 | Established (preprint) |
| "Attention over depth improves transformer quality cheaply" | Depth-Attention 2026; MoDA 2026; Attention Residuals 2026 | Established (preprints) |
| "Attention inside a recurrent feedback path gives power-law memory" | Sessa 2026 | Established theoretically (preprint) |
| "Local attention ≈ dynamic depthwise convolution" | Han et al. 2022; SASA 2019; Involution 2021 | Established |
| "More heads/state slots can span higher interaction rank" | "How Many Heads" 2025; MIMOMamba 2026; RunningTensor 2026 | Established (theory/preprint) |
| "Cross-layer routing of delta-rule signals can help a little" | CLVR 2026 | Established (preprint) |

### 10.3 What is plausibly new

1. **Depth-attention readout over a multi-timescale recurrent state.** DART attends over time-chunk states; Depth-Attention/MoDA attend over transformer layers; MS-SSM mixes scales linearly. A softmax attention over a **fixed set of recurrent timescale slots** at each token, with depth positional/timescale embeddings and a causal depth mask, is a specific combination that we have not found published as such.
2. **Block upper-triangular (cascade) transition with dense diagonal blocks.** BD-SLiCE/BD-LRU use block-diagonal transitions; a block upper-triangular transition with input-dependent leakage between timescale blocks is a different structure. It is closed under multiplication (scan-compatible), and it encodes "slow layers receive from fast layers". Its expressivity/stability properties are not covered by the existing theorems.
3. **A depth-wise convolution + attention kernel as a state readout.** A short causal conv across slots followed by attention is a "local + global" depth mixer. The primitive exists in vision (SASA/Involution), but its use as a *state readout* in an SSM is not standard.
4. **A parameter-matched comparison of state-attention vs. block-dense vs. multi-scale vs. depth-attention.** Even if the architecture is not radically new, a careful controlled study would be valuable, because the current literature compares across different scales, token budgets, and optimizers.

### 10.4 What is probably not new

- "Attention in a sliding kernel" as a primitive. SASA/Involution/dynamic conv already cover it. A 1D causal version is a local attention layer; it is not by itself a contribution.
- "A single local kernel gives global attention after sliding". It does not, unless you stack enough layers or use a global token. Be precise about receptive fields.
- "SSMs are convolutions". Only LTI SSMs. Selective SSMs are not.
- "Attention over the state gives state tracking". Not if the state update is affine. The transition structure is the lever.
- "Fewer layers because the state is richer". This is plausible for state tracking (BD-SLiCE) but not established for language modeling; BD-LRU's gains are modest on perplexity and larger on structure probes. The claim must be task-specific.

---

## 11. Open problems worth targeting

1. **A clean expressivity theorem for block upper-triangular input-dependent transitions.** Are they maximally expressive? What is the minimal block size? Do they preserve the TC⁰ escape of dense/BD transitions? This is a tractable theory contribution.
2. **Depth attention vs. a dense readout at matched parameters.** Is the gain from nonlinear selection, or just from extra parameters/compute? A parameter-matched dense `C_t` and an MLP readout are the controls.
3. **Depth attention vs. a linear scale mixer (MS-SSM).** MS-SSM's ablation found linear better; is that because softmax loses the magnitude information that a linear mixer preserves? A gated linear + softmax hybrid may be better.
4. **Depth attention vs. DART-style chunk attention.** Fixed `L` slots vs. growing chunk memory. Which is better for which kind of recall (within-chunk vs. long-range)?
5. **Depth attention vs. fixed-point/looped depth.** Both claim more capability per parameter/layer. A matched comparison would be informative.
6. **Stability and normalization of the cascade.** BD-LRU needed joint gate normalization; a block upper-triangular cascade needs an analogous condition (row-wise L1 over the full block row, including the leakage blocks). This is an engineering contribution with a theorem.
7. **When does content-based readout help?** A diagnostic: measure attention entropy over slots; if it collapses to one slot, the model is just selecting a timescale; if it is uniform, it is averaging. The useful regime is input-dependent routing.
8. **Kernel support.** A fused kernel that computes the block-triangular scan and the depth attention in one pass would make the architecture practical. Without it, the `L²` and projection costs may dominate at small `L`.
9. **Hybrid placement.** Where does a depth-attention SSM block belong in a hybrid stack (e.g., between full-attention layers)? The "What Attention Recalls" result suggests the recurrent side supplies language/format and the KV side supplies values; a depth-attention block may improve the recurrent side's value retrieval without a full KV cache.

---

## 12. Bibliography (grouped, with arXiv IDs where verified)

### RNNs, linear RNNs, gated linear attention

- Hochreiter & Schmidhuber, *Long Short-Term Memory*, Neural Computation, 1997. (no arXiv)
- Cho et al., *Learning Phrase Representations using RNN Encoder–Decoder*, EMNLP 2014, arXiv:1406.1078.
- Koutnik et al., *A Clockwork RNN*, ICML 2014, arXiv:1402.3511.
- Orvieto et al., *Resurrecting Recurrent Neural Networks for Long Sequences* (LRU), ICML 2023, arXiv:2303.06349.
- Feng et al., *Were RNNs All We Needed?* (minGRU/minLSTM), 2024, arXiv:2410.01201.
- Qin et al., *Hierarchically Gated Recurrent Neural Network* (HGRN), 2023, arXiv:2311.04823.
- Qin et al., *HGRN2: Gated Linear RNNs with State Expansion*, 2024, arXiv:2404.07904.
- Beck et al., *xLSTM: Extended Long Short-Term Memory*, 2024, arXiv:2405.04517.
- Peng et al., *RWKV: Reinventing RNNs for the Transformer Era*, 2023, arXiv:2305.13048.
- Peng et al., *RWKV-7 "Goose" with Expressive Dynamic State Evolution*, 2025, arXiv:2503.14456.
- Yang et al., *Gated Linear Attention Transformers with Hardware-Efficient Training* (GLA), 2023, arXiv:2312.06635.
- Sun et al., *Retentive Network: A Successor to Transformer for Large Language Models* (RetNet), 2023, arXiv:2307.08621.
- De et al., *Griffin: Mixing Gated Linear Recurrences with Local Attention*, 2024, arXiv:2402.19427.
- Botev et al., *RecurrentGemma: Moving Past Transformers for Efficient Open Language Models*, 2024, arXiv:2404.07839.

### SSMs and structured transitions

- Gu et al., *HiPPO: Recurrent Memory with Optimal Polynomial Projections*, NeurIPS 2020, arXiv:2008.07669.
- Gu et al., *Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers* (S4), 2021, arXiv:2110.13985.
- Gupta et al., *Diagonal State Spaces are as Effective as Structured State Spaces* (DSS), 2022, arXiv:2203.14343.
- Gupta et al., *On the Parameterization and Initialization of Diagonal State Space Models* (S4D), 2022, arXiv:2206.11893.
- Smith et al., *Simplified State Space Layers for Sequence Modeling* (S5), 2022, arXiv:2208.04933.
- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023, arXiv:2312.00752.
- Dao & Gu, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2), 2024, arXiv:2405.21060.
- Lahoti et al., *Mamba-3: Improved Sequence Modeling using State Space Principles*, ICLR 2026, arXiv:2603.15569.
- Liang & Song, *Understanding Mamba-3 Through Oscillatory Dynamics*, 2026, arXiv:2610.13341.
- Cirone et al., *Theoretical Foundations of Deep Selective State-Space Models*, 2024, arXiv:2402.19047.
- Walker et al., *Structured Linear CDEs: Maximally Expressive and Parallel-in-Time Sequence Models* (SLiCE/BD-SLiCE), NeurIPS 2025, arXiv:2505.17761.
- Terzić et al., *Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models* (PD-SSM), 2025, arXiv:2509.22284.
- Movahedi et al., *Fixed-Point RNNs: Interpolating from Diagonal to Dense*, NeurIPS 2025, arXiv:2503.10799.
- Dubinin, Orvieto, Effenberger, *Improved State Mixing in Higher-order and Block Diagonal Linear Recurrent Networks* (BD-LRU/H-LRU), 2026, arXiv:2602.12021.

### Delta rule, fast weights, test-time learning

- Schlag, Irie, Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers*, ICML 2021, arXiv:2102.11174.
- Schlag, Irie, Schmidhuber, *Going Beyond Linear Transformers with Recurrent Fast Weight Programmers* (DeltaNet), 2021, arXiv:2106.06295.
- Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule*, ICLR 2025, arXiv:2412.06464.
- Hatamizadeh et al., *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention*, 2026, arXiv:2605.22791.
- Siems et al., *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products*, 2025, arXiv:2502.10297.
- Sun et al., *Learning to (Learn at Test Time): RNNs with Expressive Hidden States* (TTT), 2024, arXiv:2407.04620.
- Behrouz et al., *Titans: Learning to Memorize at Test Time*, 2024/2025, arXiv:2501.00663.
- *MesaNet: Sequence Modeling by Locally Optimal Test-Time Training*, 2025, arXiv:2506.05233.
- *Test-Time Training Done Right*, 2025, arXiv:2505.23884.
- Herranz-Celotti et al., *RunningTensor: Generalizing Linear Attention to Higher-Order Recurrent States*, 2026, arXiv:2609.12814.
- *Expanding Attention: A Taylor-Series Route to Linear-Time Transformers with a Fixed Number of Recurrent States*, 2026, arXiv:2610.20637.

### CDEs, signatures, continuous time

- Chen et al., *Neural Ordinary Differential Equations*, NeurIPS 2018, arXiv:1806.07366.
- Kidger et al., *Neural Controlled Differential Equations for Irregular Time Series*, NeurIPS 2020, arXiv:2005.08926.
- Morrill et al., *Neural CDEs for Long Time Series via the Log-ODE Method*, 2020, arXiv:2009.08295.
- Kidger & Lyons, *Signatory: Differentiable Computations of the Signature and Logsignature Transforms*, ICLR 2021, arXiv:2001.00706.
- Hasani et al., *Closed-form Continuous-time Neural Networks* (CfC), Nature Machine Intelligence 2022, arXiv:2106.13898.
- Bloch et al., *Extending SSMs with the Exponentially Weighted Signature* (EWS), 2026, arXiv:2603.19198.
- Walker et al., *Log Neural Controlled Differential Equations: The Lie Brackets Make a Difference*, ICML 2024.

### Attention, local kernels, hybrids

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017, arXiv:1706.03762.
- Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*, ICML 2020, arXiv:2006.16236.
- Choromanski et al., *Rethinking Attention with Performers*, 2020, arXiv:2009.14794.
- Wang et al., *Linformer: Self-Attention with Linear Complexity*, 2020, arXiv:2006.04768.
- Beltagy et al., *Longformer: The Long-Document Transformer*, 2020, arXiv:2004.05150.
- Zaheer et al., *Big Bird: Transformers for Longer Sequences*, NeurIPS 2020, arXiv:2007.14062.
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022, arXiv:2205.14135.
- Ramachandran et al., *Stand-Alone Self-Attention in Vision Models* (SASA), NeurIPS 2019, arXiv:1906.05909.
- Li et al., *Involution: Inverting the Inherence of Convolution for Visual Recognition*, ICCV 2021, arXiv:2103.06255.
- Chen et al., *Dynamic Convolution: Attention over Convolution Kernels*, CVPR 2020, arXiv:1912.03458.
- Han et al., *On the Connection between Local Attention and Dynamic Depth-wise Convolution*, ICLR 2022, arXiv:2106.04263.
- Ma et al., *Mega: Moving Average Equipped Gated Attention*, 2022, arXiv:2209.10655.
- Poli et al., *Hyena Hierarchy: Towards Larger Convolutional Language Models*, NeurIPS 2023, arXiv:2302.10866.
- Arora et al., *Zoology: Measuring and Improving Recall in Efficient Language Models*, 2023, arXiv:2312.04927.
- Arora et al., *Simple Linear Attention Language Models Balance the Recall–Throughput Tradeoff* (Based), 2024, arXiv:2402.18668.
- Zhang et al., *The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry*, 2024, arXiv:2402.04347.
- Jelassi et al., *Repeat After Me: Transformers are Better than State Space Models at Copying*, 2024, arXiv:2402.01032.
- Lieber et al., *Jamba: A Hybrid Transformer–Mamba Language Model*, 2024.
- Ren et al., *Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling*, 2024.
- Glorioso et al., *Zamba: A Compact 7B SSM Hybrid Model*, 2024.
- Dong et al., *Hymba: A Hybrid Head Architecture for Small Language Models*, 2024.
- *Priming: Hybrid State Space Models From Pre-trained Transformers*, 2026, arXiv:2605.08301.
- *What Attention Recalls and Recurrence Controls in Hybrid Language Models*, 2026, arXiv:2609.04434.

### Multi-scale, memory, depth attention

- Chen et al., *MELODI: Exploring Memory Compression for Long Contexts*, 2024, arXiv:2410.03156.
- Karami et al., *MS-SSM: A Multi-Scale State Space Model for Efficient Sequence Modeling*, 2025, arXiv:2512.23824.
- Dev et al., *MS-GLA: Multi-Scale Gated Linear Attention*, 2026, arXiv:2609.35664.
- Nyoma, *Harmonic: Hierarchical State Space Models for Efficient Long-Context Language Modeling*, 2026, arXiv:2606.24650.
- Qian et al., *DART: Decoded Attention over Recurrent States for Efficient Long-Context Sequence Modeling*, 2026, arXiv:2608.02032.
- Kimi Team, *Attention Residuals*, 2026, arXiv:2603.15031.
- Zhu et al., *Mixture-of-Depths Attention* (MoDA), 2026, arXiv:2603.15619.
- Zeng et al., *Depth-Attention: Cross-Layer Value Mixing for Language Models*, 2026, arXiv:2606.05014.
- Cerruti et al., *Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing*, 2026, arXiv:2607.07953.
- *Sessa: Selective State Space Attention*, 2026, arXiv:2604.18580.
- Song et al., *Towards Compressive and Scalable Recurrent Memory* (Elastic Memory), 2026, arXiv:2602.11212.
- *Fan Duality Model: Breaking the KV Cache Bottleneck*, 2026, arXiv:2604.07716.
- *Echo: KV-Cache-Free Associative Recall with Spectral Koopman Operators*, 2026, arXiv:2605.06997.
- *Infini-attention* (Munkhdalai et al., 2024).
- Rae et al., *Compressive Transformers for Long-Range Sequence Modelling*, 2019.
- Wu et al., *Memorizing Transformers*, 2022.
- Bulatov et al., *Recurrent Memory Transformer*, 2022.

### Expressivity and state tracking

- Merrill & Sabharwal, *The Parallelism Tradeoff: Limitations of Log-Precision Transformers*, 2022, arXiv:2207.00729.
- Merrill, Petty, Sabharwal, *The Illusion of State in State-Space Models*, ICML 2024, arXiv:2404.08819.
- Sarrof, Veitsman, Hahn, *The Expressive Capacity of State Space Models: A Formal Language Perspective*, 2024, arXiv:2405.17394.
- Chen et al., *The Computational Limits of State-Space Models and Mamba via the Lens of Circuit Complexity*, 2024, arXiv:2412.06148.
- Brandoit et al., *Length-Independent State Tracking Under a Parallel Scan*, 2026, arXiv:2609.32447.
- *Rethinking State Tracking in Recurrent Models Through Error Control Dynamics*, 2026, arXiv:2605.07755.
- Zubić et al., *On the Expressive Power and Limitations of Multi-Layer SSMs*, 2026, arXiv:2604.14501.
- Ghodsi, *How Many Heads Make an SSM? A Unified Framework for Attention and State Space Models*, 2025, arXiv:2512.15115.
- Huang et al., *Understanding Input Selectivity in Mamba*, ICML 2025, arXiv:2506.11891.
- Okpekpe & Orvieto, *Revisiting Associative Recall in Modern Recurrent Models*, 2025, arXiv:2508.19029.
- Liu et al., *Transformers Learn Shortcuts to Automata*, 2022, arXiv:2210.10749.
- Hahn, *Theoretical Limitations of Self-Attention in Neural Sequence Models*, 2019, arXiv:1906.06755.

### Surveys and general references

- *Technologies on Effectiveness and Efficiency: A Survey of State Spaces Models*, 2025, arXiv:2503.11224.
- *Efficient Attention Mechanisms for Large Language Models: A Survey*, 2025, arXiv:2507.19595.
- *Not All Attention Is Equal: A Quantitative Survey of the EEI Trade-off*, 2026, arXiv:2608.15459.
- Huang et al., *Towards Looped Models Done Right, Part I*, 2026, arXiv:2610.23036.
- MIMOMamba: *From Scalar Duality to Matrix-Valued Attention*, 2026 (web preprint; no arXiv ID verified).
- *A Comparative Analysis of Attention versus State-Space Models for In-Context Learning*, 2026, arXiv:2609.32341.
