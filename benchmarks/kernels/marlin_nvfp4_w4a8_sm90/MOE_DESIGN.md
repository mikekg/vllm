# Runtime-selective NVFP4-to-FP8 MoE

<!-- markdownlint-disable MD060 -->

## Scope

This document describes the runtime-selective MoE path that keeps packed
NVFP4 weights resident and chooses between two execution paths for each MoE
layer invocation:

- the existing grouped W4A16 Marlin path;
- transient conversion to block-scaled FP8 followed by grouped DeepGEMM.

The objective is end-to-end inference throughput above the W4A16 baseline,
not merely functional FP8 execution. The current target is a 20–40% model-level
improvement where the workload exposes enough routed work to amortize
conversion.

The dense implementation and its native hybrid operator are described in
[DENSE_DESIGN.md](DENSE_DESIGN.md). MoE reuses its NVFP4 decoding and scale
conversion logic, but routing and DeepGEMM row padding make the MoE crossover a
different problem.

## Terminology and shapes

For one MoE layer invocation:

- `M`: tokens entering the router.
- `E`: number of experts represented by the local expert tensors.
- `T`: selected experts per token (`top_k`).
- `K`: input hidden dimension.
- `N`: one expert's intermediate dimension.
- `r_i`: real routed rows assigned to expert `i`.
- `R`: total real routed rows.
- `p_i`: rows presented to DeepGEMM after its per-expert alignment.
- `P`: total padded routed rows.
- `A`: DeepGEMM row alignment, currently 128.

The routing identities are

\[
R = M T = \sum_{i=1}^{E} r_i
\]

and

\[
\bar r = \frac{R}{E} = \frac{M T}{E}.
\]

For a perfectly balanced controlled workload, every expert receives
`r_i = r_bar`. Real serving is not balanced, so `r_bar` alone does not
describe the grouped GEMM cost.

If all `MT` routes were sampled independently and uniformly, the expected
number of experts receiving at least one route would be

\[
E_{active}
=E\left[1-\left(1-\frac{1}{E}\right)^{MT}\right].
\]

Actual top-k routing selects `T` distinct experts for each token. Under a
uniform without-replacement top-k model, an expert is absent from one token
with probability `1-T/E`, so the corresponding expectation is

\[
E_{active}
=E\left[1-\left(1-\frac{T}{E}\right)^M\right].
\]

This occupancy estimate approaches E quickly for the measured Q3 batches. It
describes how many experts are touched, but it does not determine grouped
cost: the complete vector of routed counts and its 128-row rounding still
controls the DeepGEMM problem.

The two expert projections have these shapes:

\[
W_{13,i}: K \times 2N
\]

\[
W_{2,i}: N \times K.
\]

For expert `i`, the corresponding GEMMs are

\[
(r_i \times K)(K \times 2N)
\]

and

\[
(r_i \times N)(N \times K).
\]

`W13` combines the gate and up projections. Its output passes through the
gated activation before `W2`.

## Router and expert execution are separate

The router gate is a dense linear layer that produces logits of shape
`[M, E]`. Top-k selection turns those logits into expert indices and routing
weights. The expert MLP then consumes the routed rows.

This distinction matters for both performance and accuracy:

- the MoE hybrid controls `W13` and `W2`;
- the dense hybrid can independently affect the router gate;
- a model result using both hybrids cannot by itself attribute an accuracy
  change to the expert path.

In Q3, the router gate is an ordinary quantized `ReplicatedLinear`, rather than
a distinct `GateLinear` type. A dense layer cannot infer from its own input and
weight shapes that its output will later feed a router.

The model tree does retain the semantic relationship. `FusedMoEFactory`
passes the actual gate module into `MoERunner`, and `MoERunner` stores that
object as `runner.gate`. A model-level traversal can therefore identify the
exact gate objects by identity:

```python
router_gates = {
    id(runner.gate): runner.gate
    for runner in model.modules()
    if isinstance(runner, MoERunner) and runner.gate is not None
}
```

A static attribute such as `routed_gate` can be set after weight finalization
and before the first compilation. The dense selector can then remain local:

```python
if not getattr(layer, "routed_gate", False) and M >= knee_m:
    return fp8_path(...)
return marlin_path(...)
```

Dynamo specializes the static attribute, while `M` remains the runtime
quantity. `model.process_weights_after_loading()` runs late enough for this
attribute update because the update does not alter the packed weight layout
and compilation occurs afterward.

There is currently no generic public registration point that installs such a
pre-compilation model-tree processor across all built-in models. Directly
changing `MoERunner` to mark its gate would provide the information, but it
would make the expert implementation responsible for policy in a separate
linear module. That option is recorded rather than used by the current
prototype. Name matching, prefix matching, and shape matching are weaker
alternatives because ordinary dense projections can satisfy the same tests.

## Resident and transient representations

The resident weights remain the original packed NVFP4 tensors for both expert
stacks. The high-M path creates block-scaled FP8 weights only for the duration
of the invocation.

The execution sequence is:

1. Route the input tokens and form the grouped expert problem.
2. Convert the complete resident `W13` expert stack from packed NVFP4 to FP8.
3. Quantize the routed activations.
4. Execute grouped FP8 `W13` with DeepGEMM.
5. Apply the gated activation and requantize the intermediate activations.
6. Reuse the same transient weight and scale arena for `W2`.
7. Convert the complete resident `W2` stack from packed NVFP4 to FP8.
8. Execute grouped FP8 `W2`.
9. Unpermute and apply the routing weights.

The low-M path executes the existing grouped Marlin implementation from the
resident NVFP4 tensors.

The sequential conversion schedule means that `W13` and `W2` do not need
simultaneous FP8 storage. `W2` overwrites the transient arena only after the
first grouped GEMM has consumed `W13`.

There is no arithmetic performed "in FP8" during conversion. The converter
decodes the packed E2M1 values and the processed E4M3 scale representation,
combines them with the per-expert global scaling information and the selected
tile divisor, and emits:

- E4M3 weight values;
- FP32 scales for 128-by-128 DeepGEMM blocks.

This is the same scale transformation used by the dense path, extended across
the expert dimension.

## Q3 concrete shape record

The Q3 model used for the first complete cost curve has:

- `E = 128`;
- `T = 8`;
- `K = 2048`;
- `N = 768`.

Its balanced routing relation is therefore

\[
\bar r = \frac{M \cdot 8}{128} = \frac{M}{16}
\]

or, inverted,

\[
M = 16\bar r.
\]

The per-expert GEMMs are:

\[
W13:\quad
(r_i \times 2048)(2048 \times 1536)
\]

\[
W2:\quad
(r_i \times 768)(768 \times 2048).
\]

The number of weight elements per expert is:

\[
|W13_i| = 2048 \cdot 1536 = 3{,}145{,}728
\]

\[
|W2_i| = 768 \cdot 2048 = 1{,}572{,}864
\]

\[
|W13_i| + |W2_i|
= 4{,}718{,}592.
\]

Across 128 experts:

\[
128 \cdot 4{,}718{,}592
= 603{,}979{,}776
\]

FP8 weight elements are converted over the complete two-GEMM route.

The maximum simultaneous FP8 weight storage is the larger `W13` stack:

\[
128 \cdot 2048 \cdot 1536
= 402{,}653{,}184
\]

bytes.

Its 128-by-128 FP32 scale tensor contains

\[
128
\cdot \frac{2048}{128}
\cdot \frac{1536}{128}
= 24{,}576
\]

values, or

\[
24{,}576 \cdot 4 = 98{,}304
\]

bytes.

`W2` needs 201,326,592 FP8 bytes and 49,152 scale bytes, so it fits in the same
arena after `W13` has been consumed.

## First-order crossover derivation

For symmetric expert shapes, converting all expert weights touches:

\[
E(2KN + NK) = 3EKN
\]

weight elements.

The routed expert GEMMs perform work proportional to:

\[
MT(2KN + NK) = 3MTKN
\]

multiply-accumulates.

The idealized ratio of conversion work to useful GEMM work is therefore

\[
\frac{3EKN}{3MTKN}
= \frac{E}{MT}
= \frac{1}{\bar r}.
\]

This produces the useful first-order mapping

\[
M_{\text{knee}}
\approx r_{\text{knee}}\frac{E}{T}.
\]

For Q3:

\[
M_{\text{knee}}
\approx 16r_{\text{knee}}.
\]

The formula explains why total token count by itself is not portable between
models. For example, if an otherwise identical grouped problem had 16 experts
and selected 4 per token, the mapping would be:

\[
M_{\text{knee}}
\approx r_{\text{knee}}\frac{16}{4}
= 4r_{\text{knee}}.
\]

A provisional per-expert interval of 256–320 rows would map to an input-M
interval of 1024–1280 in that otherwise identical case.

The cancellation of `K*N` is only an operation-count intuition. It is not a
production crossover law. Measured conversion and GEMM times depend
differently on:

- converter bandwidth and launch cost;
- `K` and `N` tile utilization;
- matrix aspect ratio;
- Marlin tile behavior;
- DeepGEMM tile behavior;
- routed-row padding;
- activation quantization;
- permutation and reduction;
- the GPU and software backend.

The useful generalization key is therefore the actual local execution shape
and backend:

```text
(E, T, K, N, input dtype, GPU, Marlin backend, FP8 backend)
```

rather than model identity. `K` and `N` remain explicit parts of that key even
though they cancel in the idealized ratio.

## Full route-time model

The W4A16 route can be represented as

\[
\begin{aligned}
T_4(\{r_i\}) ={}&
T_{\text{route},4} \\
&+ \sum_i
  \left[
    G_{4,13}(r_i,K,2N)
    + G_{4,2}(r_i,N,K)
  \right] \\
&+ T_{\text{activation},4}

- T_{\text{reduce},4}.
\end{aligned}
\]

The transient FP8 route can be represented as

\[
\begin{aligned}
T_8(\{r_i\}) ={}&
C_{13}(E,K,2N)

- C_2(E,N,K) \\
&+ T_{\text{A1 quant}}
- T_{\text{permute}} \\
&+ \sum_i
  G_{8,13}(p_i,K,2N) \\
&+ T_{\text{SiLU+A2 requant}} \\
&+ \sum_i
  G_{8,2}(p_i,N,K) \\
&+ T_{\text{unpermute/reduce}}.
\end{aligned}
\]

The summation is a cost model for a ragged grouped problem. The implementation
does not launch a Python GEMM loop over experts; DeepGEMM executes the grouped
operation.

The local advantage is

\[
\Delta(\{r_i\}) = T_4(\{r_i\}) - T_8(\{r_i\}).
\]

A positive value means the FP8 route saves time. This expression makes the
actual tradeoff visible: conversion is paid for every selected high-path
invocation across the full expert stack, while the useful GEMM work is set by
the routed rows.

## DeepGEMM row rounding

DeepGEMM aligns each expert's routed row count independently:

\[
p_i = A\left\lceil\frac{r_i}{A}\right\rceil,
\qquad A = 128.
\]

The total padded work is

\[
P = \sum_i p_i.
\]

A useful routing-efficiency statistic is

\[
\eta = \frac{R}{P}
= \frac{\sum_i r_i}
       {\sum_i 128\lceil r_i/128\rceil}.
\]

Two batches with identical `M`, `E`, and `T` can therefore have different FP8
costs. A concentrated routing distribution can use fewer aligned groups than a
uniform distribution, while a distribution that leaves many experts barely
nonempty can incur more padding.

For the balanced Q3 harness, the exact alignment points are:

| Input `M` | Real rows/expert `r` | Padded rows/expert `p` | Total real routes `R` | Total padded rows `P` | Efficiency `η` |
|---:|---:|---:|---:|---:|---:|
| 4096 | 256 | 256 | 32,768 | 32,768 | 100.000% |
| 5120 | 320 | 384 | 40,960 | 49,152 | 83.333% |
| 6144 | 384 | 384 | 49,152 | 49,152 | 100.000% |
| 7168 | 448 | 512 | 57,344 | 65,536 | 87.500% |
| 8192 | 512 | 512 | 65,536 | 65,536 | 100.000% |

The alignment creates a staircase in the FP8 cost. A particularly useful
boundary probe is:

| Input `M` | Real rows/expert | Padded rows/expert |
|---:|---:|---:|
| 4080 | 255 | 256 |
| 4096 | 256 | 256 |
| 4112 | 257 | 384 |

Moving from `M=4096` to `M=4112` adds only one real row per expert, but changes
the aligned capacity from 256 to 384 rows per expert, a 50% increase. This is a
possible performance cliff and is more informative than another point far
inside an alignment bucket.

The existing coarse plot labels its horizontal axis as “real routed rows per
expert.” For the controlled curve, those values are the deliberately balanced
mean `M*T/E`; they are not a measured production routing distribution.

## Q3 controlled complete-route measurements

The controlled benchmark times the complete route:

- `W13` conversion;
- first activation quantization;
- permutation/routing;
- grouped `W13`;
- gated activation and second quantization;
- `W2` conversion into the reused arena;
- grouped `W2`;
- weighted unpermute/reduction.

The measured coarse points are:

| Input `M` | Real rows/expert | Padded rows/expert | Marlin route | Hybrid route | Time saved | Throughput change |
|---:|---:|---:|---:|---:|---:|
| 4096 | 256 | 256 | 1.408080 ms | 1.417888 ms | -0.697% | -0.692% |
| 5120 | 320 | 384 | 1.764000 ms | 1.545632 ms | +12.379% | +14.128% |
| 6144 | 384 | 384 | 2.112944 ms | 1.655072 ms | +21.670% | +27.665% |
| 7168 | 448 | 512 | 2.442800 ms | 1.736320 ms | +28.921% | +40.688% |
| 8192 | 512 | 512 | 2.766864 ms | 1.824064 ms | +34.075% | +51.687% |

Only complete-route totals were retained for these coarse points. The harness
did not emit per-stage timings, so no conversion, quantization, permutation,
GEMM, activation, or reduction split is inferred from this table.

The `M=4096` point came from the older simultaneous-memory harness. It remains
useful as a near-equality observation, but the exact value can move under the
sequential arena implementation.

The first coarse sample exceeding 20% throughput was `M=6144`. Treating that
as the selector knee would discard the measured 14.128% gain at `M=5120`.
The 20–40% figure is a model-level project outcome, not a reason to reject a
positive layer-level saving.

For this Q3 shape, the observed crossover is bracketed by:

- 256 real rows/expert: approximately equal;
- 320 real rows/expert: a clear 14.128% throughput improvement.

The corresponding input-M bracket is:

\[
256 \cdot \frac{128}{8} = 4096
\]

to

\[
320 \cdot \frac{128}{8} = 5120.
\]

Splitting that interval gives:

\[
r_{\text{mid}} = \frac{256 + 320}{2} = 288
\]

and

\[
M_{\text{mid}}
= 288 \cdot \frac{128}{8}
= 4608.
\]

Within the aligned-384 bucket, linear extrapolation below the measured 5120 and
6144 times gives the provisional `M=4608` estimate:

- Marlin: approximately 1.590 ms;
- hybrid: approximately 1.491 ms;
- time reduction: approximately 6.2%;
- throughput increase: approximately 6.6%.

Those are estimates, not measurements.

Within one fixed alignment and kernel-tile bucket, a useful local
approximation is:

\[
T_{\text{Marlin}}(M) \approx a + bM
\]

and

\[
T_{\text{hybrid}}(M) \approx C + c + dM,
\]

where `C` represents conversion and other fixed high-path work. The throughput
ratio

\[
\frac{T_{\text{Marlin}}(M)}
     {T_{\text{hybrid}}(M)} - 1
\]

is fractional-linear, or hyperbola-like, rather than linear. Crossing an
alignment boundary changes the coefficients abruptly, producing the observed
staircase or sawtooth around the smooth trend.

The fine-curve probe set is:

```text
3072, 3584, 4080, 4096, 4112, 4352,
4608, 4864, 5120, 6144, 7168, 8192
```

It covers points below the observed crossover, both sides of the 256-row
DeepGEMM boundary, the midpoint estimate, and the existing coarse upper
points.

## Selector interpretations

There are two useful ways to interpret the Q3 boundary.

### Positive-gain selector

The selector changes paths at the lowest repeatable point where

\[
T_8 < T_4.
\]

This collects every measurable local saving, including a 5–15% improvement
that contributes toward the model-level result.

The `M=5120` point is already a clear positive-gain observation. The exact
lower boundary depends on the fine probes around 4096–5120.

### Non-inferiority selector

Near a measured tie, choosing either path has negligible regret. This allows a
selector near `M=4096` if repeated measurements show that both paths remain
within run-to-run noise there.

This criterion is distinct from claiming a 20% local improvement. Its value is
that a small selector error near equal runtime has little end-to-end cost,
while moving the selector too high discards real gains.

The current lookup remains fail-closed: `_lookup_moe_m_knee` returns no hybrid
selection for an unknown shape unless a benchmark override or calibrated entry
exists. The controlled Q3 curve supplies evidence for one shape, not a
model-name hardcode.

## Actual-model throughput

### Variant definitions

The actual-model experiments use three configurations:

- **A:** dense Marlin and MoE Marlin;
- **B:** dense Marlin and hybrid MoE;
- **C:** hybrid dense and hybrid MoE.

A versus B isolates the MoE path. B versus C measures the incremental effect
of the dense hybrid, including any router-gate behavior.

### Q3, 8k/1k

All three Q3 runs used:

- 128 requests;
- 1,048,576 input tokens;
- 131,072 output tokens;
- fixed 32 GiB KV cache.

| Variant | Job | Duration | Throughput | Change from A | Change from B |
|---|---:|---:|---:|---:|---:|
| A | 6644863 | 80.839203 s | 1621.391532 tok/s | — | — |
| B | 6644868 | 68.884268 s | 1902.785689 tok/s | +17.355% | — |
| C | 6644870 | 65.459893 s | 2002.325297 tok/s | +23.494% | +5.231% |

The MoE hybrid alone accounts for a 17.355% end-to-end improvement. Adding the
dense hybrid raises the total improvement to 23.494%.

The initial B/C run used diagnostic `m_knee=6144`, the first coarse point
above the project-level 20% target rather than an optimized local crossover.
Job 6645558 later completed the B/C GSM8K evaluations at cutoffs 4608 and
4096 as one serial job with a five-minute service-shutdown interval. Those
accuracy and evaluation-runtime results appear below; they are not replacement
8k/1k serving-throughput measurements.

### Q36M, 8k/1k

All three Q36M runs used:

- fixed 32 GiB KV cache;
- 1,875,072 KV-token capacity;
- maximum reported concurrency 192.75 at sequence length 9728;
- identical input and output token counts across A, B, and C.

| Variant | Job | Duration | Throughput | Change from A | Change from B |
|---|---:|---:|---:|---:|---:|
| A | 6645223 | 45.222909 s | 2898.353991 tok/s | — | — |
| B | 6645230 | 41.311501 s | 3172.772618 tok/s | +9.468% | — |
| C | 6645235 | 40.971829 s | 3199.076095 tok/s | +10.376% | +0.829% |

The logs confirm that `fused_moe_kernel` executed in B and C. These runs did
not record the exact per-layer route vector, padded-row count, or branch count,
so the lower Q36M improvement cannot yet be assigned to one routing or
alignment feature. Their diagnostic knee was 5120.

## Accuracy evidence

Paired job 6645286 used the Q3 A/B/C split, diagnostic `m_knee=6144`, and
1,319 examples.

| Variant | Correct | Accuracy | Change from A | Discordant pairs vs A | McNemar p |
|---|---:|---:|---:|---:|---:|
| A | 1172 | 88.8552% | — | — | — |
| B | 1171 | 88.7794% | -0.0758 pp | 16 regressions / 15 improvements | 1.0000 |
| C | 1157 | 87.7180% | -1.1372 pp | 30 regressions / 15 improvements | 0.0356978 |

The MoE-only B result differs from A by one example and the paired changes are
symmetric. The combined C result is lower and its paired asymmetry is
statistically visible in this sample.

Because B changes only the expert path while C also changes dense layers, the
current evidence points away from the MoE conversion itself as the source of
the C regression. The router gate is one specific dense layer requiring
separate treatment because small changes in its logits can alter discrete
top-k choices.

Job 6645558 completed the 4608 and 4096 B/C reruns. The same 1,319 prompts and
the A details from job 6645286 give the following paired results:

| Knee | Variant | Correct | Accuracy | Change from A | Regressions / improvements | McNemar p |
|---:|---|---:|---:|---:|---:|---:|
| 4608 | B | 1157 | 87.7180% | -1.1372 pp | 27 / 12 | 0.0237027 |
| 4608 | C | 1164 | 88.2487% | -0.6065 pp | 23 / 15 | 0.2558751 |
| 4096 | B | 1174 | 89.0068% | +0.1516 pp | 11 / 13 | 0.8388197 |
| 4096 | C | 1165 | 88.3245% | -0.5307 pp | 26 / 19 | 0.3712980 |

The aggregate evaluator timings were:

| Knee | Variant | Evaluation time | Questions/s | Output tokens/s | Output tokens |
|---:|---|---:|---:|---:|---:|
| 6144 | A | 45.981808 s | 28.685258 | 4232.608709 | 194623 |
| 6144 | B | 46.543811 s | 28.338891 | 4189.472122 | 194994 |
| 6144 | C | 47.319359 s | 27.874427 | 4144.730722 | 196126 |
| 4608 | B | 46.554372 s | 28.332462 | 4207.295499 | 195868 |
| 4608 | C | 46.596236 s | 28.307007 | 4194.394560 | 195443 |
| 4096 | B | 46.232126 s | 28.529945 | 4226.865950 | 195417 |
| 4096 | C | 46.982392 s | 28.074347 | 4146.723720 | 194823 |

Generated output lengths differ across these accuracy variants, so this table
describes the evaluation runs rather than a matched-token serving throughput
comparison. The paired correctness result changes with the cutoff: B at 6144
was neutral, B at 4608 was lower in this sample, and B at 4096 was neutral.
Each cutoff changes which invocations use the hybrid path, but these serial
runs do not separate that effect from run-to-run generation variability.

## Production routing data

A controlled balanced harness gives an exact cost curve, but serving selects
experts from real router logits. The useful per-layer record is:

- input `M`;
- `E`, `T`, `K`, and `N`;
- total real routes `R`;
- complete route-count vector `{r_i}`, or an equivalent histogram;
- minimum, median, mean, and maximum `r_i`;
- padded counts `{p_i}`;
- total padded rows `P`;
- routing efficiency `eta`;
- selected branch;
- `W13` and `W2` conversion counts;
- grouped GEMM counts.

The route vector is layer-specific even when the input token count is shared
across layers. It reveals whether a lower observed gain comes from:

- many nearly empty experts;
- a small number of hot experts;
- widespread 128-row rounding;
- conversion on steps whose real routed work is too small;
- a selector that ignores the actual local expert shape.

Aggregation after the profiling window avoids writing a trace record for
every kernel call. The existing plot can then show both the controlled
balanced curve and actual serving distributions without presenting
`M*T/E` as an observed per-expert count.

## Model-independent shape handling

Most fused expert stacks are symmetric within a layer because grouped GEMM
expects one common expert tensor shape. The relevant dimensions are the actual
local tensors after tensor parallelism or expert parallelism, not only the
global model configuration.

The initial validation inventory includes:

| Model shape | Experts `E` | Top-k `T` | Hidden `K` | Intermediate `N` | Parallel note |
|---|---:|---:|---:|---:|---|
| Q3 | 128 | 8 | 2048 | 768 | measured |
| Q36M | 256 | 8 | 2048 | 512 | actual-model A/B/C measured |
| Gemma 4 candidate | 128 | 8 | 2816 | 704 | TP4 local `N=176` |
| Nemotron Nano candidate | 128 | 6 | 2688 | 1856 | TP2 local `N=928` |

The Q3 per-expert crossover can provide an initial estimate for another shape:

\[
M_{\text{estimate}}
= r_{\text{Q3}}\frac{E}{T}.
\]

The estimate is then checked against the actual local `K`, `N`, routing
padding, and backend timings. Repeating the complete shape record on several
distinct models shows whether a simple rows-per-expert relation is stable
enough to initialize calibration or whether the lookup needs a richer
shape-time model.

A model with nonsymmetric expert tensor shapes falls outside the current
uniform grouped layout. In that case the existing W4A16 path remains
available.

## Python prototype and native endpoint

The Python implementation is the short-turnaround functional and performance
prototype. It provides a reference for:

- packed NVFP4 conversion;
- scale conversion;
- routing layout;
- grouped DeepGEMM invocation;
- sequential `W13`/`W2` arena reuse;
- runtime selection.

The eventual endpoint is a new selectable hybrid expert implementation whose
outer operation is native C++/CUDA, analogous to the dense hybrid operator.
The low branch invokes the existing grouped Marlin implementation. The high
branch launches conversion, quantization, routing support, grouped DeepGEMM,
activation, and reduction on the current stream.

DeepGEMM consumes its required grouped layout but does not create that layout.
The prototype currently creates it through vLLM's Python-launched Triton
`deepgemm_moe_permute` path. C++ PyTorch can dispatch registered PyTorch
operators, but a raw Python Triton launcher is not automatically callable as a
native registered operator.

A 218-line native DeepGEMM scatter adapter is retained as a fallback. It is
not used by the current minimal route while the existing registered/native
composition remains sufficient.

`torch.cond` was also tested. The conditional itself can survive vLLM
compilation, but the current MoE boundary contains an opaque operation,
mutable workspaces, and a raw pybind component. Those features make
`torch.cond` a poor fit for this path. An ordinary runtime `M` decision at the
existing opaque `moe_forward` boundary is the smaller prototype and remains
dynamic.

## Current implementation locations

The current work is concentrated in:

- `vllm/model_executor/layers/fused_moe/experts/nvfp4_bycopy_moe.py`;
- `tests/kernels/moe/test_nvfp4_bycopy_moe.py`;
- `tests/kernels/moe/test_deepgemm.py`;
- `csrc/libtorch_stable/moe/moe_ops.h`;
- `csrc/libtorch_stable/moe/moe_permute_unpermute_op.cu`;
- `csrc/libtorch_stable/moe/torch_bindings.cpp`;
- `csrc/deepgemm_torch_bindings.cpp`;
- `tools/build_deepgemm_C.py`;
- `cmake/external_projects/deepgemm.cmake`.

The shared NVFP4-to-FP8 conversion implementation lives with the dense
converter and supports the expert dimension.

## Remaining measurements and decisions

The unresolved evidence and implementation work is:

1. Q3 fine-curve results around 4080, 4096, and 4112, exposing the 256-to-384
   DeepGEMM padding boundary.
2. Q3 GSM8K B/C results at selector cutoffs 4608 and 4096.
3. Real per-layer expert distributions and `P/R` efficiency from the
   actual-model workloads.
4. Complete shape records for three or more additional MoE shapes.
5. Cross-shape comparison of the first-order `M = rE/T` estimate and measured
   crossover.
6. Separate interpretation of local positive gain and the overall 20–40%
   model-throughput objective.
7. A native selectable expert operation using the proven Python sequence as
   its behavioral reference.
8. Dense router-gate treatment considered independently from the MoE expert
   path.

The established result is already stronger than functional proof: on Q3, the
MoE-only hybrid improved actual-model throughput by 17.355%, and the complete
dense-plus-MoE configuration improved it by 23.494%. Q36M also improved, but
its 9.468% MoE-only result and missing route distribution show why the
selector and shape model need evidence beyond one balanced Q3 curve.
