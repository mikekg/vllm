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
- `E`: global logical expert count before expert parallel sharding.
- `E_local`: experts represented by the tensors on one rank.
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

The knee mapping uses global logical `E`, including under expert parallelism.
The backend still operates on the rank-local expert stack, so `E_local` and
the concrete tensor dimensions determine the work and workspace on that rank.

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

The model tree retains the semantic relationship. `FusedMoEFactory` passes
the actual gate module into `MoERunner`, and `MoERunner` stores that object as
`runner.gate`. Commit `42551d2aaa` added `_mark_moe_router_gates(model)` to the
common post-load path. After per-layer and model-level weight finalization, it
walks `model.modules()` and marks each non-null `MoERunner.gate` object:

```python
for module in model.modules():
    if isinstance(module, MoERunner) and module.gate is not None:
        module.gate._vllm_is_moe_router = True
```

`MarlinNvFp4ToFp8LinearKernel.apply_weights()` reads that static marker before
entering the dense hybrid operation:

```python
if getattr(layer, "_vllm_is_moe_router", False):
    return marlin.apply_weights(layer, x, bias)
return hybrid_path(...)
```

The traversal runs before the first compilation, so Dynamo can specialize the
marker while ordinary dense layers retain their runtime-M choice. It does not
change the packed weights. It also avoids name, prefix, and shape matching:
the exact module object already held by the runner is the identity source.

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

The measured boundary is the first value above 256 rows/expert. In that
otherwise identical case it maps to `floor(256*16/4)+1 = 1025`.

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
&+ T_{\text{activation},4} \\
&+ T_{\text{reduce},4}.
\end{aligned}
\]

The transient FP8 route can be represented as

\[
\begin{aligned}
T_8(\{r_i\}) ={}&
C_{13}(E,K,2N) \\
&+ C_2(E,N,K) \\
&+ T_{\text{A1 quant}} \\
&+ T_{\text{permute}} \\
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

## DeepGEMM row rounding and route capacity

DeepGEMM aligns each expert's real routed count independently:

\[
p_i = A\left\lceil\frac{r_i}{A}\right\rceil,
\qquad P = \sum_i p_i,
\qquad A = 128.
\]

The executed padding efficiency is `R/P`, where `R=sum_i(r_i)`. Two calls with
the same `M`, `E`, and `T` can therefore have different FP8 costs when their
expert distributions differ.

The Q3 r13 harness reported a different quantity: the conservative route
workspace capacity used when CPU expert counts are unavailable:

\[
C = \operatorname{round\_up}
\left(R + \min(R,E)(A-1), A\right).
\]

All Q3 points had `R=M*T>E`, so `C=R+16,256`. The resulting capacity values
are not actual padded GEMM rows and must not be used as a serving histogram.

| Input `M` | Balanced mean rows/expert | Real routes `R` | Workspace capacity `C` | `R/C` |
|---:|---:|---:|---:|---:|
| 3,072 | 192 | 24,576 | 40,832 | 60.19% |
| 3,584 | 224 | 28,672 | 44,928 | 63.82% |
| 4,080 | 255 | 32,640 | 48,896 | 66.75% |
| 4,096 | 256 | 32,768 | 49,024 | 66.84% |
| 4,112 | 257 | 32,896 | 49,152 | 66.93% |
| 4,352 | 272 | 34,816 | 51,072 | 68.17% |
| 4,608 | 288 | 36,864 | 53,120 | 69.40% |
| 4,864 | 304 | 38,912 | 55,168 | 70.53% |
| 5,120 | 320 | 40,960 | 57,216 | 71.59% |
| 6,144 | 384 | 49,152 | 65,408 | 75.15% |
| 7,168 | 448 | 57,344 | 73,600 | 77.91% |
| 8,192 | 512 | 65,536 | 81,792 | 80.13% |

The cyclic synthetic route makes `M*T/E` the real count for every expert in
this controlled run. Production needs its actual per-expert counts to derive
`P`, padding efficiency, and min/median/max residency. The sharp measured
256-to-257 transition below comes from the W4A16 Marlin row-tile boundary; it
is not evidence that the conservative capacity `C` was executed.

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

Run `nvfp4-moe-deepgemm-q3-curve-r13` retained 15 raw CUDA-event samples for
the hybrid leg and each surrounding Marlin leg. The Marlin median uses all 30
A samples. These are the authoritative sequential-arena results:

| Input `M` | Mean rows/expert | Marlin ms | Hybrid ms | Time saved | Throughput-equivalent gain |
|---:|---:|---:|---:|---:|---:|
| 3,072 | 192 | 1.086416 | 1.330176 | -22.44% | -18.33% |
| 3,584 | 224 | 1.407600 | 1.391040 | +1.18% | +1.19% |
| 4,080 | 255 | 1.431488 | 1.435808 | -0.30% | -0.30% |
| 4,096 | 256 | 1.433408 | 1.438112 | -0.33% | -0.33% |
| 4,112 | 257 | 1.719280 | 1.459936 | +15.08% | +17.76% |
| 4,352 | 272 | 1.732736 | 1.475808 | +14.83% | +17.41% |
| 4,608 | 288 | 1.743616 | 1.500832 | +13.92% | +16.18% |
| 4,864 | 304 | 1.758752 | 1.519520 | +13.60% | +15.74% |
| 5,120 | 320 | 1.772464 | 1.548256 | +12.65% | +14.48% |
| 6,144 | 384 | 2.115040 | 1.652928 | +21.85% | +27.96% |
| 7,168 | 448 | 2.455040 | 1.729984 | +29.53% | +41.91% |
| 8,192 | 512 | 2.796240 | 1.842176 | +34.12% | +51.79% |

The curve is a staircase, not the earlier interpolated hyperbola. From 256 to
257 rows/expert, hybrid latency rises 1.52% while Marlin rises 19.94%; the
throughput-equivalent result moves from a practical tie to +17.76%. The gain
then declines within the Marlin tile bucket and rises at its next boundary.

Median component times were:

| M | W13 convert | A1 quant+route | W13 GEMM | SiLU+A2 quant | W2 convert | W2 GEMM | Unpermute+reduce |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3,072 | 0.468032 | 0.093408 | 0.250080 | 0.062368 | 0.254816 | 0.150176 | 0.056000 |
| 3,584 | 0.476576 | 0.107552 | 0.272416 | 0.068000 | 0.260256 | 0.161920 | 0.064864 |
| 4,080 | 0.476448 | 0.118912 | 0.286176 | 0.073408 | 0.260192 | 0.170432 | 0.069472 |
| 4,096 | 0.476896 | 0.119232 | 0.285792 | 0.073760 | 0.259840 | 0.170336 | 0.069600 |
| 4,112 | 0.477120 | 0.119968 | 0.300160 | 0.073888 | 0.259712 | 0.178560 | 0.072256 |
| 4,352 | 0.475648 | 0.124736 | 0.298016 | 0.076032 | 0.259552 | 0.181504 | 0.075648 |
| 4,608 | 0.475744 | 0.130560 | 0.309632 | 0.079008 | 0.259744 | 0.185152 | 0.078048 |
| 4,864 | 0.475776 | 0.136384 | 0.312384 | 0.081664 | 0.259712 | 0.190240 | 0.080704 |
| 5,120 | 0.475552 | 0.143392 | 0.328864 | 0.084288 | 0.259616 | 0.194272 | 0.083424 |
| 6,144 | 0.475328 | 0.167520 | 0.362720 | 0.095296 | 0.259488 | 0.215456 | 0.096960 |
| 7,168 | 0.476256 | 0.191328 | 0.368608 | 0.105792 | 0.259744 | 0.238464 | 0.111488 |
| 8,192 | 0.475648 | 0.214752 | 0.408128 | 0.116416 | 0.260032 | 0.262368 | 0.125600 |

The two conversions remain nearly constant at about 0.736 ms combined. The
activation, routing, GEMM, and reduction terms grow with M. Component medians
are independent stage measurements and do not algebraically sum to the median
complete call. Raw rows and plots are under
`.benchmark/cudagym-nvfp4-moe-deepgemm-r13-20260818/`.

## Q36 backend comparison

Q36 has `E=256`, `T=8`, `K=2048`, and `N=512`, so 256 and 257 mean routed
rows/expert occur at `M=8192` and `M=8224`. Two matched A/B/A curves measured
the complete route: one invoked staged DeepGEMM directly, while the other used
the production staged-Triton implementation and its normal config selection.

| M | Rows/expert | Marlin ms (DG run) | Direct DG ms | DG gain | Marlin ms (Triton run) | Production Triton ms | Triton gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 128 | 1.054048 | 1.778816 | -40.74% | 1.114080 | 1.707744 | -34.76% |
| 5,120 | 160 | 1.509728 | 1.804000 | -16.31% | 1.540320 | 1.896192 | -18.77% |
| 6,144 | 192 | 1.555632 | 1.881920 | -17.34% | 1.596464 | 1.982272 | -19.46% |
| 7,168 | 224 | 1.994944 | 1.968096 | +1.36% | 2.025296 | 2.191648 | -7.59% |
| 8,160 | 255 | 2.037440 | 2.053952 | -0.80% | 2.065488 | 2.264416 | -8.78% |
| 8,192 | 256 | 2.041632 | 2.056832 | -0.74% | 2.066144 | 2.260608 | -8.60% |
| 8,224 | 257 | 2.453888 | 2.074112 | +18.31% | 2.470480 | 2.424000 | +1.92% |
| 9,216 | 288 | 2.492704 | 2.146080 | +16.15% | 2.508880 | 2.481856 | +1.09% |
| 10,240 | 320 | 2.534848 | 2.230944 | +13.62% | 2.544352 | 2.547328 | -0.12% |
| 12,288 | 384 | 3.030864 | 2.419008 | +25.29% | 3.036320 | 2.843520 | +6.78% |
| 14,336 | 448 | 3.522080 | 2.550880 | +38.07% | 3.523632 | 3.133504 | +12.45% |
| 16,384 | 512 | 4.008336 | 2.754432 | +45.52% | 4.010896 | 3.436736 | +16.71% |

Both alternatives expose the same 256-to-257 discontinuity because Marlin,
not the FP8 backends, jumps at that row-tile boundary. Direct DeepGEMM moves
from -0.74% to +18.31%; staged Triton moves from -8.60% to +1.92%. The Triton
run also logged that the tuned E256/N512 H100 FP8 JSON was absent, so its normal
selector used vLLM's default MoE config. Raw results are under
`.benchmark/cudagym-nvfp4-moe-q36-curve-r2-20260818/` and
`.benchmark/cudagym-nvfp4-moe-q36-triton-curve-r2-20260818/`.

### Production staged-Triton CUDA-graph curves

Two later runs forced the production high-M branch while setting
`VLLM_USE_DEEP_GEMM=0`, so the hybrid measurements below include NVFP4 weight
conversion and the production staged-Triton MoE route. Each backend received
three eager warmups; one graph captured 10 calls; five graph replays warmed the
capture; and the reported latency is the median of 30 CUDA-event replay
samples divided by 10. This was one Marlin graph followed by one hybrid graph,
not an A/B/A series, and the retained data do not contain raw samples or error
bars.

The Q3 balanced cyclic curve was:

| M | Mean rows/expert | Marlin ms | Hybrid ms | Marlin / hybrid - 1 |
|---:|---:|---:|---:|---:|
| 3,072 | 192 | 1.074907 | 0.993976 | +8.142% |
| 3,584 | 224 | 1.385997 | 1.140187 | +21.559% |
| 4,080 | 255 | 1.405494 | 1.181037 | +19.005% |
| 4,096 | 256 | 1.424488 | 1.182646 | +20.449% |
| 4,112 | 257 | 1.683333 | 1.297443 | +29.742% |
| 4,352 | 272 | 1.718410 | 1.311562 | +31.020% |
| 4,608 | 288 | 1.736202 | 1.330493 | +30.493% |
| 4,864 | 304 | 1.751750 | 1.349811 | +29.777% |
| 5,120 | 320 | 1.765330 | 1.371443 | +28.721% |
| 6,144 | 384 | 2.108510 | 1.565053 | +34.725% |
| 7,168 | 448 | 2.444494 | 1.755896 | +39.216% |
| 8,192 | 512 | 2.787758 | 1.948053 | +43.105% |

Q36 balanced routing cycled over all 256 experts:

| M | Active experts | Mean rows/expert | Marlin ms | Hybrid ms | Marlin / hybrid - 1 |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 256 | 128 | 1.048419 | 1.130374 | -7.250% |
| 5,120 | 256 | 160 | 1.505234 | 1.338438 | +12.462% |
| 6,144 | 256 | 192 | 1.559254 | 1.403866 | +11.069% |
| 7,168 | 256 | 224 | 2.002179 | 1.622056 | +23.435% |
| 8,160 | 256 | 255 | 2.044622 | 1.686422 | +21.240% |
| 8,192 | 256 | 256 | 2.045811 | 1.689619 | +21.081% |
| 8,193 | 256 | 256.03125 | 2.058576 | 1.694854 | +21.460% |
| 8,224 | 256 | 257 | 2.437675 | 1.839550 | +32.515% |
| 9,216 | 256 | 288 | 2.482307 | 1.896320 | +30.901% |
| 10,240 | 256 | 320 | 2.526370 | 1.959565 | +28.925% |
| 12,288 | 256 | 384 | 3.014123 | 2.245864 | +34.208% |
| 14,336 | 256 | 448 | 3.505235 | 2.528990 | +38.602% |
| 16,384 | 256 | 512 | 3.994112 | 2.829040 | +41.183% |

Q36 half routing cycled over experts 0 through 127:

| M | Active experts | Mean rows/active expert | Marlin ms | Hybrid ms | Marlin / hybrid - 1 |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 128 | 256 | 1.043522 | 1.070299 | -2.502% |
| 5,120 | 128 | 320 | 1.280973 | 1.223163 | +4.726% |
| 6,144 | 128 | 384 | 1.536710 | 1.368163 | +12.319% |
| 7,168 | 128 | 448 | 1.795974 | 1.508438 | +19.062% |
| 8,160 | 128 | 510 | 2.034394 | 1.660363 | +22.527% |
| 8,192 | 128 | 512 | 2.036589 | 1.662685 | +22.488% |
| 8,193 | 128 | 512.0625 | 2.051434 | 1.667142 | +23.051% |
| 8,224 | 128 | 514 | 2.228566 | 1.728426 | +28.936% |
| 9,216 | 128 | 576 | 2.271157 | 1.788574 | +26.981% |
| 10,240 | 128 | 640 | 2.516477 | 1.932982 | +30.186% |
| 12,288 | 128 | 768 | 3.006238 | 2.217421 | +35.574% |
| 14,336 | 128 | 896 | 3.498912 | 2.492550 | +40.375% |
| 16,384 | 128 | 1,024 | 3.986341 | 2.784693 | +43.152% |

The Q36 formula knee is
`floor(256 * 256 / 8) + 1 = 8,193`. At `M=8,192`, balanced routing has exactly
256 rows per expert. At `M=8,193`, the cyclic route gives eight experts 257
rows and the other 248 experts 256 rows, so the selector changes branch at the
first input that cannot keep every expert at or below 256. The fully balanced
257-row point is `M=8,224`, where the Marlin step raises the measured hybrid
gain from +21.460% at 8,193 to +32.515%.

This graph curve also shows that 8,193 is a conservative shape rule, not a
best-fit Q36 crossover: balanced staged Triton was already faster at the
sampled `M=5,120`, and the concentrated half route crossed between 4,096 and
5,120. The selector intentionally uses global `E` rather than an observed
active-expert count, so both routing patterns retain the same model-independent
8,193 boundary. Likewise, the Q3 formula boundary is 4,097 even though this
particular graph run measured a +20.449% hybrid gain at 4,096. That result
supersedes any general claim that choosing either backend at the adjacent Q3
point is necessarily inconsequential.

The run emitted `max_abs`, relative-L2, and cosine comparisons against Marlin
for every point, but it applied no numerical acceptance threshold. Those
fields are recorded as comparison data rather than described as a correctness
pass. Exact JSONL, hashes, source paths, and the plotting program are under
`results/moe-production-curves/`.

![Q3 and Q36 production MoE curves](results/moe-production-curves/q3-q36-r10-r9.svg)

*Production staged-Triton versus W4A16 Marlin under CUDA-graph replay. The top
axes show the arithmetic mean for the synthetic cyclic route, not a measured
serving histogram. Q36 half routing averages only across its 128 active
experts. Lines connect sampled points and do not imply interpolation; no error
bars are available from these retained medians.*

## Runtime selector

The Q3 and Q36 cliffs agree on the first input M whose mean routed residency
exceeds 256 rows/expert:

\[
M_{\text{knee}}
= \left\lfloor 256\frac{E}{T}\right\rfloor + 1.
\]

The outer hybrid uses the concrete, post-dispatch `M` on every invocation and
selects FP8 when `M >= M_knee`; Q3 therefore uses 4097 and Q36 uses 8193. This
rule depends on global logical expert count and top-k, not a model name or a
fitted layer identity. The earlier sequential DeepGEMM curve differed by about
0.3% at Q3's adjacent 256-row point; the later production staged-Triton graph
curve measured +20.449% there, so regret at the boundary is backend- and
protocol-dependent rather than a property of the formula.

After the outer M decision, the FP8 implementation is selected from the actual
local `K` and `N`, activation, and available backend. SILU shapes with
DeepGEMM support, `N >= 512`, and valid `(M,2N,K)` and `(M,K,N)` contracts use
DeepGEMM; other supported shapes use staged Triton. Thus the knee is generic,
while K/N tiling and backend capability remain explicit per-layer constraints.

Commit `47b1ab3960` made those quantities concrete in parallel execution:

- `_moe_shape()` reads `num_logical_experts`, rather than the rank-local expert
  count, when constructing the knee;
- the inner backend selector derives `N` from the packed `W13`/`W2` tensors and
  `K` from the runtime hidden states;
- ordinary tensor parallel execution is accepted when expert parallelism is
  off, and pure expert parallel execution is accepted when `TP=1` and `EP>1`;
- configurations with `dp_size>1`, `pcp_size>1`, `sp_size>1`, or EPLB continue
  through the existing backend selection instead of this hybrid;
- when pure EP supplies an `expert_map`, staged Triton clears its second-GEMM
  output workspace before use so non-local expert slots cannot retain data
  from an earlier invocation.

The commit also changed the DeepGEMM intermediate-size boundary from `N>512`
to `N>=512`, which admits the measured Q36 shape.

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

### Q3, 8k/1k concurrency and knee matrix

The complete serving matrix used one H100, fixed 32 GiB KV-cache admission,
8,192 input tokens and 1,024 output tokens per request. The baseline disabled
the dense hybrid and selected Marlin MoE. Every knee row enabled the full dense
and MoE hybrid with that diagnostic MoE threshold. Concurrency order is:

```text
[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
```

The exact output-token/s arrays are:

```text
W4A16 job 6646233:
[199.300015, 347.278057, 568.426165, 838.895570, 1157.092413,
 1508.158562, 1779.984815, 1617.086187, 1594.011694, 1623.589042]
knee 3072 job 6646239:
[203.571913, 360.339807, 603.203072, 909.242226, 1341.096388,
 1819.519932, 2220.580526, 2009.301421, 1984.932100, 2037.699665]
knee 3584 job 6646241:
[203.819969, 360.532472, 599.044728, 916.983085, 1335.026937,
 1853.018511, 2244.733880, 2026.334598, 2007.346734, 2055.172602]
knee 4080 job 6646243:
[206.232007, 364.215119, 605.318110, 927.790704, 1341.226028,
 1851.876648, 2253.426611, 2037.431969, 2005.274704, 2053.644926]
knee 4096 job 6646245:
[202.846058, 359.300076, 596.518149, 907.421455, 1322.155851,
 1804.333735, 2214.849633, 2002.675603, 1975.261083, 2020.854997]
knee 4112 job 6646247:
[202.907348, 358.974351, 598.819935, 920.222982, 1337.762501,
 1821.973775, 2231.659113, 2021.225857, 1989.029628, 2057.847477]
knee 4352 job 6646249:
[203.580388, 358.187742, 593.031497, 906.620932, 1320.241817,
 1801.296555, 2212.888130, 2005.164577, 1971.256078, 2025.325156]
knee 4608 job 6646251:
[203.913800, 359.548302, 597.838163, 917.623157, 1323.785976,
 1818.241851, 2226.455411, 2009.475529, 1979.803861, 2054.389594]
knee 4864 job 6646253:
[204.147117, 359.801743, 600.276320, 917.759684, 1335.393773,
 1811.563030, 2228.195648, 2009.717447, 1981.739739, 2034.612285]
knee 5120 job 6646255:
[216.712321, 380.353333, 628.424678, 951.631253, 1373.405459,
 1848.237521, 2249.728186, 2035.284215, 2008.412823, 2055.366074]
knee 6144 job 6646257:
[202.971293, 359.321404, 598.660085, 915.237450, 1334.690906,
 1813.909687, 2218.875058, 2004.147631, 1977.776486, 2026.666974]
knee 7168 job 6646259:
[203.219264, 359.951061, 596.146654, 912.171815, 1332.255216,
 1811.506773, 2215.699972, 2007.223221, 1983.070458, 2031.811945]
knee 8192 job 6646261:
[203.283096, 357.422759, 595.950725, 909.918884, 1321.648310,
 1812.649306, 2226.080772, 2003.338372, 1979.572618, 2029.452330]
```

All 130 concurrency points completed with zero nonempty benchmark errors. The
matrix records the whole-model response to a knee, while the controlled curves
above measure one MoE operation at a specified runtime M; they are related but
not interchangeable measurements.

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

### Q36M, derived-knee concurrency ladder

Jobs 6646398 through 6646400 used the derived `M_knee=8193` and completed the
8k/1k powers-of-two ladder. The hybrid route in this series used the production
staged-Triton FP8 implementation. Concurrency order is:

```text
[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
```

Exact output-token/s arrays are:

```text
A: Marlin dense + Marlin MoE, job 6646398:
[220.0432666, 404.0652744, 696.9073713, 1062.8874600, 1510.8721746,
 2042.1104185, 2506.7712234, 2914.7652113, 2752.3690089, 2870.5154107]
B: Marlin dense + hybrid MoE, job 6646399:
[221.1562262, 432.2056645, 754.0502025, 1145.9857638, 1624.4314987,
 2175.5886607, 2692.8749164, 3182.4137426, 3024.7278640, 3155.5385613]
C: hybrid dense + hybrid MoE, job 6646400:
[220.6201431, 404.1341633, 702.6082464, 1101.3417264, 1582.8796484,
 2158.6717285, 2704.2783047, 3222.0570617, 3050.9239561, 3199.3768603]
```

All 30 concurrency points completed with zero nonempty benchmark errors.

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
result points away from the MoE conversion itself as the source of the C
regression. These runs preceded commit `42551d2aaa`; the exact-object marker
described above now keeps the router gate on Marlin because small changes in
its logits can alter discrete top-k choices.

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

The later serialized full-hybrid knee matrix uses W4A16 baseline job 6646346
and retains per-question details for every candidate. These are the completed
points:

| MoE knee | Job | Correct / 1,319 | Accuracy | Output tok/s | TPS delta | `n01` improvements | `n10` regressions | Accuracy delta | Exact McNemar p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| — | 6646346 | 1,164 | 0.8824867 | 4,236.742 | — | — | — | — | — |
| 3,072 | 6646351 | 1,163 | 0.8817286 | 4,100.469 | -3.216% | 22 | 23 | -0.075815 pp | 1.0000000 |
| 3,584 | 6646353 | 1,158 | 0.8779378 | 4,136.298 | -2.371% | 17 | 23 | -0.454890 pp | 0.4295905 |
| 4,080 | 6646355 | 1,155 | 0.8756634 | 3,997.598 | -5.645% | 16 | 25 | -0.682335 pp | 0.2110236 |
| 4,096 | 6646357 | 1,168 | 0.8855193 | 4,016.347 | -5.202% | 22 | 18 | +0.303260 pp | 0.6358280 |
| 4,112 | 6646359 | 1,163 | 0.8817286 | 4,075.061 | -3.816% | 14 | 15 | -0.075815 pp | 1.0000000 |
| 4,352 | 6646361 | 1,167 | 0.8847612 | 4,155.321 | -1.922% | 21 | 18 | +0.227445 pp | 0.7492586 |
| 4,608 | 6646363 | 1,162 | 0.8809704 | 4,051.695 | -4.368% | 20 | 22 | -0.151630 pp | 0.8776143 |
| 4,864 | 6646365 | 1,156 | 0.8764215 | 4,017.315 | -5.179% | 19 | 27 | -0.606520 pp | 0.3019956 |
| 5,120 | 6646367 | 1,162 | 0.8809704 | 3,959.391 | -6.546% | 17 | 19 | -0.151630 pp | 0.8679394 |
| 6,144 | 6646369 | 1,167 | 0.8847612 | 4,066.142 | -4.027% | 19 | 16 | +0.227445 pp | 0.7358788 |
| 7,168 | 6646371 | 1,161 | 0.8802123 | 4,048.157 | -4.451% | 19 | 22 | -0.227445 pp | 0.7552287 |
| 8,192 | 6646373 | 1,152 | 0.8733889 | 4,123.041 | -2.684% | 13 | 25 | -0.909780 pp | 0.0729514 |

Here `n01` means baseline wrong and candidate right; `n10` means baseline
right and candidate wrong. All 12 candidate comparisons have two-sided paired
p-values above 0.05; knee 8192 is closest at 0.0729514. These candidates were
built before commit `42551d2aaa`, so their dense hybrid could still process the
router gate. They record that earlier execution path; router-guard measurements
use the same paired format as a separate comparison. Evaluator TPS includes
different generated-token counts and is not a matched-token serving result.
Artifacts are under
`.benchmark/gsm8k-q3-knee-matrix-q3-knees-20260818-r2/`.

The post-router-guard A/B/C repetition used the derived `M_knee=4097` and
included both `42551d2aaa` and `47b1ab3960`:

| Variant | Job | Correct / 1,319 | Accuracy fraction | Eval s | Output tok/s | `n01` improvements | `n10` regressions | Accuracy delta | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: Marlin dense + Marlin MoE | 6646885 | 1,157 | 0.87717968 | 46.2953 | 4,225.395 | -- | -- | -- | -- |
| B: Marlin dense + hybrid MoE | 6646889 | 1,162 | 0.88097043 | 47.6191 | 4,100.368 | 17 | 12 | +0.379075 pp | 0.4582583 |
| C: hybrid dense + hybrid MoE | 6646896 | 1,159 | 0.87869598 | 49.4624 | 3,956.847 | 20 | 18 | +0.151630 pp | 0.8714147 |

Neither hybrid variant has a statistically significant paired accuracy change
against A in this repetition. Artifacts are under
`.benchmark/gsm8k-q3-current-formula-current-formula-20260818-r1/`.

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
expects one common expert tensor shape. The knee uses the global logical expert
count and top-k. Backend capability and workspace use the concrete rank-local
weight shapes after tensor or expert parallel partitioning.

The initial validation inventory includes:

| Model shape | Global logical experts `E` | Top-k `T` | Hidden `K` | Intermediate `N` | Parallel note |
|---|---:|---:|---:|---:|---|
| Q3 | 128 | 8 | 2048 | 768 | measured |
| Q36M | 256 | 8 | 2048 | 512 | actual-model A/B/C measured |
| Gemma 4 candidate | 128 | 8 | 2816 | 704 | TP4 local `N=176` |
| Nemotron Nano candidate | 128 | 6 | 2688 | 1856 | TP2 local `N=928` |

The runtime selector transfers the measured row boundary to another shape:

\[
M_{\text{knee}}
= \left\lfloor 256\frac{E}{T}\right\rfloor + 1.
\]

The per-invocation M decision is followed by capability selection using the
actual local `K`, `N`, activation, and backend contracts. Repeating the
complete shape record on additional models tests how broadly the 256-row
Marlin boundary holds without introducing model-specific cutoffs.

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

The production endpoint is a selectable hybrid expert implementation whose
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
- `vllm/model_executor/model_loader/utils.py`;
- `vllm/model_executor/kernels/linear/nvfp4/marlin_fp8.py`;
- `tests/kernels/moe/test_nvfp4_bycopy_moe.py`;
- `tests/kernels/moe/test_deepgemm.py`;
- `tests/quantization/test_modelopt.py`;
- `csrc/libtorch_stable/moe/moe_ops.h`;
- `csrc/libtorch_stable/moe/moe_permute_unpermute_op.cu`;
- `csrc/libtorch_stable/moe/torch_bindings.cpp`;
- `csrc/deepgemm_torch_bindings.cpp`;
- `tools/build_deepgemm_C.py`;
- `cmake/external_projects/deepgemm.cmake`.

The shared NVFP4-to-FP8 conversion implementation lives with the dense
converter and supports the expert dimension.

For router commit `42551d2aaa`, the three focused tests passed, the
non-checkpoint `test_modelopt.py` selection reported 29 passed and 3
deselected, and the pre-commit hooks passed. For topology commit `47b1ab3960`,
the focused MoE selection/workspace run reported 13 passed and 1 skipped, and
the pre-commit hooks passed. The tests cover exact-object gate marking, Marlin
router dispatch, global-E knees under local expert sharding, TP and pure-EP
support, the `N=496/512` backend boundary, runtime K/N extraction, and clearing
remote EP slots before the second staged-Triton GEMM.

## Final ABI-matched validation artifact

OCI-NRT build job 6166032 produced
`native-overlay-9340d68ade-r1` from source `9340d68ade`. The artifact targets
the serving container's PyTorch 2.11.0+cu130 and vLLM 0.24.0 ABI. Its manifest
records base stable-ABI DSO hash prefix `7d723b` and extension hash prefix
`ffa52a`.

H100 smoke job 6166055 loaded that overlay through the container's bind-mounted
paths and reported 3 passed in 12.31 seconds. The cases exercised the native
hybrid operation, an `M=1` call below `M_knee=2` that selected Marlin, and an
`M=3` call above `M_knee=1` that selected padded FP8.

## Native outer-operation numerical validation

Commit `4f577ce2aa` made packaged DeepGEMM NVRTC compilation self-contained by
staging its patched runtime headers, using CUDA standard-library utilities in
the JIT-visible header, and installing the exact patched `utils.cuh` consumed
after packaging. Commit `c26551b19a` then matched the native high-M composition
to the Python DeepGEMM prototype. The fused SiLU-plus-quantization call was
replaced by the existing `_C::silu_and_mul` operation followed by
`_C::per_token_group_fp8_quant`. The shared arena was enlarged when necessary
to hold that BF16 activation result; after quantization, `W2` conversion reuses
the same storage.

The numerical investigation used the existing DeepGEMM whole-tensor metric

\[
d(x,y)
= 1 - \frac{2\sum_j x_jy_j}{\sum_j (x_j^2+y_j^2)}
= \frac{\lVert x-y\rVert_2^2}
       {\lVert x\rVert_2^2+\lVert y\rVert_2^2}.
\]

Identical tensors give zero; smaller values mean closer aggregate agreement.
The existing DeepGEMM unit-test boundary is `d < 0.001`. The first high-M
native-versus-Marlin comparison measured `0.001113852751636557`, or about
`0.111%` when the metric itself is expressed as a percentage. When the two
output norms are similar, the corresponding relative L2 difference is
approximately `sqrt(2d)`, or `4.72%`. This is valid replacement-consistency
evidence: it quantifies the output drift between the current W4A16 Marlin path
and the proposed FP8 DeepGEMM path. It is not an isolated oracle for the native
wrapper's implementation fidelity because it also includes the arithmetic,
conversion, quantization, and rounding differences between the two backends.
The implementation-fidelity reference is the Python implementation of the
same DeepGEMM sequence.

After the two-operation activation correction, another comparison with Marlin
measured `0.0011665152362726472`. This second value records the numerical drift
between the W4A16 Marlin and FP8 DeepGEMM backends and provides another
replacement-consistency measurement; it does not isolate native composition
correctness. The `0.001` threshold was not relaxed. These whole-tensor metric
values are retained alongside `max_abs`, relative-L2, and cosine comparisons;
end-to-end output accuracy is evaluated separately with paired GSM8K outcomes
and an exact McNemar test.

The final branch-aware test compares the low-M branch with Marlin and the
high-M branch with the Python DeepGEMM prototype. H100 job `6170209` exercised
the Cartesian product of below/at-knee selection, local/expert-parallel
routing, and router-weight-on-input disabled/enabled. In every case the output
also began at the same data pointer as the shared arena. The job compiled three
SM90a DeepGEMM kernels through NVRTC, passed all eight combinations, and emitted
`native-low-high-ep-router-alias-ok`.

## Unseen-model validation campaign

Campaign `unseen-9340d68ade-20260818-r1` uses the tracked model matrix at the
same source commit. Its 11 cached model entries are `nano30`, `super120`,
`ultra550`, `deepseek_r1`, `deepseek_v4_flash`, `deepseek_v4_pro`, `qwen397`,
`qwen36_35b`, `gemma4_26b`, `nano4_dense`, and `nano30_omni`.

The performance graph contains 88 GPU services: 11 models, two variants, and
the 1k/1k, 5k/1k, 8k/1k, and 50k/1k workloads. Every service runs the complete
power-of-two concurrency ladder from 1 through 512. The 77 intervening CPU
jobs provide five-minute shutdown separation without holding GPUs. The GSM8K
graph contains 22 GPU services and 11 equivalent CPU gaps; each model retains
all 1,319 per-question outcomes for an exact paired McNemar comparison.

The native-reference variant disables only
`MarlinNvFp4ToFp8LinearKernel` and `NvFp4ByCopyExperts`. The adaptive variant
keeps the per-invocation M selector, the global-expert/top-k knee formula, and
the exact-object router guard. Native NVFP4 W4A4 and non-NVFP4 paths are
unchanged in both variants. The controller accepted the campaign as job IDs
6166687 through 6166885.

## Remaining measurements and decisions

The unresolved evidence and implementation work is:

1. Real per-layer expert distributions and `P/R` efficiency from the
   actual-model workloads.
2. Complete shape records for three or more additional MoE shapes.
3. Cross-shape comparison of the generic 256-row rule and measured
   crossover.
4. Results from the submitted full ladders on the additional model shapes.
5. Separate interpretation of local positive gain and the overall 20–40%
   model-throughput objective.

The established result is already stronger than functional proof: on Q3, the
MoE-only hybrid improved actual-model throughput by 17.355%, and the complete
dense-plus-MoE configuration improved it by 23.494%. Q36M's operator curve
independently reproduces the 256-to-257 Marlin cliff, while its 9.468%
actual-model MoE-only result and missing route distribution leave the
operator-to-serving mapping to be measured.
