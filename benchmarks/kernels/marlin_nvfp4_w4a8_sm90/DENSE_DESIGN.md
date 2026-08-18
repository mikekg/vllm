# Runtime-Selective Dense NVFP4-to-FP8 Design

<!-- markdownlint-disable MD060 -->

## Purpose

This is the standalone design for dense NVFP4 linear layers. It records the
architecture, cost equations, exact matched measurements, correctness and
memory observations, alternative designs, rejected paths, and unresolved
decisions.
Chronological evidence remains in [EXPERIMENTS.md](EXPERIMENTS.md). The
separable MoE design is in [MOE_DESIGN.md](MOE_DESIGN.md).

The design is experimental. Packed NVFP4 remains the only resident weight
representation. The measurements below keep throughput, memory, and accuracy
as separate observed properties.

## Objective and invariants

Every eligible dense invocation chooses between:

1. the existing packed-NVFP4 W4A16 Marlin GEMM; and
2. transient activation quantization, exact NVFP4-to-FP8 conversion, and a
   block-scaled FP8 CUTLASS GEMM.

The current scope is:

- end-to-end throughput above W4A16, broadly targeting 20--40%;
- selection from hardware, dtype, tensor shape, and runtime M rather than
  model names;
- no persistent converted FP8 weight;
- matched fixed-KV admission, CUDA-graph memory, and peak HBM;
- paired model-accuracy measurements;
- an independent decision for every layer invocation.

## Cost model

For

\[
Y_{M\times N}=X_{M\times K}W^\mathsf{T},
\qquad W\in\mathbb{R}^{N\times K},
\]

define

\[
T_4(M,N,K)=T_{\mathrm{Marlin}}(M,N,K)
\]

and

\[
T_8(M,N,K)=T_{Aq}(M,K)+T_{Wc}(N,K)+T_{FP8}(M,N,K)
            +T_{scratch}+T_{pad}.
\]

The terms are activation quantization, weight conversion and scale expansion,
FP8 GEMM, scratch/allocation cost, and padding cost. The selector measurements
include complete T8; converter-only or GEMM-only timing omits charged work.

Local throughput gain and time saved are reciprocal metrics:

\[
G_{local}=T_4/T_8-1,
\qquad
S_{local}=1-T_8/T_4.
\]

If fraction f of baseline end-to-end time is affected and accelerates by s,
the removable-time bound is

\[
G_{e2e}=\frac{1}{(1-f)+f/s}-1.
\]

This is why a 15% local gain can remain useful even when it is below the
project's 20% end-to-end objective.

The current first-order policy carries a scalar per-layer M knee:

\[
\mathrm{FP8}\ \mathrm{iff}\ M\ge M_{knee}.
\]

This is not assumed universal. A generic calibration may also key on stable
N, K, dtype, GPU, and backend facts. Model identity is not part of that
execution key.

Within one stable kernel-tile regime, a useful local approximation is

\[
T_4(M;s)\approx a_4(s)+b_4(s)M
\]

and

\[
T_8(M;s)\approx C_W(s)+a_8(s)+b_8(s)M.
\]

The corresponding first-order crossing is

\[
M_{knee}(s)\approx
\frac{C_W(s)+a_8(s)-a_4(s)}
     {b_4(s)-b_8(s)}.
\]

The approximately fixed conversion term explains why large prefill is the
favorable end of the range. Tile changes make both curves piecewise, so this
equation explains the direction rather than supplying a universal knee.

### Measured transient full-path fit

Before the current native boundary, jobs 6627480/6627481 and 6627494/6627495
measured 60 M values over 16 `(N,K)` shapes for each of fixed and rotating
targets. With `m=M/1024`, `n=N/1024`, `k=K/1024`, and
`delta=T_8-T_4` in microseconds, the pooled fit was

\[
\widehat\Delta=27.0321415293-1.11012762491nk+5.12906422841n
+0.252633354334k+m(0.835812553727k-2.75518223775n
-3.87707405808nk).
\]

Writing the constant and M coefficients as A(n,k) and B(n,k),

\[
\widehat\Delta=A(n,k)+(M/1024)B(n,k).
\]

The strict integer threshold for margin q and B<0 is

\[
M_{min}=\left\lfloor\frac{-1024(A+q)}{B}\right\rfloor+1.
\]

Shape-LOSO validation gave R2=0.9714992641, 6.170117 us median absolute error,
93.3854% sign accuracy, and TP/FP/TN/FN=629/59/1164/68 without a margin. The
empirical zero-false-selection margin was 13.673077 us, rounded upward to
q=14 us. It produced knees 23,064 for 512x512, 8,004 for 1024x1024, 659 for
4096x4096, and 3,317 for 4096x512; 128x4096 had no FP8 region.

That fit predates the native internal-allocation boundary and is not the
current dispatch rule. The current implementation uses m_knee=1, raised to
max_cudagraph_capture_size+1 when compilation configuration is present; the
constant-512 `_lookup_nvfp4_bycopy_m_knee` helper is not called by this class.

### Historical cached-weight fit

An earlier experiment converted dense weights once during loading and retained
the FP8 copy. Its latency fit, using n=N/1024, k=K/1024, and m=M/1024, was

\[
\widehat{\Delta} =
12.168719 + 0.557642n - 2.757411k + 0.641265nk
+m(-1.177660 - 1.305558n + 2.357256k - 4.587304nk).
\]

A scalar q margin was then applied to that fitted delta; q=15 microseconds was
the smallest integer without a loss in its measured table, while q=14 exposed
a losing (N,K,M)=(4096,512,2048) point. This is the “q14 formula” discussed in
the experiment history.

That fit measured a different design: a persistent converted-weight cache
whose model residency rose from 19.97 to 53.09 GiB. It excludes per-call
conversion and does not describe the current transient path. It is retained
here to distinguish the historical shape fit from the present runtime
full-cost calibration.

## Current implementation

The selectable kernel is MarlinNvFp4ToFp8LinearKernel in
vllm/model_executor/kernels/linear/nvfp4/marlin_fp8.py. It embeds the
established MarlinNvFp4LinearKernel.

Post-load processing:

1. performs normal Marlin packing and scale preparation;
2. retains packed NVFP4, processed E4M3 block scales, global scale,
   per-tile divisor codes, and the Marlin workspace;
3. records logical and physical N/K;
4. establishes static shape eligibility and the M policy;
5. creates no FP8 weight.

For an admitted layer, the resident and transient layouts are:

| Value | Dtype and physical shape |
|---|---|
| Packed weight | int32 [K/16, 2N] |
| Processed block scales | E4M3 [K/16, N] |
| Processed global scale | FP32 scalar |
| Tile divisor codes | uint8 [N/128, K/128] |
| Quantized activation | E4M3 [M4, K] |
| Activation-scale backing | FP32 [K/128, M4] |
| Activation-scale view | FP32 [M4, K/128], stride (1, M4) |
| Converted weight | E4M3 [N, K] |
| Converted weight scale | FP32 [N/128, K/128] |
| Output | input dtype [M4, N] |

The weight and scale tensors are transposed as zero-copy views for the
CUTLASS operand layout. The output is narrowed back to M rows.

### Exact tile-divisor construction

The divisor metadata is derived before Marlin repacking, while the checkpoint
weight is still packed as two E2M1 values per byte. For every 16-value scale
group, preprocessing strips the sign bit from both nibbles, takes the larger
magnitude code, maps codes 0--7 to

\[
(0,\tfrac12,1,\tfrac32,2,3,4,6),
\]

and multiplies that magnitude by the absolute checkpoint block scale. It pads
N to 128 rows and K/16 to eight scale groups, then takes the maximum over each
128-by-128 tile:

\[
a_{tile}=\max_{(n,k)\in tile}|q_{nk}|\,|s_{group(n,k)}|.
\]

With the Marlin scale normalization factor `scale_factor`, preprocessing forms

\[
u=2\,scale\_factor\,a_{tile}.
\]

It first rounds `u/7` to FP16, retains the FP16 exponent and top three
mantissa bits (`half_bits >> 7`), increments the code when its decoded value
times seven is still below `u`, then rounds the S0E5M3 mantissa upward to four
with an exponent carry when needed. An all-zero tile receives code `0x78`,
which decodes to FP16 1.0. One uint8 code is stored per 128-by-128 tile.

The CUDA converter decodes a processed scale byte and divisor code by shifting
each left seven bits into an FP16 bit pattern. It multiplies each unpacked
E2M1 pair by

\[
s_{normalized}=\min(s_{processed}/d_{tile},65504)
\]

and saturating-rounds the result to E4M3. The emitted FP32 tile scale is

\[
s_{tile}=s_{global,processed}\,d_{tile}\,
\begin{cases}2^{-14},&\text{FP16 resident},\\2^{-126},&\text{BF16 resident}.
\end{cases}
\]

The E4M3 byte times this FP32 scale is the converted representation consumed
by CUTLASS. The divisor changes where rounding occurs; it does not add a
second resident weight.

Runtime enters one functional native operator,
_C::marlin_nvfp4_hybrid_linear, implemented in
csrc/libtorch_stable/quantization/marlin/marlin_nvfp4_to_fp8.cu.

For M below the knee, the native operator directly calls the existing Marlin
GEMM. For M at or above the knee it:

1. makes input contiguous if needed;
2. pads M to a multiple of four;
3. allocates FP8 activation and column-major activation scales;
4. allocates transient FP8 weight and 128-by-128 FP32 weight scales;
5. quantizes activation in 128-element groups;
6. converts packed NVFP4 and its exact scale representation;
7. calls the existing block-scaled FP8 CUTLASS GEMM;
8. narrows output to logical M.

### Native ABI and exact branch sequence

Python flattens every leading activation dimension into `x_2d=[M,K]` and
passes nine values to `_C::marlin_nvfp4_hybrid_linear`: input, packed int32
weight, processed E4M3 scales, processed FP32 global scale, uint8 divisor
codes, Marlin int32 workspace, `m_knee`, `use_atomic_add`, and
`use_fp32_reduce`. The native entry checks rank, dtype, physical dimensions,
positive knee, and that every resident tensor is on the input CUDA device.

For `M<m_knee`, it immediately calls the existing E2M1 Marlin GEMM with
`is_k_full=true` and the two stored reduction flags. No activation, converted
weight, FP8 scale, or output scratch has been allocated at that point.

For `M>=m_knee`, the sequence is:

1. obtain a contiguous `[M,K]` input, aliasing the original when already
   contiguous;
2. compute `M4=4*ceil(M/4)`;
3. allocate output `[M4,N]` in the input dtype, activation `[M4,K]` in E4M3,
   activation-scale backing `[K/128,M4]` in FP32, converted weight `[N,K]` in
   E4M3, and weight scales `[N/128,K/128]` in FP32;
4. transpose the activation-scale backing to the column-major `[M4,K/128]`
   view, then quantize the logical first M rows in groups of 128 with epsilon
   `1e-10`, saturation range `[-448,448]`, and column-major scales;
5. convert the complete packed weight and tile scales;
6. pass activation, `weight.T`, activation scales, and `weight_scale.T` to the
   existing block-scaled CUTLASS GEMM;
7. return `output[:M]`, after which Python restores the original leading
   activation dimensions.

The extra padded activation rows need not contribute observable values because
GEMM rows are independent and the returned tensor excludes them.

Current admitted N and K are positive multiples of 128. Bias and unsupported
layouts use Marlin.

Eligibility is currently SM90-only, excludes VocabParallelEmbedding, requires
logical N/K to equal the Marlin physical N/K, and requires both dimensions to
be divisible by 128. A biased invocation also takes Marlin. Eligible layers
start with m_knee=1; when a vLLM compilation configuration is present, the
knee is raised to max_cudagraph_capture_size+1. This is a capture boundary,
not the completed per-shape performance calibration.

Marlin's use_atomic_add decision is computed during post-load processing with
m=1 and the physical N/K. The native call also carries
USE_FP32_REDUCE_DEFAULT, matching the existing Marlin helper.

The M padding is

\[
M_4=4\left\lceil M/4\right\rceil,
\qquad
\Delta M=M_4-M\in\{0,1,2,3\}.
\]

The underlying vLLM dispatch supports unaligned M by selecting
swap_ab=(M%4)!=0. In this block-scaled SM90 path that selects a much slower
swapped 128-by-16 kernel. Padding keeps the problem on the non-swapped kernel.
The hybrid admits only K and N divisible by 128; it does not infer broader K/N
performance behavior from the absence of an analogous dispatch expression.

The converter has several native implementations. For the current SM90 dense
path, `tile_count=(N/128)*(K/128)`: at most 64 tiles dispatches
`marlin_nvfp4_to_fp8_sparse64_kernel`; 65 tiles and above dispatches
`marlin_nvfp4_to_fp8_tiled_kernel<false, true>`, the synchronous paired
producer. The double-buffered branch is used only by the non-SM90
single-expert path when the K-block count is even, so the SM90-only dense
hybrid does not enter it. Rank-3 expert tensors use the multi-expert tiled
path, and other supported layouts use the generic fallback. The focused SM90
boundary test covers 64 and 65 tiles in both FP16 and BF16; built-operator job
6627092 passed the same boundary against an independent oracle and output
canaries.

With `x=NK/2^20`, jobs 6627093 and 6627094 measured the built selector with a
distinct cold source on every timed launch:

\[
T_{fixed}=9.17897921+0.66824616\max(0,x-14.88865048),
\]
\[
T_{rot}=11.11049993+0.73446567\max(0,x-13.95401231).
\]

Fixed-target train/holdout R2 was 0.999636627/0.993343840; rotating-target
R2 was 0.998633357/0.993419824. These are converter-only surfaces, not the
complete dense branch knee.

## Torch compile and CUDA graphs

TorchDynamo/Inductor sees one _C::marlin_nvfp4_hybrid_linear node. The branch
executes inside C++ once per layer, so different layers may make different
decisions.

During CUDA-graph capture, C++ evaluates the branch for that capture shape and
the selected CUDA kernels become graph nodes. Replay contains those recorded
kernels rather than a host-side M branch. Calls outside captured buckets enter
the native operation and evaluate their own runtime M.

The transparent compile experiment showed that Inductor does not fuse the
three high-path stages. It emits activation quantization, conversion, and
CUTLASS as three native dispatcher calls and plans activation, scale, and
output tensors. Marlin emits one native Marlin call. The current native
boundary preserves the same operations without a Python callback or
branch-independent visible scratch.

The boundary has a tradeoff: scratch allocated inside C++ is not planned by
Inductor. CUDA's caching allocator can reuse blocks, but high-M calls still
make allocation requests. This remains an optimization point.

An earlier caller-owned design exposed every possible buffer before the M
branch. Functionalization and graph capture retained about 0.50 GiB of
otherwise unused decode buffers. Full Nsight also found approximately 83.9
seconds of extra generated Triton slice/view/copy work. That representation
was rejected.

## Memory model

Ignoring allocator rounding and an optional contiguous-input copy, high-M
scratch is:

\[
B_{Aq}=M_4K
\]

bytes of E4M3 activation,

\[
B_{As}=4M_4(K/128)
\]

bytes of FP32 activation scales,

\[
B_{W8}=NK
\]

bytes of E4M3 weight,

\[
B_{Ws}=4(N/128)(K/128)
\]

bytes of FP32 weight scales, and

\[
B_Y=M_4Nb_y
\]

bytes of output, with by=2 for BF16/FP16.

All are transient. Persistent state is ordinary packed NVFP4 Marlin state
plus one uint8 divisor code per 128-by-128 tile. In the current
implementation, the low-M branch returns before creating high-path scratch.

Combining the terms, the allocated high-branch payload is

\[
B_{high}=M_4K+4M_4K/128+NK+4NK/128^2+2M_4N
\]

or

\[
B_{high}=\frac{33}{32}M_4K+\frac{4097}{4096}NK+2M_4N
\]

bytes for FP16/BF16 output. A noncontiguous input can additionally allocate a
logical contiguous copy of `2MK` bytes. Allocator rounding is outside these
payload equations.

The only converter-specific persistent tensor is the divisor grid:

\[
B_{divisor}=(N/128)(K/128)=NK/16384\ \text{bytes}.
\]

The packed FP4 weight occupies `NK/2` bytes, so divisor metadata is `1/8192`
(0.01220703125%) of packed-weight storage. Processed block scales, processed
global scale, packed weight, and Marlin workspace are the ordinary Marlin
resident representation.

## Measured development chronology

The performance history separates three failures that initially looked like
one end-to-end regression.

### Unpadded FP8 path

The first full matched Nsight comparison attributed the GPU time replacing
mixed/prefill Marlin as follows:

| Operator family | Accumulated GPU time |
|---|---:|
| Baseline mixed/prefill Marlin | 133.944193 s |
| Hybrid activation quantization | 6.181891 s |
| Hybrid NVFP4-to-FP8 conversion | 1.166414 s |
| Hybrid FP8 CUTLASS | 175.728514 s |
| Hybrid three-stage total | 183.076819 s |

The three hybrid terms were 36.681% slower than the replaced Marlin time.
CUTLASS alone was 31.195% slower, before adding quantization or conversion.
The interval-union calculation also showed that GPU busy time grew from
260.690150 to 310.959861 seconds while idle gaps fell. The wall regression
was additional GPU work, not hidden host callback time.

The runtime-M split then exposed the shape problem. Of 304 non-decode
iterations, 228 had unaligned M and consumed 167.14 of the 183.08 seconds of
hybrid target work. The aligned M=32768 projections were already 2.48--3.01
times faster than Marlin. The long-prefill premise was therefore sound; most
of the run was reaching the CUTLASS remainder path instead of the aligned
kernel measured by the earlier operator screens.

A narrower early profile had reported about 8.411 seconds of CUTLASS,
1.131 seconds of activation quantization, and 69.7 milliseconds of
conversion. Those numbers described only its selected window. They did not
account for the whole matched run and are not used as its cost decomposition.

### M padding with caller-owned buffers

Rounding M to a multiple of four reversed the linear-compute comparison:

| Replaced work | Accumulated GPU time |
|---|---:|
| Baseline Marlin linear compute | 166.491871 s |
| Padded hybrid linear compute | 79.011461 s |

The high path saved 87.480410 seconds of linear compute. Surrounding generated
work then erased nearly all of it:

| Surrounding family | Baseline | Caller-buffer hybrid | Difference |
|---|---:|---:|---:|
| Generated Triton | 7.812919 s | 91.707721 s | +83.894802 s |
| Attention | 84.237279 s | 87.317397 s | +3.080118 s |

The additional Triton kernels were slice/view/copy operations created while
functionalizing mutable views into the caller-owned shared scratch base.
This also raised the decode graph pool by about 0.50 GiB even though captured
decode selected Marlin.

Job 6640002 measured the same design end to end at 1362.227 output tokens/s:
1.208% above the W4A16 baseline and 18.21% above the earlier broken hybrid.
That result isolated integration overhead from the now-fast padded kernels.

### Functional native operation

The current operation confines the M branch and high-path mutation to C++ and
returns one tensor. The matched Llama result was:

| Configuration | Job | Duration | Output throughput |
|---|---:|---:|---:|
| W4A16 Marlin | 6634231 | 264.583048 s | 1345.974364 tok/s |
| Native runtime-selective hybrid | 6641279 | 178.264048 s | 1997.721946 tok/s |

The throughput change is

\[
\left(\frac{1997.721946}{1345.974364}-1\right)=48.422\%.
\]

Both runs completed 384 requests with 2,827,749 input tokens and 356,122
output tokens. Both used a fixed 32 GiB KV allocation containing 524,288 KV
tokens. The native hybrid recorded the same 0.68 GiB graph pool, 43,605 MiB
post-capture HBM, and 48,417 MiB peak HBM as the matched baseline.

The chronology matters: M padding fixed the CUTLASS kernel regime, while the
functional native boundary independently removed the surrounding
functionalization/view-copy cost. Neither change alone produced the final
result.

### Other workload and eager controls

The earlier unpadded path behaved differently when the scheduler produced
mostly aligned work:

| Workload | Baseline | Hybrid | Throughput change |
|---|---:|---:|---:|
| 50k/1k C64 | 175.272 tok/s, 1004.306 s | 224.144 tok/s, 785.329 s | +27.8835% |
| 1k/1k C128 | 4636.510 tok/s, 76.852 s | 4710.505 tok/s, 75.645 s | +1.5959% |

For 50k/1k, baseline Marlin accumulated 430.238 seconds. The hybrid trace
contained 135.320 seconds of CUTLASS, 18.651 seconds of activation
quantization, 4.624 seconds of conversion, and 41.358 seconds of residual
Marlin. For 1k/1k, the comparable values were 36.614 seconds of baseline
Marlin versus 19.992 residual Marlin, 12.813 CUTLASS, 0.798 quantization, and
0.854 conversion seconds. The short workload had little removable dense time
after decode and attention.

Matched enforce-eager diagnostics reproduced the workload dependence:

| Workload | W4A16 | Hybrid | Change |
|---|---:|---:|---:|
| 8k/1k | 1333.005 tok/s | 1122.983 tok/s | -15.7555% |
| 50k/1k | 193.706 tok/s | 255.113 tok/s | +31.7014% |

Job 6637239 passed every invocation through the hybrid outer interface while
forcing its Marlin branch. It reached 1319.436 tok/s versus 1333.005 for
direct Marlin, a -1.0179% difference. That bounds the old outer callback
interface near 1% in this diagnostic; it cannot explain the 8k regression.

### ABI-matched M-padding operator check

CudaGym measured the actual unaligned M=12,358 shape and its padded M=12,360
form after warmup in both eager and CUDA-graph execution:

| Projection (N,K) | Unpadded full | Padded full | Padded vs current | Marlin | Padded vs Marlin | Unpadded graph | Padded graph |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate/up (28672,4096) | 18,067.024 us | 2,485.968 us | 7.2676x | 8,491.488 us | 3.4158x | 17,908.656 us | 2,474.560 us |
| down (4096,14336) | 7,285.744 us | 1,606.704 us | 4.5346x | 4,075.184 us | 2.5364x | 7,248.464 us | 1,579.328 us |

For gate/up, quantization was 120.432 versus 124.272 microseconds and
conversion was 169.280 microseconds. For down, quantization was 385.728 versus
388.128 microseconds and conversion was 86.576 microseconds. The first 12,358
FP8 output rows were bit-identical between padded and unpadded execution.
Relative L2 against Marlin was 0.031419 and 0.031067.

The retained production-runtime run used the endpoint's PyTorch/CUDA ABI.
Earlier attempts with an incompatible DSO/PTX toolchain or an incorrect
quantizer destination view did not produce comparable timing.

### Focused native behavior checks

The native tests cover three observable properties:

- the registered hybrid schema is functional and has no mutable arguments;
- M below the knee returns exactly the established Marlin result;
- an unaligned M=3 high-path input pads to four rows and agrees with Marlin
  within the established quantization tolerance.

A separate CUTLASS-consumer test uses a strided input view. The focused H100
native run completed both branch cases.

## Additional actual-model measurements

The Q3 and Q36M experiments used three variants:

- A: Marlin dense and Marlin MoE;
- B: Marlin dense and hybrid MoE;
- C: hybrid dense and hybrid MoE.

This makes C minus B the observed incremental effect of the dense path in the
same complete-model setup.

### Q3

All three runs processed 128 requests, 1,048,576 input tokens, and 131,072
output tokens with fixed 32 GiB KV allocation.

| Variant | Job | Duration | Throughput | Change from A |
|---|---:|---:|---:|---:|
| A | 6644863 | 80.839203 s | 1621.391532 tok/s | -- |
| B | 6644868 | 68.884268 s | 1902.785689 tok/s | +17.355% |
| C | 6644870 | 65.459893 s | 2002.325297 tok/s | +23.494% |

The incremental C-over-B throughput change was +5.231%.

### Q36M

The Q36M variants used identical token counts and fixed 32 GiB KV. The
1,875,072-token KV allocation corresponded to a reported maximum concurrency
of 192.75 at length 9728.

| Variant | Job | Duration | Throughput | Change from A |
|---|---:|---:|---:|---:|
| A | 6645223 | 45.222909 s | 2898.353991 tok/s | -- |
| B | 6645230 | 41.311501 s | 3172.772618 tok/s | +9.468% |
| C | 6645235 | 40.971829 s | 3199.076095 tok/s | +10.376% |

A later Q36 formula-ladder run used `M_knee=8,193` and measured the complete
concurrency sequence `[1,2,4,8,16,32,64,128,256,512]`. Job 6646399 was B
(Marlin dense plus hybrid MoE); job 6646400 was C (hybrid dense plus hybrid
MoE). Their exact output-token/s values and incremental dense effect were:

| Concurrency | B tok/s | C tok/s | C/B - 1 |
|---:|---:|---:|---:|
| 1 | 221.1562262 | 220.6201431 | -0.242400% |
| 2 | 432.2056645 | 404.1341633 | -6.494941% |
| 4 | 754.0502025 | 702.6082464 | -6.822086% |
| 8 | 1,145.9857638 | 1,101.3417264 | -3.895689% |
| 16 | 1,624.4314987 | 1,582.8796484 | -2.557932% |
| 32 | 2,175.5886607 | 2,158.6717285 | -0.777580% |
| 64 | 2,692.8749164 | 2,704.2783047 | +0.423465% |
| 128 | 3,182.4137426 | 3,222.0570617 | +1.245700% |
| 256 | 3,024.7278640 | 3,050.9239561 | +0.866064% |
| 512 | 3,155.5385613 | 3,199.3768603 | +1.389249% |

All 20 B/C points completed without a nonempty benchmark error. This is a
complete-model C-minus-B observation, not an isolated dense-GEMM surface.

The incremental C-over-B change was +0.829%. These complete-model increments
are not isolated GEMM timings: the dense selector also sees projections whose
semantic role is not encoded in the local weight object.

## Paired accuracy and router handling

Job 6645286 preceded router protection. Its A/B/C counts were 1,172, 1,171,
and 1,157; A-versus-C had 30 regressions, 15 improvements, and exact McNemar
p=0.0356978. That result identified the router as the missing semantic case,
not an N/K/M calibration failure.

Q3's router is an ordinary quantized `ReplicatedLinear`. Its local input,
weight type, and output shape do not show that its output feeds top-k, so the
dense kernel cannot identify it from layer-local properties.

### Current post-load object-identity scan

`MoERunner.gate` holds the exact linear module whose output becomes router
logits. After per-layer quantization finalization, deferred-attention and HPC
processing, and the optional model-level post-load hook,
`process_weights_after_loading()` walks `model.modules()`. For every
`MoERunner` with a gate it sets `_vllm_is_moe_router=True` on that exact gate.
This occurs before first compilation and changes no prepared weight layout.

`MarlinNvFp4ToFp8LinearKernel.apply_weights()` checks the static marker before
calling the native hybrid and delegates marked layers to its embedded W4A16
Marlin kernel. The focused test confirms that the exact gate is marked while
the shared-expert gate and an unrelated linear are not.

Jobs 6646885, 6646889, and 6646896 repeated the 1,319-question deterministic
Q3 comparison with this guard and `M_knee=4,097`:

| Variant | Job | Correct | Accuracy | Eval s | Output tok/s | `n01` | `n10` | Delta | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: Marlin dense + Marlin MoE | 6646885 | 1,157 | 87.717968% | 46.2953 | 4,225.395 | -- | -- | -- | -- |
| B: Marlin dense + hybrid MoE | 6646889 | 1,162 | 88.097043% | 47.6191 | 4,100.368 | 17 | 12 | +0.379075 pp | 0.4582583 |
| C: hybrid dense + hybrid MoE | 6646896 | 1,159 | 87.869598% | 49.4624 | 3,956.847 | 20 | 18 | +0.151630 pp | 0.8714147 |

Here `n01` is A-wrong/candidate-right and `n10` is
A-right/candidate-wrong. Neither hybrid variant had a statistically
significant paired accuracy change. Artifacts are under
`.benchmark/gsm8k-q3-current-formula-current-formula-20260818-r1/`.

## Router-identification alternatives

### Marking the object at the MoE construction boundary

FusedMoEFactory passes the exact gate object to MoERunner. Marking it at that
handoff is model-independent and occurs before loading. This option was
rejected for the current work because it places dense-kernel policy in a
vLLM-core ownership point rather than keeping the feature local.

### Type, name, and shape inference

These alternatives do not cover the Q3 case:

- GateLinear type checks miss an ordinary ReplicatedLinear;
- module-name and prefix matches depend on model naming;
- shapes are shared by unrelated projections;
- a local linear has no information about its future consumer.

They remain useful diagnostics, but not semantic identification.

### FX or Inductor mutation

Compilation occurs after post-load processing. Dynamo normally inlines the
linear call, so the original module object does not reliably survive as a
call_module node. The MoE body can also be opaque at the outer graph. A late
pass could rewrite a graph operation, but side-effecting the original
kernel's static state from that phase is less direct and interacts with graph
cache identity. This path was recorded and not selected.

## Scratch-placement alternatives

### Current internal allocation

The C++ high branch creates output, quantized activation, activation scales,
converted weight, and converted weight scales after the M test. CUDA's
caching allocator reuses the blocks after warmup. This is the implementation
used by job 6641279.

### Compiler-planned ordinary intermediates

The transparent three-operation compile showed that Inductor can plan
ordinary single-owner activation, scale, and output tensors and dispatch the
three existing native operations directly. It did not fuse the operations.
A future version could expose those functional values again without exposing
mutable views into a shared base.

### Shared native arena

A native arena could remove allocator requests and reuse one maximum-sized
region. It would also own concurrency, stream, and graph-capture lifetime.
No measured allocator bottleneck currently separates it from the working
internal-allocation implementation.

### Caller-owned mutable views

This was the job-6640002 implementation. Its real shared-base views expanded
into clone/slice-scatter/copy work during functionalization and enlarged the
graph pool. The measured failure was representation overhead, not the padded
hybrid kernels.

## Current interpretation

The evidence separates four results:

1. Large-M block-FP8 GEMM throughput can repay activation quantization and
   complete transient weight conversion.
2. Dynamic M alignment changes the selected CUTLASS kernel enough to reverse
   the result.
3. Functionalization of mutable shared-base views can cost more than the
   linear optimization saves.
4. A functional native operation removes that surrounding work and produced
   the matched +48.422% Llama result.

The remaining measured questions are the native post-router throughput across
additional static `(K,N)` shapes, actual per-layer M distributions over the
powers-of-two concurrency ladder, a full-cost N/K refit for the current native
allocation boundary, allocator activity after warmup, and ordinary
tile-boundary steps in additional supported shapes.
