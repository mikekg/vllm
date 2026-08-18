# NVFP4 Hybrid Build, JIT, and CUDA Graph Flow

This diagram separates package compilation, model-startup compilation, and
CUDA graph capture. CUDA graph capture records launches; it does not compile
kernels.

```mermaid
flowchart TB
    subgraph build["1. Package build — CPU node; no GPU required"]
        source["vLLM + pinned CUTLASS and DeepGEMM sources"] --> cmake["CMake + Ninja"]
        cmake --> host["g++ · C++20"]
        cmake --> nvcc["nvcc · SM90a cubins + selected 8.0+PTX images"]
        host --> dense_so["_C_stable_libtorch.abi3.so<br/>Marlin · vLLM CUTLASS · converter · quantization"]
        nvcc --> dense_so
        host --> moe_so["_moe_C_stable_libtorch.abi3.so<br/>MoE alignment · routing · Marlin MoE"]
        nvcc --> moe_so
        cmake --> dg_component["_deep_gemm_C install component"]
        host --> dg_so["vllm/third_party/deep_gemm/<br/>_C.cpython-312-x86_64-linux-gnu.so<br/>outer MoE operator · runtime-JIT launcher"]
        dg_component --> dg_so
        dg_component --> dg_runtime["DeepGEMM Python package<br/>vendored CUTLASS/CuTe headers · envs.py"]
    end

    subgraph startup["2. Model startup — GPU process"]
        package["Load stable DSOs<br/>assert vendored DeepGEMM origin and native MoE op"] --> weights["Load weights and run post-load preparation<br/>Marlin packing · divisor codes · router marking"]
        weights --> dynamo["torch.compile<br/>Dynamo → FX graph → vLLM passes"]
        dynamo --> inductor{"Inductor lowering"}
        inductor --> generated["Generated Triton<br/>Triton compiler → PTX → ptxas → cubin"]
        inductor --> external["External calls<br/>ATen / cuBLAS / cuBLASLt / cuDNN"]
        inductor --> opaque["Opaque native vLLM hybrid operators"]
        inductor -. max-autotune + backend opt-in .-> generated_cutlass["Generated CUTLASS CUDA<br/>nvcc → cached object / shared library"]
        generated --> warmup["Warm representative shapes"]
        generated_cutlass --> warmup
        external --> warmup
        opaque --> warmup
        warmup -. DeepGEMM path reached .-> dg_jit["DeepGEMM runtime JIT<br/>NVRTC when selected; otherwise nvcc<br/>cache by generated specialization/config"]
        warmup -. fallback only .-> triton_jit["Handwritten Triton JIT<br/>first signature → cached kernel"]
        warmup -. optional attention backend .-> flashinfer["FlashInfer<br/>prebuilt cubin or nvcc JIT"]
    end

    subgraph capture["3. CUDA graph capture and serving"]
        ready["Kernels needed for capture are ready"] --> descriptor{"Capture descriptor<br/>padded tokens · mode-specific request fields<br/>LoRA presence and count bucket"}
        descriptor --> decisions{"Each hybrid invocation compares<br/>its M with its own knee"}
        decisions --> low["Marlin branch"]
        decisions --> high["conversion + FP8 branch"]
        low --> graph["One fixed CUDA-graph entry for the descriptor<br/>possibly a mix of branches"]
        high --> graph
        request["Serving step"] --> dispatch{"vLLM graph lookup by batch descriptor"}
        dispatch -->|"matching captured descriptor"| graph
        graph --> replay["CUDA graph replay"]
        dispatch -->|"no matching graph"| ordinary["Ordinary execution"]
        ordinary --> runtime_knee{"Native operator evaluates<br/>the padded tensor-row M"}
        runtime_knee -->|"below knee"| ordinary_low["Marlin"]
        runtime_knee -->|"at or above knee"| ordinary_high["conversion + FP8 GEMM<br/>MoE may JIT an uncached config"]
    end

    dense_so --> package
    moe_so --> package
    dg_so --> package
    dg_runtime --> package
    warmup -->|"AOT / already cached"| ready
    dg_jit --> ready
    triton_jit --> ready
    flashinfer --> ready
```

`torch.jit`/TorchScript is not used by this path. The similarly named Triton,
DeepGEMM, and FlashInfer JITs are independent runtime kernel compilers.

During CUDA graph capture, the native C++ knee decision executes once per
hybrid-operator invocation. Replay contains only that invocation's recorded GPU
launches and no host branch. The operator's M is the padded tensor row count
produced by that execution. Full-graph keys retain their computed request-count
field; piecewise keys replace it with `None` and set uniformity to `False`.
Uniformity is otherwise specialized only when the configured mode separates
decode and mixed routines. LoRA specialization records presence and an
active-count bucket, not adapter identity. Each `CUDAGraphWrapper` has its own
descriptor entries (full mode has one full-model wrapper).

The current Q3/Q36 configurations capture at most 512 tokens while every MoE
knee is at least 3,072; the dense selector also raises its knee above the graph
capture limit. Every hybrid invocation in their captured entries therefore
chooses Marlin. Larger prefills use ordinary execution, where each native
operator evaluates M against its own knee and can select conversion plus FP8
dynamically. A captured entry could contain FP8 or a mix of branches in a future
configuration whose operator knees lie inside its capture range, but that is
not current campaign behavior.

The generic DeepGEMM warmup does not enumerate `NvFp4ByCopyExperts`. A startup
profile may still compile the high-M configuration it exercises, but it does
not exhaustively cover later ordinary-execution configurations. DeepGEMM keys
its cache by generated code, compiler, and flags; this hybrid compiles N and K
while keeping M dynamic, so multiple M values can share one specialization and
a later M can select and compile another configuration.

## Compiler inventory

| Name | Role | Why it exists here |
| --- | --- | --- |
| CMake and Ninja | Configure and schedule the package build; they are not compilers | Build the exact source/dependency graph once and avoid recompiling unchanged objects |
| `g++` | Compile `.cpp` dispatcher and binding code, including the DeepGEMM host launcher | Produce loadable Python/PyTorch bindings and runtime-JIT control code |
| `nvcc` at package build | Compile Marlin, CUTLASS, conversion, quantization, and MoE CUDA translation units, including their host wrappers, to SM90a and selected forward-compatible PTX images | Ship stable, tested GPU kernels in the vLLM libraries |
| TorchDynamo | Trace Python bytecode into an FX graph; it does not emit GPU machine code | Expose model computation to vLLM and Inductor while leaving registered custom operators opaque |
| vLLM FX/Inductor passes | Partition and rewrite the FX graph; they are not standalone machine-code compilers | Preserve vLLM custom operators and apply model-level fusions |
| TorchInductor | Choose lowerings and generate wrappers/kernels | Mix fused generated code with calls to ATen and vLLM native operators |
| Triton compiler | Compile Inductor-generated and handwritten Triton kernels | Efficient shape-specialized pointwise, reduction, routing, and fallback kernels |
| `ptxas` | Assemble Triton's PTX into a cubin | Produce the GPU binary loaded for a Triton kernel; standard Triton does not invoke `nvcc` |
| Inductor CUTLASS templates plus `nvcc` | Optional Inductor GEMM candidate when max-autotune is active and `CUTLASS` is in its backend list | Generate and benchmark a CUTLASS CUDA instantiation that Inductor owns |
| ATen / cuBLAS / cuBLASLt / cuDNN | Prebuilt external libraries selected by Inductor; not compilers | Reuse vendor kernels when generating a new kernel is unnecessary or slower |
| DeepGEMM NVRTC | Compile each generated grouped-FP8 GEMM specialization/config directly to a cached cubin | Avoid AOT-instantiating the large DeepGEMM configuration space; this hybrid keeps M dynamic |
| DeepGEMM `nvcc` JIT | Alternate DeepGEMM runtime compiler when `DG_JIT_USE_NVRTC` is false | Compatibility fallback; the unseen-model campaign selects NVRTC |
| FlashInfer cubin loader or `nvcc` JIT | Load a packaged FlashInfer cubin, otherwise compile its requested kernel | Support optional attention/MoE backends; it is not required by this NVFP4 hybrid |
| CuTe DSL compiler | Compile model/backend-specific `cute.compile` kernels | Used by selected DeepSeek, Kimi, Mamba, and other specialized paths, not by this NVFP4 hybrid itself |
| CUDA driver | Load cubins and JIT any forward-compatible PTX images to device code | Converter/Hopper CUTLASS arrive as SM90a cubins; selected Marlin kernels arrive as 8.0+PTX |
| CUDA graph capture | Record already-compiled launches and addresses; not a compiler | Remove steady-state CPU launch overhead through descriptor entries owned by each graph wrapper |

## Build and runtime requirements

This validation artifact builds all three binary components in the serving
environment: CPython 3.12, PyTorch 2.11.0+cu130, C++20, SM90a cubins, and
selected forward-compatible PTX. The two stable DSOs intentionally use the
Python abi3 and PyTorch stable ABI; the DeepGEMM CPython DSO is tied to the
target CPython SOABI and PyTorch C++ ABI. CMake/Ninja schedule the build; `g++`
compiles `.cpp` code, while `nvcc` compiles CUDA translation units and their
host portions. The build also needs PyTorch headers and libraries, the pinned
vLLM CUTLASS source, the separately pinned DeepGEMM source and submodules,
CUDA/CCCL headers, a compatible NVRTC library, and the wheel-split cuSPARSE
headers discovered through `cuda.pathfinder`.

The relevant install has three DSOs plus nonbinary DeepGEMM artifacts.
`_deep_gemm_C` also installs the DeepGEMM Python files, `envs.py`, and its
vendored CUTLASS/CuTe headers because runtime JIT consumes them. vLLM's CUTLASS
sources build the AOT custom operators; DeepGEMM's vendored CUTLASS is a
distinct runtime-JIT input. `build_pdx.sh` creates the copied source and venv;
before `render.py` can use them, the artifact also needs a separately
provisioned `cuda` symlink to the merged CUDA/CCCL shim and a `source-revision`
marker. Because the venv uses `--system-site-packages`, it remains tied to the
serving image rather than being a self-contained wheel or container.

vLLM normally prefers an external `deep_gemm` package over its vendored copy.
The campaign must therefore assert that the imported module originates under
`vllm.third_party.deep_gemm` and that
`torch.ops._C.marlin_nvfp4_hybrid_moe` is registered; inspecting the bundled
DSO alone does not establish which implementation is active.

At runtime `_deep_gemm_C` must be able to load NVRTC even when
`DG_JIT_USE_NVRTC=0`. NVRTC mode needs the CUDA driver and CUDA/CCCL headers;
the `nvcc` mode additionally needs `nvcc` and its supported host C++ toolchain.
Both modes need a writable `DG_JIT_CACHE_DIR`. This CUDA-13 build uses driver
kernel enumeration, so DeepGEMM itself needs `cuobjdump` only for requested
assembly/SASS dumps; the campaign launcher nevertheless preflights it. The
packaged campaign points `CUDA_HOME` at its separately provisioned merged shim
and gives every job a private cache directory. CUDA graph replay itself needs
no compiler, but uncached ordinary-execution specializations still do.

## What the current campaign actually selects

In the current vLLM runs, Inductor's GEMM candidate set is the default
`ATEN,TRITON,CPP`. These configurations provide no single compile size, so
vLLM leaves PyTorch `max_autotune` false; the campaign also does not add
`CUTLASS`. Inductor therefore does not generate or `nvcc`-compile CUTLASS for
these models.

Production dense execution presents one opaque
`_C::marlin_nvfp4_hybrid_linear` call to Inductor. Its C++ body selects Marlin
or invokes the packaged converter and `_C::cutlass_scaled_mm`. Production MoE
similarly presents `_C::marlin_nvfp4_hybrid_moe`, whose DeepGEMM extension owns
the low/high branch. Thus a serving graph mixes Inductor-generated Triton with
opaque AOT Marlin/CUTLASS calls and runtime-JIT DeepGEMM calls; Inductor does
not generate those three implementations. Direct converter and scaled-mm nodes
seen in the retained compiler probes came from the synthetic branch-isolation
benchmark, not the production composition.

The unseen-model campaign explicitly sets `DG_JIT_USE_NVRTC=1`. A run that
does not set it uses DeepGEMM's `nvcc` JIT instead; this changes the compiler,
not the grouped-GEMM operator selected by the hybrid.
