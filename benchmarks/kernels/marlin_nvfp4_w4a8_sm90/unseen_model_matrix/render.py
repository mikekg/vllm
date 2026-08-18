#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import shlex
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONCURRENCIES = [1 << power for power in range(10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=HERE / "matrix.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    runtime = parser.add_mutually_exclusive_group(required=True)
    runtime.add_argument("--overlay")
    runtime.add_argument("--venv")
    parser.add_argument("--python")
    parser.add_argument("--source-revision")
    return parser.parse_args()


def validate(data: dict) -> None:
    assert data["schema_version"] == 1
    matrix = data["matrix"]
    assert matrix["variants"] == ["native_reference", "adaptive"]
    assert matrix["concurrencies"] == CONCURRENCIES
    assert matrix["output_tokens"] == 1024
    assert matrix["kv_cache_memory_bytes"] > 0
    assert [workload["id"] for workload in matrix["workloads"]] == [
        "1k1k",
        "5k1k",
        "8k1k",
        "50k1k",
    ]
    for workload in matrix["workloads"]:
        expected = workload["input_tokens"] + matrix["output_tokens"] + 64
        assert workload["max_model_len"] == expected

    ids = [model["id"] for model in data["models"]]
    assert len(ids) == len(set(ids))
    for model in data["models"]:
        assert model["model_path"].startswith("/lustre/fs1/")
        if shape := model.get("moe_shape"):
            assert model["formula_knee"] == (
                256 * shape["global_experts"] // shape["top_k"] + 1
            )
        else:
            assert model["dense_shape"]
            assert "formula_knee" not in model
        topology = model["topology"]
        assert topology["dp"] == 1
        assert topology["engine_tp"] == (topology["nodes"] * topology["gpus_per_node"])
        if topology["enable_expert_parallel"]:
            assert topology["moe_tp"] == 1
            assert topology["moe_ep"] == topology["engine_tp"]
        else:
            assert topology["moe_ep"] == 1
            assert topology["moe_tp"] == topology["engine_tp"]
        assert model["hybrid_action"] in {"native_only", "per_layer"}


def harness(runtime: dict) -> str:
    del runtime
    return str(HERE)


def extra_serve(model: dict, kv_cache_memory_bytes: int) -> str:
    overrides = model.get("serving_overrides", {})
    tokens = [
        "--max-num-batched-tokens="
        + str(overrides.get("max_num_batched_tokens", 32768)),
        f"--kv-cache-memory-bytes={kv_cache_memory_bytes}",
    ]
    if config := overrides.get("compilation_config"):
        tokens.append(f"--compilation-config={config}")
    return " ".join(tokens)


def common_env(
    data: dict,
    model: dict,
    variant: str,
    overlay: str | None,
    python: str,
) -> dict[str, str]:
    runtime = data["runtime"]
    source = harness(runtime)
    topology = model["topology"]
    overrides = model.get("serving_overrides", {})
    env = {
        "IMAGE": runtime["image"],
        "MOUNTS": "/lustre:/lustre",
        "SCRIPTS": source,
        "RUN_SCRIPT": "serve_overlay.sh",
        "PYTHON": python,
        "PATH": (
            f"{Path(python).parent}:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        ),
        "LD_LIBRARY_PATH": (
            "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:"
            "/usr/local/lib/python3.12/dist-packages/torch/lib"
        ),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_MODULE_LOADING": "EAGER",
        "VLLM_USE_DEEP_GEMM": "1",
        "DG_JIT_USE_NVRTC": "1",
        "VLLM_TEST_FORCE_FP8_MARLIN": "0",
        "VLLM_DISABLED_KERNELS": (
            "MarlinNvFp4ToFp8LinearKernel,NvFp4ByCopyExperts"
            if variant == "native_reference"
            else ""
        ),
        "MODEL_PATH": model["model_path"],
        "SERVED": model["repo"],
        "MODEL_REVISION": model["revision"] or "",
        "TP": str(topology["engine_tp"]),
        "EP": "1" if topology["enable_expert_parallel"] else "0",
        "GPUS": str(topology["engine_tp"]),
        "NODES": str(topology["nodes"]),
        "GPU_UTIL": "0.90",
        "RANGE_RATIO": str(data["matrix"]["random_range_ratio"]),
        "MAX_NUM_SEQS": str(overrides.get("max_num_seqs", 512)),
        "KV_DTYPE": "fp8",
        "EXTRA_SERVE": extra_serve(
            model,
            data["matrix"]["kv_cache_memory_bytes"],
        ),
    }
    if topology["dp"] > 1:
        env["DP_SIZE"] = str(topology["dp"])
    if overlay:
        env.update(
            {
                "PYTHONPATH": f"{overlay}/overlay",
                "CUDA_HOME": f"{overlay}/cuda",
                "LD_PRELOAD": (
                    f"{overlay}/overlay/vllm/_C_stable_libtorch.abi3.so:"
                    f"{overlay}/converter/marlin_nvfp4_to_fp8_sm90a.so"
                ),
            }
        )
    else:
        venv = Path(python).parent.parent
        env["VIRTUAL_ENV"] = str(venv)
        env["CUDA_HOME"] = str(venv.parent / "cuda")
    return env


def write_env(path: Path, env: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Generated by render.py; source only inside the Slurm allocation.\n"
        + "".join(
            f"export {key}={shlex.quote(str(value))}\n" for key, value in env.items()
        ),
        encoding="utf-8",
    )


def sbatch(
    model: dict,
    job_file: str,
    config_file: str,
    name: str,
    time_limit: str,
    dependency: str | None = None,
) -> str:
    topology = model["topology"]
    options = [
        "--parsable",
        f"--nodes={topology['nodes']}",
        f"--ntasks={topology['nodes']}",
        "--ntasks-per-node=1",
        f"--gres=gpu:{topology['gpus_per_node']}",
        f"--time={time_limit}",
        f"--job-name={name}",
    ]
    if dependency:
        options.append(f'--dependency="afterok:${dependency}"')
    return (
        f"job=$(sbatch {' '.join(options)} {shlex.quote(job_file)} "
        f'"$here"/{shlex.quote(config_file)})'
    )


def header(
    manifest: str,
    package: tuple[str, str] | None,
) -> list[str]:
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "[[ $# -eq 1 && $1 =~ ^[0-9]+(:[0-9]+)*$ ]] || { "
        'echo "usage: $0 PREVIOUS_SLURM_JOB[:PREVIOUS_SLURM_JOB...]" >&2; '
        "exit 2; }",
        "job=$1",
        "unset VLLM_DIAGNOSTIC_MOE_M_KNEE VLLM_DISABLED_KERNELS",
        'here=$(cd -- "$(dirname -- "$0")" && pwd)',
        "if [[ $here != /lustre/fs1/* ]]; then",
        '  echo "render under /lustre/fs1" >&2',
        "  exit 2",
        "fi",
        f'manifest="$here/{manifest}"',
        "[[ ! -e $manifest ]]",
    ]
    if package:
        venv, revision = package
        artifact = str(Path(venv).parent)
        site = f"{venv}/lib/python3.12/site-packages/vllm"
        libraries = [
            f"{site}/_C_stable_libtorch.abi3.so",
            f"{site}/_moe_C_stable_libtorch.abi3.so",
            f"{site}/third_party/deep_gemm/_C.cpython-312-x86_64-linux-gnu.so",
        ]
        lines.extend(
            [
                f"venv={shlex.quote(venv)}",
                f"site={shlex.quote(site)}",
                f"[[ $(<{shlex.quote(f'{artifact}/source-revision')}) == "
                f"{shlex.quote(revision)} ]]",
                f"[[ -L {shlex.quote(f'{artifact}/cuda')} ]]",
                *[f"[[ -s {shlex.quote(path)} ]]" for path in libraries],
                'grep -q _vllm_is_moe_router "$site/model_executor/'
                'model_loader/utils.py"',
                'grep -q _vllm_is_moe_router "$site/model_executor/kernels/'
                'linear/nvfp4/marlin_fp8.py"',
                "grep -a -q marlin_nvfp4_hybrid_linear "
                '"$site/_C_stable_libtorch.abi3.so"',
                "grep -a -q marlin_nvfp4_hybrid_moe "
                '"$site/third_party/deep_gemm/'
                '_C.cpython-312-x86_64-linux-gnu.so"',
                f"printf 'source_revision %s\\nvenv %s\\npredecessor %s\\n' "
                f'{shlex.quote(revision)} "$venv" "$job" >"$manifest"',
                "sha256sum "
                + " ".join(shlex.quote(path) for path in libraries)
                + ' >>"$manifest"',
            ]
        )
    else:
        lines.append('printf "predecessor %s\\n" "$job" >"$manifest"')
    lines.extend(
        [
            "gap_after() {",
            "  sbatch --parsable --partition=cpu_short --account=sw_aidot "
            "--qos=normal --cpus-per-task=1 --mem=1G --time=00:06:00 "
            '--dependency="${3:-afterany}:$1" --job-name="$2-gap" '
            "--wrap='sleep 300'",
            "}",
            "",
        ]
    )
    return lines


def record(lines: list[str], label: str) -> None:
    lines.append(f'printf "%s %s\\n" {shlex.quote(label)} "$job" >>"$manifest"')


def record_gap(lines: list[str], label: str) -> None:
    lines.append(f'printf "%s %s\\n" {shlex.quote(label)} "$gap" >>"$manifest"')


def render_performance(
    data: dict,
    run_id: str,
    overlay: str | None,
    python: str,
    output: Path,
    package: tuple[str, str] | None,
) -> tuple[str, str]:
    runtime = data["runtime"]
    matrix = data["matrix"]
    source = harness(runtime)
    job_file = f"{source}/unseen_model_once.sbatch"
    concurrencies = ",".join(map(str, matrix["concurrencies"]))
    workload_time_limits = {
        "1k1k": "00:45:00",
        "5k1k": "01:00:00",
        "8k1k": "01:30:00",
    }
    lines = header("performance.jobs", package)
    high_lines = header("performance_high.jobs", package)

    for model in data["models"]:
        for workload in matrix["workloads"]:
            slices = (
                [
                    (
                        lines,
                        "c1-c128",
                        ",".join(map(str, matrix["concurrencies"][:-2])),
                        "02:00:00",
                    ),
                    (high_lines, "c256", "256", "01:30:00"),
                    (high_lines, "c512", "512", "01:30:00"),
                ]
                if workload["id"] == "50k1k"
                else [
                    (
                        lines,
                        "",
                        concurrencies,
                        workload_time_limits[workload["id"]],
                    )
                ]
            )
            for job_lines, suffix, slice_concurrencies, time_limit in slices:
                for variant in matrix["variants"]:
                    result = (
                        f"{runtime['results']}/{run_id}/{model['id']}/"
                        f"performance/{variant}/{workload['id']}"
                    )
                    if suffix:
                        result += f"/{suffix}"
                    env = common_env(data, model, variant, overlay, python)
                    env.update(
                        {
                            "ISL": str(workload["input_tokens"]),
                            "OSL": str(matrix["output_tokens"]),
                            "CONCS": slice_concurrencies,
                            "RESULT_DIR": result,
                            "CACHE_ROOT": f"{result}/cache",
                            "IXBENCH": (
                                f"{runtime['shared_scripts']}/"
                                "bench_serving/benchmark_serving.py"
                            ),
                        }
                    )
                    stem = f"{model['id']}-{variant}-{workload['id']}"
                    if suffix:
                        stem += f"-{suffix}"
                    config_name = (
                        f"{workload['id']}{'-' + suffix if suffix else ''}.env"
                    )
                    config_file = (
                        Path("configs")
                        / "performance"
                        / model["id"]
                        / variant
                        / config_name
                    )
                    write_env(output / config_file, env)
                    job_lines.append(
                        'gap=$(gap_after "$job" ' + shlex.quote(stem) + ")"
                    )
                    record_gap(job_lines, f"{stem}-gap-before")
                    job_lines.append(
                        sbatch(
                            model,
                            job_file,
                            str(config_file),
                            stem,
                            time_limit,
                            dependency="gap",
                        )
                    )
                    record(job_lines, stem)
        lines.append("")
        high_lines.append("")
    lines.append('printf "tail %s\\n" "$job" >>"$manifest"')
    high_lines.append('printf "tail %s\\n" "$job" >>"$manifest"')
    return "\n".join(lines), "\n".join(high_lines)


def render_gsm8k(
    data: dict,
    run_id: str,
    overlay: str | None,
    python: str,
    output: Path,
    package: tuple[str, str] | None,
) -> str:
    runtime = data["runtime"]
    matrix = data["matrix"]
    source = harness(runtime)
    job_file = f"{source}/unseen_model_once.sbatch"
    examples = str(matrix["accuracy"]["examples"])
    lines = header("gsm8k.jobs", package)

    for model in data["models"]:
        baseline = (
            f"{runtime['results']}/{run_id}/{model['id']}/gsm8k/"
            "native_reference/details.jsonl"
        )
        for index, variant in enumerate(matrix["variants"]):
            result = f"{runtime['results']}/{run_id}/{model['id']}/gsm8k/{variant}"
            env = common_env(data, model, variant, overlay, python)
            env.update(
                {
                    "ISL": "1024",
                    "OSL": "256",
                    "MML_OVERRIDE": "2048",
                    "MAX_NUM_SEQS": "64",
                    "RESULT_DIR": result,
                    "CACHE_ROOT": f"{result}/cache",
                    "GSM8K_CLIENT": f"{source}/gsm8k_endpoint.py",
                    "GSM8K_DATA": runtime["gsm8k_data"],
                    "GSM8K_VARIANT": variant,
                    "GSM8K_EXAMPLES": examples,
                    "GSM8K_MAX_CONCURRENCY": "64",
                }
            )
            if variant == "adaptive":
                env["GSM8K_BASELINE_DETAILS"] = baseline
            stem = f"{model['id']}-{variant}-gsm8k"
            config_file = Path("configs") / "gsm8k" / model["id"] / f"{variant}.env"
            write_env(output / config_file, env)
            dependency = "afterok" if index else "afterany"
            lines.append(
                'gap=$(gap_after "$job" ' + shlex.quote(stem) + f" {dependency})"
            )
            record_gap(lines, f"{stem}-gap-before")
            lines.append(
                sbatch(
                    model,
                    job_file,
                    str(config_file),
                    stem,
                    (
                        "03:00:00"
                        if model["topology"]["nodes"] > 1
                        or model["topology"]["engine_tp"] >= 8
                        else "02:00:00"
                    ),
                    dependency="gap",
                )
            )
            record(lines, stem)
        lines.append("")
    lines.append('printf "tail %s\\n" "$job" >>"$manifest"')
    return "\n".join(lines)


def render_gsm_fixed_timing(
    data: dict,
    run_id: str,
    overlay: str | None,
    python: str,
    output: Path,
    package: tuple[str, str] | None,
) -> str:
    runtime = data["runtime"]
    matrix = data["matrix"]
    source = harness(runtime)
    job_file = f"{source}/unseen_model_once.sbatch"
    concurrencies = ",".join(map(str, matrix["concurrencies"]))
    lines = header("gsm_fixed_timing.jobs", package)
    sequence = (
        ("native_reference", "a1", "1"),
        ("adaptive", "b1", "2"),
        ("native_reference", "a2", "3"),
    )

    for model in data["models"]:
        result_root = f"{runtime['results']}/{run_id}/{model['id']}/gsm_fixed_timing"
        accuracy_root = (
            f"{runtime['results']}/{run_id}/{model['id']}/gsm8k/native_reference"
        )
        for variant, run_label, run_index in sequence:
            result = f"{result_root}/runs/{run_label}"
            env = common_env(data, model, variant, overlay, python)
            env.update(
                {
                    "ISL": "1024",
                    "OSL": "256",
                    "MML_OVERRIDE": "2048",
                    "RESULT_DIR": result,
                    "CACHE_ROOT": f"{result}/cache",
                    "GSM_FIXED_TIMING_CLIENT": f"{source}/gsm_fixed_timing.sh",
                    "GSM8K_DETAILS": f"{accuracy_root}/details.jsonl",
                    "ACCURACY_SUMMARY": f"{accuracy_root}/summary.json",
                    "RESULT_ROOT": result_root,
                    "VARIANT": variant,
                    "RUN_LABEL": run_label,
                    "RUN_INDEX": run_index,
                    "BASELINE_VARIANT": "native_reference",
                    "OUTPUT_TOKENS": "256",
                    "CONCS": concurrencies,
                    "SOURCE_REVISION": package[1] if package else "",
                }
            )
            stem = f"{model['id']}-{variant}-gsm-fixed-{run_label}"
            config_file = (
                Path("configs") / "gsm_fixed_timing" / model["id"] / f"{run_label}.env"
            )
            write_env(output / config_file, env)
            lines.append('gap=$(gap_after "$job" ' + shlex.quote(stem) + ")")
            record_gap(lines, f"{stem}-gap-before")
            lines.append(
                sbatch(
                    model,
                    job_file,
                    str(config_file),
                    stem,
                    (
                        "03:00:00"
                        if model["topology"]["nodes"] > 1
                        or model["topology"]["engine_tp"] >= 8
                        else "02:00:00"
                    ),
                    dependency="gap",
                )
            )
            record(lines, stem)
        lines.append("")
    lines.append('printf "tail %s\\n" "$job" >>"$manifest"')
    return "\n".join(lines)


def render_route_diagnostics(
    data: dict,
    run_id: str,
    overlay: str | None,
    python: str,
    output: Path,
    package: tuple[str, str] | None,
) -> str:
    runtime = data["runtime"]
    matrix = data["matrix"]
    source = harness(runtime)
    job_file = f"{source}/unseen_model_once.sbatch"
    examples = str(matrix["accuracy"]["examples"])
    lines = header("route_diagnostics.jobs", package)

    for model in data["models"]:
        shape = model.get("moe_shape")
        if not shape:
            continue
        retained = (
            f"{runtime['results']}/{run_id}/{model['id']}/gsm8k/"
            "native_reference/details.jsonl"
        )
        baseline = (
            f"{runtime['results']}/{run_id}/{model['id']}/route_diagnostic/"
            "native_reference/summary.json"
        )
        for index, variant in enumerate(matrix["variants"]):
            result = (
                f"{runtime['results']}/{run_id}/{model['id']}/"
                f"route_diagnostic/{variant}"
            )
            env = common_env(data, model, variant, overlay, python)
            env.update(
                {
                    "ISL": "1024",
                    "OSL": "256",
                    "MML_OVERRIDE": "2048",
                    "MAX_NUM_SEQS": "64",
                    "RESULT_DIR": result,
                    "CACHE_ROOT": f"{result}/cache",
                    "GSM8K_CLIENT": f"{source}/routed_expert_diagnostic.py",
                    "GSM8K_DATA": retained,
                    "GSM8K_VARIANT": variant,
                    "GSM8K_EXAMPLES": examples,
                    "GSM8K_MAX_CONCURRENCY": "64",
                    "ROUTE_EXPERTS": str(shape["global_experts"]),
                    "ROUTE_TOP_K": str(shape["top_k"]),
                    "EXTRA_SERVE": (
                        f"{env['EXTRA_SERVE']} --enable-return-routed-experts"
                    ),
                }
            )
            if variant == "adaptive":
                env["GSM8K_BASELINE_DETAILS"] = baseline
            stem = f"{model['id']}-{variant}-route-diagnostic"
            config_file = (
                Path("configs") / "route_diagnostic" / model["id"] / f"{variant}.env"
            )
            write_env(output / config_file, env)
            dependency = "afterok" if index else "afterany"
            lines.append(
                'gap=$(gap_after "$job" ' + shlex.quote(stem) + f" {dependency})"
            )
            record_gap(lines, f"{stem}-gap-before")
            lines.append(
                sbatch(
                    model,
                    job_file,
                    str(config_file),
                    stem,
                    (
                        "03:00:00"
                        if model["topology"]["nodes"] > 1
                        or model["topology"]["engine_tp"] >= 8
                        else "02:00:00"
                    ),
                    dependency="gap",
                )
            )
            record(lines, stem)
        lines.append("")
    lines.append('printf "tail %s\\n" "$job" >>"$manifest"')
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    validate(data)
    package = None
    if args.venv:
        if args.python or not args.source_revision:
            raise SystemExit("--venv requires --source-revision and derives --python")
        if not args.venv.startswith("/lustre/fs1/"):
            raise SystemExit("--venv must be a validated /lustre/fs1 path")
        if len(args.source_revision) != 40 or any(
            char not in "0123456789abcdef" for char in args.source_revision
        ):
            raise SystemExit("--source-revision must be a full lowercase Git SHA")
        overlay = None
        python = f"{args.venv.rstrip('/')}/bin/python"
        package = (args.venv.rstrip("/"), args.source_revision)
    else:
        if not args.overlay.startswith("/lustre/fs1/"):
            raise SystemExit("--overlay must be a validated /lustre/fs1 path")
        if not args.python or args.source_revision:
            raise SystemExit("--overlay requires --python only")
        overlay = args.overlay
        python = args.python
    image_pythons = {
        "/opt/venv/bin/python",
        "/usr/bin/python3",
        "/usr/local/bin/python",
    }
    if python not in image_pythons and not python.startswith("/lustre/fs1/"):
        raise SystemExit(
            "--python must be /opt/venv/bin/python, /usr/bin/python3, "
            "/usr/local/bin/python, or a validated /lustre/fs1 path"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    performance = args.output / "submit_performance.sh"
    performance_high = args.output / "submit_performance_high.sh"
    gsm8k = args.output / "submit_gsm8k.sh"
    fixed_timing = args.output / "submit_gsm_fixed_timing.sh"
    routes = args.output / "submit_route_diagnostics.sh"
    rendered_performance, rendered_performance_high = render_performance(
        data,
        args.run_id,
        overlay,
        python,
        args.output,
        package,
    )
    performance.write_text(
        rendered_performance + "\n",
        encoding="utf-8",
    )
    performance_high.write_text(
        rendered_performance_high + "\n",
        encoding="utf-8",
    )
    gsm8k.write_text(
        render_gsm8k(
            data,
            args.run_id,
            overlay,
            python,
            args.output,
            package,
        )
        + "\n",
        encoding="utf-8",
    )
    fixed_timing.write_text(
        render_gsm_fixed_timing(
            data,
            args.run_id,
            overlay,
            python,
            args.output,
            package,
        )
        + "\n",
        encoding="utf-8",
    )
    routes.write_text(
        render_route_diagnostics(
            data,
            args.run_id,
            overlay,
            python,
            args.output,
            package,
        )
        + "\n",
        encoding="utf-8",
    )
    performance.chmod(0o755)
    performance_high.chmod(0o755)
    gsm8k.chmod(0o755)
    fixed_timing.chmod(0o755)
    routes.chmod(0o755)
    moe_models = sum("moe_shape" in model for model in data["models"])
    print(
        f"rendered {len(data['models']) * 8} performance and "
        f"{len(data['models']) * 4} high-concurrency and "
        f"{len(data['models']) * 2} GSM8K and "
        f"{len(data['models']) * 3} fixed-token timing and "
        f"{moe_models * 2} route-diagnostic jobs"
    )


if __name__ == "__main__":
    main()
