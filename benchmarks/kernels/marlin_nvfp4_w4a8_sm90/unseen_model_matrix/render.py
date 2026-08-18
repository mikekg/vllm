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
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--python", required=True)
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
    suffix = (
        "build-source/benchmarks/kernels/marlin_nvfp4_w4a8_sm90/unseen_model_matrix"
    )
    return f"{runtime['root']}/{suffix}"


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
    overlay: str,
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
        "PYTHONPATH": f"{overlay}/overlay",
        "CUDA_HOME": f"{overlay}/cuda",
        "LD_LIBRARY_PATH": (
            "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:"
            "/usr/local/lib/python3.12/dist-packages/torch/lib"
        ),
        "LD_PRELOAD": (
            f"{overlay}/overlay/vllm/_C_stable_libtorch.abi3.so:"
            f"{overlay}/converter/marlin_nvfp4_to_fp8_sm90a.so"
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
    dependency: str | None = None,
    time_limit: str = "12:00:00",
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


def header(manifest: str) -> list[str]:
    return [
        "#!/bin/bash",
        "set -euo pipefail",
        "unset VLLM_DIAGNOSTIC_MOE_M_KNEE VLLM_DISABLED_KERNELS",
        'here=$(cd -- "$(dirname -- "$0")" && pwd)',
        "if [[ $here != /lustre/fs1/* ]]; then",
        '  echo "render under /lustre/fs1" >&2',
        "  exit 2",
        "fi",
        f'manifest="$here/{manifest}"',
        ': >"$manifest"',
        "gap_after() {",
        "  sbatch --parsable --partition=cpu_short --account=sw_aidot "
        "--qos=normal --cpus-per-task=1 --mem=1G --time=00:06:00 "
        '--dependency="${3:-afterany}:$1" --job-name="$2-gap" '
        "--wrap='sleep 300'",
        "}",
        "",
    ]


def record(lines: list[str], label: str) -> None:
    lines.append(f'printf "%s %s\\n" {shlex.quote(label)} "$job" >>"$manifest"')


def record_gap(lines: list[str], label: str) -> None:
    lines.append(f'printf "%s %s\\n" {shlex.quote(label)} "$gap" >>"$manifest"')


def render_performance(
    data: dict,
    run_id: str,
    overlay: str,
    python: str,
    output: Path,
) -> str:
    runtime = data["runtime"]
    matrix = data["matrix"]
    source = harness(runtime)
    job_file = f"{source}/unseen_model_once.sbatch"
    concurrencies = ",".join(map(str, matrix["concurrencies"]))
    lines = header("performance.jobs")

    for model in data["models"]:
        first = True
        for variant in matrix["variants"]:
            for workload in matrix["workloads"]:
                result = (
                    f"{runtime['results']}/{run_id}/{model['id']}/"
                    f"performance/{variant}/{workload['id']}"
                )
                env = common_env(data, model, variant, overlay, python)
                env.update(
                    {
                        "ISL": str(workload["input_tokens"]),
                        "OSL": str(matrix["output_tokens"]),
                        "CONCS": concurrencies,
                        "RESULT_DIR": result,
                        "CACHE_ROOT": f"{result}/cache",
                        "IXBENCH": (
                            f"{runtime['shared_scripts']}/"
                            "bench_serving/benchmark_serving.py"
                        ),
                    }
                )
                stem = f"{model['id']}-{variant}-{workload['id']}"
                config_file = (
                    Path("configs")
                    / "performance"
                    / model["id"]
                    / variant
                    / f"{workload['id']}.env"
                )
                write_env(output / config_file, env)
                if first:
                    lines.append(sbatch(model, job_file, str(config_file), stem))
                    first = False
                else:
                    lines.append('gap=$(gap_after "$job" ' + shlex.quote(stem) + ")")
                    record_gap(lines, f"{stem}-gap-before")
                    lines.append(
                        sbatch(
                            model,
                            job_file,
                            str(config_file),
                            stem,
                            dependency="gap",
                        )
                    )
                record(lines, stem)
        lines.append("")
    return "\n".join(lines)


def render_gsm8k(
    data: dict,
    run_id: str,
    overlay: str,
    python: str,
    output: Path,
) -> str:
    runtime = data["runtime"]
    matrix = data["matrix"]
    source = harness(runtime)
    job_file = f"{source}/unseen_model_once.sbatch"
    examples = str(matrix["accuracy"]["examples"])
    lines = header("gsm8k.jobs")

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
            if index == 0:
                lines.append(
                    sbatch(
                        model,
                        job_file,
                        str(config_file),
                        stem,
                        time_limit="08:00:00",
                    )
                )
            else:
                lines.append(
                    'gap=$(gap_after "$job" ' + shlex.quote(stem) + " afterok)"
                )
                record_gap(lines, f"{stem}-gap-before")
                lines.append(
                    sbatch(
                        model,
                        job_file,
                        str(config_file),
                        stem,
                        dependency="gap",
                        time_limit="08:00:00",
                    )
                )
            record(lines, stem)
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    validate(data)
    if not args.overlay.startswith("/lustre/fs1/"):
        raise SystemExit("--overlay must be a validated /lustre/fs1 path")
    image_pythons = {
        "/opt/venv/bin/python",
        "/usr/bin/python3",
        "/usr/local/bin/python",
    }
    if args.python not in image_pythons and not args.python.startswith("/lustre/fs1/"):
        raise SystemExit(
            "--python must be /opt/venv/bin/python, /usr/bin/python3, "
            "/usr/local/bin/python, or a validated /lustre/fs1 path"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    performance = args.output / "submit_performance.sh"
    gsm8k = args.output / "submit_gsm8k.sh"
    performance.write_text(
        render_performance(
            data,
            args.run_id,
            args.overlay,
            args.python,
            args.output,
        )
        + "\n",
        encoding="utf-8",
    )
    gsm8k.write_text(
        render_gsm8k(
            data,
            args.run_id,
            args.overlay,
            args.python,
            args.output,
        )
        + "\n",
        encoding="utf-8",
    )
    performance.chmod(0o755)
    gsm8k.chmod(0o755)
    print(
        f"rendered {len(data['models']) * 8} performance and "
        f"{len(data['models']) * 2} GSM8K jobs"
    )


if __name__ == "__main__":
    main()
