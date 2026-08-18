#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import json
import math
import tempfile
from collections import Counter
from pathlib import Path

import regex as re
import yaml
from gsm8k_endpoint import exact_mcnemar
from render import validate

HERE = Path(__file__).resolve().parent
PERFORMANCE_TARGET = 0.20
PERFORMANCE_STRETCH = 0.40
ACCURACY_POINT_MARGIN_PP = -0.5
ACCURACY_LOWER_MARGIN_PP = -1.0
MCNEMAR_ALPHA = 0.05
SHA256_LINE = re.compile(r"^([0-9a-f]{64})\s+(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=HERE / "matrix.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--accuracy-threshold", action="append", default=[])
    parser.add_argument("--performance-target-cell", action="append", default=[])
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def close(value: object, expected: float) -> bool:
    return (
        isinstance(value, int | float)
        and math.isfinite(value)
        and math.isclose(value, expected, rel_tol=1e-9)
    )


def file_record(path: Path, *, nonempty: bool = True) -> tuple[dict | None, str | None]:
    try:
        data = path.read_bytes()
    except OSError as error:
        return None, f"missing {path}: {error}"
    if nonempty and not data:
        return None, f"empty {path}"
    return {
        "path": str(path),
        "sha256": digest(data),
        "bytes": len(data),
    }, None


def json_record(path: Path) -> tuple[dict | None, dict | None, str | None]:
    record, error = file_record(path)
    if error:
        return None, record, error
    assert record is not None
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, record, f"invalid JSON {path}: {error}"
    if not isinstance(value, dict):
        return None, record, f"JSON root is not an object: {path}"
    return value, record, None


def jsonl_record(path: Path) -> tuple[list[dict] | None, dict | None, str | None]:
    record, error = file_record(path)
    if error:
        return None, record, error
    assert record is not None
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, record, f"invalid JSONL {path}: {error}"
    if not all(isinstance(row, dict) for row in rows):
        return None, record, f"JSONL contains a non-object row: {path}"
    return rows, record, None


def parse_thresholds(values: list[str]) -> dict[str, float]:
    thresholds = {}
    for value in values:
        model, separator, threshold = value.partition("=")
        if not separator or not model:
            raise ValueError(f"invalid accuracy threshold: {value}")
        number = float(threshold)
        if not 0 <= number <= 1:
            raise ValueError(f"accuracy threshold must be in [0, 1]: {value}")
        thresholds[model] = number
    return thresholds


def parse_target_cells(values: list[str], data: dict) -> set[tuple[str, str, int]]:
    expected = {
        (model["id"], workload["id"], concurrency)
        for model in data["models"]
        if model["hybrid_action"] == "per_layer"
        for workload in data["matrix"]["workloads"]
        for concurrency in data["matrix"]["concurrencies"]
    }
    cells = set()
    for value in values:
        try:
            model, workload, concurrency_text = value.split("/")
            cell = model, workload, int(concurrency_text)
        except ValueError as error:
            raise ValueError(f"invalid performance target cell: {value}") from error
        if cell not in expected:
            raise ValueError(f"unknown performance target cell: {value}")
        cells.add(cell)
    return cells


def parse_manifests(paths: list[Path]) -> tuple[list[dict], dict[str, dict]]:
    manifests = []
    selected_jobs = {}
    for index, path in enumerate(paths):
        record, error = file_record(path)
        manifest = {
            "index": index,
            "path": str(path),
            "status": "missing" if error else "complete",
            "artifact": record,
            "metadata": {},
            "file_hashes": {},
            "jobs": {},
            "issues": [error] if error else [],
        }
        if error:
            manifests.append(manifest)
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line:
                continue
            if match := SHA256_LINE.fullmatch(line):
                manifest["file_hashes"][match.group(2)] = match.group(1)
                continue
            key, separator, value = line.partition(" ")
            if not separator:
                manifest["issues"].append(f"line {line_number}: {line!r}")
                continue
            if key in {"source_revision", "venv", "predecessor", "tail"}:
                manifest["metadata"][key] = value
                continue
            if not value.isdigit():
                manifest["issues"].append(f"line {line_number}: {line!r}")
                continue
            if key in manifest["jobs"]:
                manifest["issues"].append(f"duplicate job label: {key}")
            manifest["jobs"][key] = value
        revision = manifest["metadata"].get("source_revision", "")
        if len(revision) != 40 or any(
            char not in "0123456789abcdef" for char in revision
        ):
            manifest["issues"].append("missing full source_revision")
        if not manifest["metadata"].get("venv", "").startswith("/lustre/fs1/"):
            manifest["issues"].append("missing validated venv")
        for suffix in (
            "/_C_stable_libtorch.abi3.so",
            "/_moe_C_stable_libtorch.abi3.so",
            "/third_party/deep_gemm/_C.cpython-312-x86_64-linux-gnu.so",
        ):
            if not any(name.endswith(suffix) for name in manifest["file_hashes"]):
                manifest["issues"].append(f"missing package hash for {suffix}")
        if manifest["issues"]:
            manifest["status"] = "invalid"
        for label, job_id in manifest["jobs"].items():
            previous = selected_jobs.get(label)
            selected_jobs[label] = {
                "job_id": job_id,
                "manifest_index": index,
                "supersedes": previous,
            }
        manifests.append(manifest)
    return manifests, selected_jobs


def expected_disabled(variant: str) -> set[str]:
    if variant == "adaptive":
        return set()
    return {"MarlinNvFp4ToFp8LinearKernel", "NvFp4ByCopyExperts"}


def server_log_record(path: Path, variant: str) -> tuple[dict | None, str | None]:
    record, error = file_record(path)
    if error:
        return record, error
    marker = "UNSEEN_MODEL_RUNTIME disabled_kernels=" + ",".join(
        sorted(expected_disabled(variant))
    )
    if marker not in path.read_text(encoding="utf-8", errors="replace").splitlines():
        return record, f"server log backend marker mismatch: {path}"
    return record, None


def runtime_provenance(
    directory: Path,
    model: dict,
    variant: str,
    client_name: str,
    filename: str = "runtime.provenance",
) -> tuple[dict | None, list[str]]:
    path = directory / filename
    record, error = file_record(path)
    if error:
        return None, [error]
    fields = {}
    hashes = {}
    issues = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if match := SHA256_LINE.fullmatch(line):
            hashes[match.group(2)] = match.group(1)
            continue
        key, separator, value = line.partition(" ")
        if not separator or key in fields:
            issues.append(f"runtime provenance line {line_number}: {line!r}")
        else:
            fields[key] = value
    if fields.get("schema_version") != "1":
        issues.append("runtime provenance schema_version != 1")
    if fields.get("model") != model["repo"]:
        issues.append("runtime provenance model mismatch")
    revision = model["revision"] or ""
    if fields.get("model_revision") != revision:
        issues.append("runtime provenance model revision mismatch")
    disabled = set(filter(None, fields.get("disabled_kernels", "").split(",")))
    if disabled != expected_disabled(variant):
        issues.append("runtime provenance backend mismatch")
    required_suffixes = (".env", "/serve_overlay.sh", f"/{client_name}")
    for suffix in required_suffixes:
        if not any(name.endswith(suffix) for name in hashes):
            issues.append(f"runtime provenance lacks {suffix} hash")
    return {
        "artifact": record,
        "fields": fields,
        "file_hashes": hashes,
    }, issues


def performance_payload_issues(
    result: dict,
    model: dict,
    concurrency: int,
    prompts: int,
) -> list[str]:
    issues = []

    def expect(condition: bool, message: str) -> None:
        if not condition:
            issues.append(message)

    expect(result.get("backend") == "vllm", "backend != vllm")
    expect(result.get("model_id") == model["repo"], "model_id mismatch")
    expect(result.get("max_concurrency") == concurrency, "concurrency mismatch")
    expect(result.get("num_prompts") == prompts, "num_prompts mismatch")
    expect(result.get("completed") == prompts, "completed request count mismatch")
    expect(result.get("failed") == 0, "failed request count is nonzero")
    input_lens = result.get("input_lens")
    output_lens = result.get("output_lens")
    errors = result.get("errors")
    expect(isinstance(input_lens, list), "input_lens missing")
    expect(isinstance(output_lens, list), "output_lens missing")
    expect(isinstance(errors, list), "errors missing")
    if isinstance(input_lens, list):
        expect(len(input_lens) == prompts, "input_lens count mismatch")
        expect(
            all(isinstance(value, int) and value > 0 for value in input_lens),
            "invalid input_lens",
        )
        expect(
            sum(input_lens) == result.get("total_input_tokens"), "input total mismatch"
        )
    if isinstance(output_lens, list):
        expect(len(output_lens) == prompts, "output_lens count mismatch")
        expect(
            all(isinstance(value, int) and value > 0 for value in output_lens),
            "invalid output_lens",
        )
        expect(
            sum(output_lens) == result.get("total_output_tokens"),
            "output total mismatch",
        )
    if isinstance(errors, list):
        expect(len(errors) == prompts, "errors count mismatch")
        expect(not any(errors), "request error list is nonempty")
    duration = result.get("duration")
    throughput = result.get("output_throughput")
    expect(
        isinstance(duration, int | float) and math.isfinite(duration) and duration > 0,
        "invalid duration",
    )
    expect(
        isinstance(throughput, int | float)
        and math.isfinite(throughput)
        and throughput > 0,
        "invalid output throughput",
    )
    if (
        isinstance(duration, int | float)
        and duration > 0
        and isinstance(throughput, int | float)
        and isinstance(result.get("total_output_tokens"), int)
    ):
        expect(
            math.isclose(
                throughput,
                result["total_output_tokens"] / duration,
                rel_tol=1e-9,
            ),
            "output throughput does not reproduce",
        )
    return issues


def performance_directory(
    results: Path,
    model_id: str,
    variant: str,
    workload: str,
    concurrency: int,
) -> Path:
    directory = results / model_id / "performance" / variant / workload
    if workload == "50k1k":
        directory /= "c1-c128" if concurrency <= 128 else f"c{concurrency}"
    return directory


def performance_label(
    model_id: str, variant: str, workload: str, concurrency: int
) -> str:
    label = f"{model_id}-{variant}-{workload}"
    if workload == "50k1k":
        label += "-c1-c128" if concurrency <= 128 else f"-c{concurrency}"
    return label


def performance_row(
    results: Path,
    model: dict,
    variant: str,
    workload: dict,
    concurrency: int,
    jobs: dict[str, dict],
) -> tuple[dict, dict | None]:
    directory = performance_directory(
        results, model["id"], variant, workload["id"], concurrency
    )
    path = directory / f"bench_c{concurrency}.json"
    prompts = min(max(3 * concurrency, 20), 512)
    label = performance_label(model["id"], variant, workload["id"], concurrency)
    row = {
        "kind": "performance",
        "model": model["id"],
        "variant": variant,
        "workload": workload["id"],
        "concurrency": concurrency,
        "job_label": label,
        "submission": jobs.get(label),
        "result": str(path),
        "model_revision_pinned": model["revision"] is not None,
        "issues": [],
        "artifacts": {},
    }
    if not directory.is_dir():
        row["issues"].append(f"missing result directory: {directory}")
        if not row["submission"]:
            row["issues"].append("submission manifest lacks expected job label")
        row["status"] = "missing"
        row["retry_required"] = True
        return row, None
    result, result_record, error = json_record(path)
    if result_record:
        row["artifacts"]["result"] = result_record
    if error:
        row["issues"].append(error)
    if not row["submission"]:
        row["issues"].append("submission manifest lacks expected job label")
    if result is not None:
        row["issues"].extend(
            performance_payload_issues(result, model, concurrency, prompts)
        )
    provenance, issues = runtime_provenance(
        directory,
        model,
        variant,
        "benchmark_serving.py",
        f"runtime_c{concurrency}.provenance",
    )
    if provenance:
        row["artifacts"]["runtime_provenance"] = provenance
    row["issues"].extend(issues)
    for name, support in (
        ("client_log", directory / f"bench_c{concurrency}.log"),
        ("server_log", directory / f"server_c{concurrency}.log"),
    ):
        record, support_error = (
            server_log_record(support, variant)
            if name == "server_log"
            else file_record(support)
        )
        if record:
            row["artifacts"][name] = record
        if support_error:
            row["issues"].append(support_error)
    row["status"] = (
        ("missing" if result is None and not path.exists() else "invalid")
        if row["issues"]
        else "complete"
    )
    row["retry_required"] = row["status"] != "complete"
    if result is not None:
        row["metrics"] = {
            key: result.get(key)
            for key in (
                "duration",
                "completed",
                "failed",
                "total_input_tokens",
                "total_output_tokens",
                "output_throughput",
                "request_throughput",
                "total_token_throughput",
            )
        }
    return row, result


def performance_pair(
    model: dict,
    workload: dict,
    concurrency: int,
    rows: dict[str, tuple[dict, dict | None]],
    target_applicable: bool,
) -> dict:
    baseline_row, baseline = rows["native_reference"]
    candidate_row, candidate = rows["adaptive"]
    pair = {
        "kind": "performance_pair",
        "model": model["id"],
        "workload": workload["id"],
        "concurrency": concurrency,
        "issues": [],
    }
    if baseline_row["status"] != "complete":
        pair["issues"].append("native_reference artifact is incomplete")
    if candidate_row["status"] != "complete":
        pair["issues"].append("adaptive artifact is incomplete")
    if not pair["issues"]:
        assert baseline is not None and candidate is not None
        for key in ("completed", "total_input_tokens", "total_output_tokens"):
            if baseline[key] != candidate[key]:
                pair["issues"].append(f"paired {key} mismatch")
        for key in ("input_lens", "output_lens"):
            if baseline[key] != candidate[key]:
                pair["issues"].append(f"paired {key} mismatch")
    if not pair["issues"]:
        assert baseline is not None and candidate is not None
        gain = candidate["output_throughput"] / baseline["output_throughput"] - 1
        time_saved = 1 - candidate["duration"] / baseline["duration"]
        pair.update(
            {
                "native_output_tps": baseline["output_throughput"],
                "adaptive_output_tps": candidate["output_throughput"],
                "throughput_gain": gain,
                "throughput_gain_percent": 100 * gain,
                "time_saved": time_saved,
                "time_saved_percent": 100 * time_saved,
                "selector_candidate": model["hybrid_action"] == "per_layer",
                "target_applicable": target_applicable,
                "gates": {
                    "fixed_token_match": {"pass": True},
                    "gain_at_least_20_percent": {
                        "pass": gain >= PERFORMANCE_TARGET,
                        "requires_selector_engagement": True,
                    },
                    "gain_at_least_40_percent": {
                        "pass": gain >= PERFORMANCE_STRETCH,
                        "stretch": True,
                    },
                    "target_acceptance": {
                        "pass": gain >= PERFORMANCE_TARGET,
                        "applicable": target_applicable,
                    },
                },
            }
        )
    source_statuses = baseline_row["status"], candidate_row["status"]
    pair["status"] = (
        ("missing" if "missing" in source_statuses else "invalid")
        if pair["issues"]
        else "complete"
    )
    pair["retry_required"] = pair["status"] != "complete" and all(
        status == "complete" for status in source_statuses
    )
    return pair


def gsm_payload_issues(
    summary: dict,
    details: list[dict],
    model: dict,
    variant: str,
    questions: int,
) -> list[str]:
    issues = []

    def expect(condition: bool, message: str) -> None:
        if not condition:
            issues.append(message)

    expect(summary.get("model") == model["repo"], "model mismatch")
    expect(summary.get("variant") == variant, "variant mismatch")
    expect(summary.get("num_questions") == questions, "question count mismatch")
    expect(len(details) == questions, "details row count mismatch")
    expect(
        summary.get("workload_kind") == "gsm8k_accuracy_variable_output",
        "workload kind mismatch",
    )
    expect(summary.get("matched_token_throughput") is False, "GSM timing mislabeled")
    expect(summary.get("max_concurrency") == 64, "GSM concurrency mismatch")
    revision = model["revision"]
    if revision is not None:
        expect(summary.get("model_revision") == revision, "model revision mismatch")
    correct = sum(bool(row.get("correct")) for row in details)
    invalid = sum(row.get("parsed_answer") is None for row in details)
    for index, row in enumerate(details):
        expect(isinstance(row.get("question_id"), str), f"row {index} question_id")
        expect(isinstance(row.get("question"), str), f"row {index} question")
        expect(isinstance(row.get("prompt"), str), f"row {index} prompt")
        expect(isinstance(row.get("gold_answer"), int), f"row {index} gold_answer")
        expect(isinstance(row.get("correct"), bool), f"row {index} correct")
        expect(
            isinstance(row.get("generated_answer"), str),
            f"row {index} generated_answer",
        )
    completion_tokens = [row.get("completion_tokens") for row in details]
    prompt_tokens = [row.get("prompt_tokens") for row in details]
    expect(
        all(isinstance(value, int) and value >= 0 for value in completion_tokens),
        "invalid completion token counts",
    )
    expect(
        all(isinstance(value, int) and value > 0 for value in prompt_tokens),
        "invalid prompt token counts",
    )
    expect(summary.get("correct") == correct, "correct count mismatch")
    expect(summary.get("invalid") == invalid, "invalid count mismatch")
    if details:
        expect(
            close(summary.get("accuracy"), correct / len(details)),
            "accuracy mismatch",
        )
        expect(
            close(summary.get("invalid_rate"), invalid / len(details)),
            "invalid rate mismatch",
        )
    if all(isinstance(value, int) for value in completion_tokens):
        expect(
            summary.get("total_output_tokens") == sum(completion_tokens),
            "output token total mismatch",
        )
        completion_summary = summary.get("completion_tokens")
        expect(isinstance(completion_summary, dict), "completion token summary missing")
        if isinstance(completion_summary, dict) and completion_tokens:
            expect(
                completion_summary.get("min") == min(completion_tokens),
                "minimum completion tokens mismatch",
            )
            expect(
                completion_summary.get("max") == max(completion_tokens),
                "maximum completion tokens mismatch",
            )
            expect(
                completion_summary.get("total") == sum(completion_tokens),
                "completion token summary total mismatch",
            )
    if all(isinstance(value, int) for value in prompt_tokens):
        expect(
            summary.get("total_prompt_tokens") == sum(prompt_tokens),
            "prompt token total mismatch",
        )
    prompts = [row.get("prompt") for row in details]
    if all(isinstance(prompt, str) for prompt in prompts):
        prompt_hash = digest(
            json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        expect(
            summary.get("prompt_corpus_sha256") == prompt_hash,
            "prompt corpus hash mismatch",
        )
    else:
        issues.append("detail prompt missing")
    latency = summary.get("latency")
    output_tps = summary.get("tokens_per_second")
    expect(
        isinstance(latency, int | float) and math.isfinite(latency) and latency > 0,
        "invalid latency",
    )
    expect(
        isinstance(output_tps, int | float)
        and math.isfinite(output_tps)
        and output_tps > 0,
        "invalid output TPS",
    )
    if isinstance(latency, int | float) and latency > 0:
        expect(
            close(summary.get("questions_per_second"), questions / latency),
            "questions per second does not reproduce",
        )
        if isinstance(summary.get("total_output_tokens"), int):
            expect(
                close(output_tps, summary["total_output_tokens"] / latency),
                "GSM output TPS does not reproduce",
            )
    return issues


def gsm_row(
    results: Path,
    model: dict,
    variant: str,
    questions: int,
    jobs: dict[str, dict],
) -> tuple[dict, list[dict] | None, dict | None]:
    directory = results / model["id"] / "gsm8k" / variant
    label = f"{model['id']}-{variant}-gsm8k"
    row = {
        "kind": "gsm8k",
        "model": model["id"],
        "variant": variant,
        "job_label": label,
        "submission": jobs.get(label),
        "model_revision_pinned": model["revision"] is not None,
        "issues": [],
        "artifacts": {},
    }
    if not directory.is_dir():
        row["issues"].append(f"missing result directory: {directory}")
        if not row["submission"]:
            row["issues"].append("submission manifest lacks expected job label")
        row["status"] = "missing"
        row["retry_required"] = True
        return row, None, None
    summary, summary_record, summary_error = json_record(directory / "summary.json")
    details, details_record, details_error = jsonl_record(directory / "details.jsonl")
    if summary_record:
        row["artifacts"]["summary"] = summary_record
    if details_record:
        row["artifacts"]["details"] = details_record
    row["issues"].extend(filter(None, (summary_error, details_error)))
    if not row["submission"]:
        row["issues"].append("submission manifest lacks expected job label")
    if summary is not None and details is not None:
        row["issues"].extend(
            gsm_payload_issues(summary, details, model, variant, questions)
        )
        row["metrics"] = {
            key: summary.get(key)
            for key in (
                "correct",
                "accuracy",
                "invalid",
                "invalid_rate",
                "latency",
                "questions_per_second",
                "total_prompt_tokens",
                "total_output_tokens",
                "tokens_per_second",
                "completion_tokens",
            )
        }
    provenance, issues = runtime_provenance(
        directory, model, variant, "gsm8k_endpoint.py"
    )
    if provenance:
        row["artifacts"]["runtime_provenance"] = provenance
    row["issues"].extend(issues)
    server_record, server_error = server_log_record(
        directory / "server-rank0.log", variant
    )
    if server_record:
        row["artifacts"]["server_log"] = server_record
    if server_error:
        row["issues"].append(server_error)
    summary_path = directory / "summary.json"
    details_path = directory / "details.jsonl"
    missing = not summary_path.exists() or not details_path.exists()
    row["status"] = (
        ("missing" if missing else "invalid") if row["issues"] else "complete"
    )
    row["retry_required"] = row["status"] != "complete"
    return row, details, summary


def accuracy_gates(
    baseline: dict,
    candidate: dict,
    paired: dict,
    threshold: float | None,
) -> dict:
    delta = paired["delta_percentage_points"]
    p_value = paired["exact_two_sided_mcnemar_p"]
    gates = {
        "absolute_accuracy": {
            "threshold": threshold,
            "value": candidate["accuracy"],
            "pass": None if threshold is None else candidate["accuracy"] >= threshold,
        },
        "point_delta_at_least_minus_0_5_pp": {
            "threshold": ACCURACY_POINT_MARGIN_PP,
            "value": delta,
            "pass": delta >= ACCURACY_POINT_MARGIN_PP,
        },
        "paired_lower95_above_minus_1_pp": {
            "threshold": ACCURACY_LOWER_MARGIN_PP,
            "value": paired["delta_pp_lower95"],
            "pass": paired["delta_pp_lower95"] > ACCURACY_LOWER_MARGIN_PP,
        },
        "no_significant_mcnemar_regression": {
            "alpha": MCNEMAR_ALPHA,
            "value": p_value,
            "pass": not (delta < 0 and p_value <= MCNEMAR_ALPHA),
        },
        "no_invalid_increase": {
            "baseline": baseline["invalid"],
            "candidate": candidate["invalid"],
            "pass": candidate["invalid"] <= baseline["invalid"],
        },
    }
    values = [gate["pass"] for gate in gates.values()]
    gates["overall"] = (
        "fail" if False in values else "pending" if None in values else "pass"
    )
    return gates


def gsm_pair(
    results: Path,
    model: dict,
    questions: int,
    variants: dict[str, tuple[dict, list[dict] | None, dict | None]],
    threshold: float | None,
) -> dict:
    baseline_row, baseline_details, baseline = variants["native_reference"]
    candidate_row, candidate_details, candidate = variants["adaptive"]
    path = results / model["id"] / "gsm8k" / "paired.summary.json"
    pair = {
        "kind": "gsm8k_pair",
        "model": model["id"],
        "result": str(path),
        "issues": [],
        "artifacts": {},
    }
    if baseline_row["status"] != "complete":
        pair["issues"].append("native_reference GSM artifact is incomplete")
    if candidate_row["status"] != "complete":
        pair["issues"].append("adaptive GSM artifact is incomplete")
    paired, paired_record, error = json_record(path)
    if paired_record:
        pair["artifacts"]["paired_summary"] = paired_record
    if error:
        pair["issues"].append(error)
    if not pair["issues"]:
        assert baseline_details is not None and candidate_details is not None
        assert baseline is not None and candidate is not None and paired is not None
        identity = ("question_id", "question", "prompt", "gold_answer")
        baseline_ids = [
            tuple(row.get(key) for key in identity) for row in baseline_details
        ]
        candidate_ids = [
            tuple(row.get(key) for key in identity) for row in candidate_details
        ]
        if baseline_ids != candidate_ids:
            pair["issues"].append("paired GSM item identity mismatch")
        n01 = sum(
            not left["correct"] and right["correct"]
            for left, right in zip(baseline_details, candidate_details)
        )
        n10 = sum(
            left["correct"] and not right["correct"]
            for left, right in zip(baseline_details, candidate_details)
        )
        expected = {
            "n": questions,
            "n01_baseline_wrong_candidate_right": n01,
            "n10_baseline_right_candidate_wrong": n10,
            "baseline_correct": baseline["correct"],
            "candidate_correct": candidate["correct"],
        }
        for key, value in expected.items():
            if paired.get(key) != value:
                pair["issues"].append(f"paired {key} mismatch")
        delta = 100.0 * (n01 - n10) / questions
        p_value = exact_mcnemar(n01, n10)
        if paired.get("model") != model["repo"]:
            pair["issues"].append("paired model mismatch")
        if paired.get("ci_method") != (
            "paired multinomial bootstrap, seed 0, 200000 draws"
        ):
            pair["issues"].append("paired confidence method mismatch")
        if not close(paired.get("delta_percentage_points"), delta):
            pair["issues"].append("paired accuracy delta mismatch")
        if not close(paired.get("exact_two_sided_mcnemar_p"), p_value):
            pair["issues"].append("paired McNemar p-value mismatch")
        lower95 = paired.get("delta_pp_lower95")
        if not isinstance(lower95, int | float) or not math.isfinite(lower95):
            pair["issues"].append("paired delta_pp_lower95 missing")
    if not pair["issues"]:
        assert paired is not None and baseline is not None and candidate is not None
        pair.update(
            {
                "summary": {
                    "questions": questions,
                    "baseline_correct": baseline["correct"],
                    "candidate_correct": candidate["correct"],
                    "baseline_accuracy": baseline["accuracy"],
                    "candidate_accuracy": candidate["accuracy"],
                    "baseline_invalid": baseline["invalid"],
                    "candidate_invalid": candidate["invalid"],
                    "n01": paired["n01_baseline_wrong_candidate_right"],
                    "n10": paired["n10_baseline_right_candidate_wrong"],
                    "delta_percentage_points": paired["delta_percentage_points"],
                    "delta_pp_lower95": paired["delta_pp_lower95"],
                    "exact_two_sided_mcnemar_p": paired["exact_two_sided_mcnemar_p"],
                    "baseline_natural_output_tps": baseline["tokens_per_second"],
                    "candidate_natural_output_tps": candidate["tokens_per_second"],
                },
                "gates": accuracy_gates(baseline, candidate, paired, threshold),
            }
        )
    source_statuses = baseline_row["status"], candidate_row["status"]
    pair["status"] = (
        (
            "missing"
            if "missing" in source_statuses or (paired is None and not path.exists())
            else "invalid"
        )
        if pair["issues"]
        else "complete"
    )
    pair["retry_required"] = pair["status"] != "complete" and all(
        status == "complete" for status in source_statuses
    )
    return pair


def count_status(rows: list[dict]) -> dict[str, int]:
    counts = Counter(row["status"] for row in rows)
    return {key: counts.get(key, 0) for key in ("complete", "missing", "invalid")}


def build_report(args: argparse.Namespace) -> dict:
    matrix_bytes = args.matrix.read_bytes()
    data = yaml.safe_load(matrix_bytes)
    validate(data)
    thresholds = parse_thresholds(args.accuracy_threshold)
    target_cells = parse_target_cells(args.performance_target_cell, data)
    unknown = sorted(set(thresholds) - {model["id"] for model in data["models"]})
    if unknown:
        raise ValueError(f"unknown accuracy-threshold models: {', '.join(unknown)}")
    manifests, jobs = parse_manifests(args.manifest)
    performance_rows = []
    performance_pairs = []
    for model in data["models"]:
        for workload in data["matrix"]["workloads"]:
            for concurrency in data["matrix"]["concurrencies"]:
                variants = {}
                for variant in data["matrix"]["variants"]:
                    variants[variant] = performance_row(
                        args.results, model, variant, workload, concurrency, jobs
                    )
                    performance_rows.append(variants[variant][0])
                performance_pairs.append(
                    performance_pair(
                        model,
                        workload,
                        concurrency,
                        variants,
                        (model["id"], workload["id"], concurrency) in target_cells,
                    )
                )
    gsm_rows = []
    gsm_pairs = []
    questions = data["matrix"]["accuracy"]["examples"]
    for model in data["models"]:
        variants = {}
        for variant in data["matrix"]["variants"]:
            variants[variant] = gsm_row(args.results, model, variant, questions, jobs)
            gsm_rows.append(variants[variant][0])
        gsm_pairs.append(
            gsm_pair(
                args.results,
                model,
                questions,
                variants,
                thresholds.get(model["id"]),
            )
        )
    all_rows = performance_rows + performance_pairs + gsm_rows + gsm_pairs
    retries = [
        {
            "kind": row["kind"],
            "model": row["model"],
            **{
                key: row[key]
                for key in ("variant", "workload", "concurrency", "job_label")
                if key in row
            },
            "status": "retry",
            "causes": row["issues"],
        }
        for row in all_rows
        if row["retry_required"]
    ]
    harness = {}
    for path in (
        HERE / "render.py",
        HERE / "serve_overlay.sh",
        HERE / "unseen_model_once.sbatch",
        HERE / "gsm8k_endpoint.py",
        Path(__file__).resolve(),
    ):
        record, error = file_record(path)
        if error:
            raise RuntimeError(error)
        harness[path.name] = record
    accuracy_states = Counter(
        pair.get("gates", {}).get("overall", "incomplete") for pair in gsm_pairs
    )
    model_revision_gate = all(model["revision"] for model in data["models"])
    structural_complete = not retries and all(
        manifest["status"] == "complete" for manifest in manifests
    )
    accuracy_decision = (
        "fail"
        if accuracy_states["fail"]
        else "pending"
        if accuracy_states["pending"] or accuracy_states["incomplete"]
        else "pass"
    )
    target_pairs = [pair for pair in performance_pairs if pair.get("target_applicable")]
    performance_decision = (
        "pending_target_scope"
        if not target_cells
        else "pending"
        if len(target_pairs) != len(target_cells)
        or any(pair["status"] != "complete" for pair in target_pairs)
        else "fail"
        if any(not pair["gates"]["target_acceptance"]["pass"] for pair in target_pairs)
        else "pass"
    )
    successful = (
        structural_complete
        and model_revision_gate
        and accuracy_decision == "pass"
        and performance_decision == "pass"
    )
    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "results": str(args.results),
        "gate_policy": {
            "performance_target": PERFORMANCE_TARGET,
            "performance_stretch": PERFORMANCE_STRETCH,
            "accuracy_point_margin_pp": ACCURACY_POINT_MARGIN_PP,
            "accuracy_lower95_margin_pp": ACCURACY_LOWER_MARGIN_PP,
            "mcnemar_alpha": MCNEMAR_ALPHA,
            "natural_gsm_tps_is_matched_performance": False,
        },
        "provenance": {
            "matrix": {
                "path": str(args.matrix),
                "sha256": digest(matrix_bytes),
            },
            "harness": harness,
            "manifests": manifests,
            "manifest_precedence": "later --manifest arguments supersede labels",
            "accuracy_thresholds": thresholds,
            "performance_target_cells": [
                f"{model}/{workload}/{concurrency}"
                for model, workload, concurrency in sorted(target_cells)
            ],
            "model_revisions_pinned": model_revision_gate,
        },
        "progress": {
            "performance_artifacts": count_status(performance_rows),
            "performance_pairs": count_status(performance_pairs),
            "gsm_artifacts": count_status(gsm_rows),
            "gsm_pairs": count_status(gsm_pairs),
            "retry_rows": len(retries),
        },
        "performance_rows": performance_rows,
        "performance_pairs": performance_pairs,
        "gsm_rows": gsm_rows,
        "gsm_pairs": gsm_pairs,
        "retry_rows": retries,
        "decision": {
            "structural_evidence": "pass" if structural_complete else "fail",
            "checkpoint_revision_provenance": (
                "pass" if model_revision_gate else "fail"
            ),
            "accuracy": accuracy_decision,
            "performance": performance_decision,
            "successful": successful,
            "successful_when": (
                "structural and revision provenance pass, every GSM gate passes, "
                "and the 20% gain passes at selector-engaged fixed-token points"
            ),
        },
    }


def self_test() -> None:
    payload = {
        "backend": "vllm",
        "model_id": "repo/model",
        "max_concurrency": 1,
        "num_prompts": 2,
        "completed": 2,
        "failed": 0,
        "input_lens": [3, 4],
        "output_lens": [5, 6],
        "errors": [None, None],
        "total_input_tokens": 7,
        "total_output_tokens": 11,
        "duration": 2.0,
        "output_throughput": 5.5,
    }
    model = {"repo": "repo/model"}
    assert not performance_payload_issues(payload, model, 1, 2)
    payload["errors"][1] = "failed"
    assert "request error list is nonempty" in performance_payload_issues(
        payload, model, 1, 2
    )
    baseline = {"accuracy": 0.8, "invalid": 0}
    candidate = {"accuracy": 0.8, "invalid": 0}
    paired = {
        "delta_percentage_points": 0.0,
        "delta_pp_lower95": -0.9,
        "exact_two_sided_mcnemar_p": 1.0,
    }
    assert accuracy_gates(baseline, candidate, paired, 0.75)["overall"] == "pass"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "jobs"
        path.write_text(
            "source_revision " + "a" * 40 + "\n"
            "venv /lustre/fs1/package/venv\n"
            + "b" * 64
            + "  /package/_C_stable_libtorch.abi3.so\n"
            + "c" * 64
            + "  /package/_moe_C_stable_libtorch.abi3.so\n"
            + "d" * 64
            + "  /package/third_party/deep_gemm/"
            "_C.cpython-312-x86_64-linux-gnu.so\n"
            "model-adaptive-1k1k 42\n"
        )
        manifests, jobs = parse_manifests([path])
        assert manifests[0]["status"] == "complete"
        assert jobs["model-adaptive-1k1k"]["job_id"] == "42"
        results = Path(directory) / "results"
        result_dir = performance_directory(results, "model", "adaptive", "1k1k", 1)
        missing_row, _ = performance_row(
            results,
            {"id": "model", "repo": "repo/model", "revision": "rev"},
            "adaptive",
            {"id": "1k1k"},
            1,
            {},
        )
        assert missing_row["status"] == "missing"
        assert "submission manifest lacks expected job label" in missing_row["issues"]
        missing_gsm, details, summary = gsm_row(
            results,
            {"id": "model", "repo": "repo/model", "revision": "rev"},
            "adaptive",
            20,
            {},
        )
        assert (missing_gsm["status"], details, summary) == (
            "missing",
            None,
            None,
        )
        assert "submission manifest lacks expected job label" in missing_gsm["issues"]
        result_dir.mkdir(parents=True)
        complete_payload = {
            **payload,
            "num_prompts": 20,
            "completed": 20,
            "input_lens": [3] * 20,
            "output_lens": [5] * 20,
            "errors": [None] * 20,
            "total_input_tokens": 60,
            "total_output_tokens": 100,
            "output_throughput": 50.0,
        }
        (result_dir / "bench_c1.json").write_text(json.dumps(complete_payload))
        (result_dir / "bench_c1.log").write_text("complete\n")
        (result_dir / "server_c1.log").write_text(
            "UNSEEN_MODEL_RUNTIME disabled_kernels=\n"
        )
        (result_dir / "runtime_c1.provenance").write_text(
            "schema_version 1\nmodel repo/model\nmodel_revision rev\n"
            "disabled_kernels \n"
            + "a" * 64
            + "  /config.env\n"
            + "b" * 64
            + "  /serve_overlay.sh\n"
            + "c" * 64
            + "  /benchmark_serving.py\n"
        )
        row, _ = performance_row(
            results,
            {"id": "model", "repo": "repo/model", "revision": "rev"},
            "adaptive",
            {"id": "1k1k"},
            1,
            jobs,
        )
        assert row["status"] == "complete", row["issues"]


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    missing = [
        name for name in ("run_id", "results", "output") if getattr(args, name) is None
    ]
    if missing or not args.manifest:
        raise SystemExit(
            "missing arguments: "
            + ", ".join([*missing, *([] if args.manifest else ["manifest"])])
        )
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"output": str(args.output), **report["progress"]}, sort_keys=True)
    )
    if not report["decision"]["successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
