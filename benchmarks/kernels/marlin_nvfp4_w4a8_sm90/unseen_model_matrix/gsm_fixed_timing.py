#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector_sha256(values: list[int]) -> str:
    data = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip() and not line.startswith("#")
        ]


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def natural_stop_health(summary: dict, count: int) -> dict:
    latency = summary.get("latency")
    questions_per_second = summary.get("questions_per_second")
    tokens_per_second = summary.get("tokens_per_second")
    output_tokens = summary.get("total_output_tokens")
    if questions_per_second is None and latency:
        questions_per_second = count / latency
    if tokens_per_second is None and latency and output_tokens is not None:
        tokens_per_second = output_tokens / latency
    required = {
        "questions_per_second": questions_per_second,
        "tokens_per_second": tokens_per_second,
        "total_prompt_tokens": summary.get("total_prompt_tokens"),
        "total_output_tokens": output_tokens,
        "completion_tokens": summary.get("completion_tokens"),
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "natural-stop summary lacks health fields: " + ", ".join(missing)
        )
    return {
        "interpretation": "descriptive_only_not_matched_token_throughput",
        "matched_token_throughput": False,
        "num_questions": count,
        **required,
    }


def make_random_prompts(tokenizer, targets: list[int], seed: int) -> list[str]:
    import numpy as np

    from vllm.benchmarks.datasets.datasets import gen_prompt_decode_to_target_len

    special_count = int(tokenizer.num_special_tokens_to_add())
    allowed = np.setdiff1d(
        np.arange(tokenizer.vocab_size), np.asarray(tokenizer.all_special_ids)
    )
    rng = np.random.default_rng(seed)
    prompts = []
    for index, target in enumerate(targets):
        inner_length = target - special_count
        if inner_length < 1:
            raise ValueError(f"prompt token target {target} is too short")
        offset = int(rng.integers(0, len(allowed)))
        token_ids = allowed[
            (offset + index + np.arange(inner_length)) % len(allowed)
        ].tolist()
        prompt, _, mismatch = gen_prompt_decode_to_target_len(
            tokenizer,
            token_ids,
            inner_length,
            rng=rng,
        )
        actual = len(tokenizer(prompt).input_ids)
        if mismatch or actual != target:
            raise ValueError(
                f"random prompt {index} has {actual} tokens, expected {target}"
            )
        prompts.append(prompt)
    return prompts


def prepare(args: argparse.Namespace) -> None:
    from vllm.tokenizers import get_tokenizer

    if args.output_tokens < 1:
        raise ValueError("output token count must be positive")
    details = read_jsonl(args.details)
    if not details:
        raise ValueError("GSM8K details are empty")
    prompts = [row.get("prompt") for row in details]
    targets = [row.get("prompt_tokens") for row in details]
    if not all(isinstance(prompt, str) and prompt for prompt in prompts):
        raise ValueError("every GSM8K detail must contain a non-empty prompt")
    if not all(isinstance(value, int) and value > 0 for value in targets):
        raise ValueError("every GSM8K detail must contain positive prompt_tokens")
    question_ids = [
        str(row.get("question_id", index)) for index, row in enumerate(details)
    ]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("GSM8K question IDs are not unique")

    summary_path = args.accuracy_summary or args.details.with_name("summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("num_questions", len(details)) != len(details):
        raise ValueError("natural-stop summary and details counts differ")

    tokenizer = get_tokenizer(
        args.tokenizer,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )
    actual_targets = [len(tokenizer(prompt).input_ids) for prompt in prompts]
    if actual_targets != targets:
        mismatch = next(
            index
            for index, (actual, expected) in enumerate(zip(actual_targets, targets))
            if actual != expected
        )
        raise ValueError(
            f"tokenizer disagrees with frozen prompt {mismatch}: "
            f"{actual_targets[mismatch]} != {targets[mismatch]}"
        )

    random_prompts = make_random_prompts(tokenizer, targets, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gsm_path = args.output_dir / "gsm8k-fixed.jsonl"
    random_path = args.output_dir / "random-matched-fixed.jsonl"
    write_jsonl(
        gsm_path,
        [{"output_tokens": args.output_tokens, "prompt": prompt} for prompt in prompts],
    )
    write_jsonl(
        random_path,
        [
            {"output_tokens": args.output_tokens, "prompt": prompt}
            for prompt in random_prompts
        ],
    )

    prompt_corpus_sha = hashlib.sha256(
        json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    if summary.get("prompt_corpus_sha256") not in (None, prompt_corpus_sha):
        raise ValueError("natural-stop summary prompt corpus hash differs")
    runner = Path(__file__).with_suffix(".sh")
    provenance = {
        "schema_version": 1,
        "workload_kind": "gsm8k_fixed_token_timing_twin",
        "source_revision": args.source_revision or None,
        "model_revision": args.model_revision or None,
        "tokenizer": args.tokenizer,
        "tokenizer_mode": args.tokenizer_mode,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
        "num_requests": len(prompts),
        "fixed_output_tokens": args.output_tokens,
        "ignore_eos": True,
        "disable_shuffle": True,
        "skip_chat_template": True,
        "request_rate": "inf",
        "temperature": 0,
        "warmup_policy": "2 * max_concurrency",
        "request_count_policy": (
            "min(total_requests, max(20, min(3 * max_concurrency, 512)))"
        ),
        "limitation": (
            "Both twins force a constant output length; natural GSM8K completion "
            "lengths are intentionally not reproduced."
        ),
        "source_details": str(args.details.resolve()),
        "source_details_sha256": sha256(args.details),
        "source_summary": str(summary_path.resolve()),
        "source_summary_sha256": sha256(summary_path),
        "prompt_corpus_sha256": prompt_corpus_sha,
        "question_id_vector_sha256": hashlib.sha256(
            json.dumps(question_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "prompt_token_lengths": targets,
        "prompt_token_vector_sha256": vector_sha256(targets),
        "datasets": {
            "gsm8k": {"path": str(gsm_path.resolve()), "sha256": sha256(gsm_path)},
            "random": {
                "path": str(random_path.resolve()),
                "sha256": sha256(random_path),
            },
        },
        "code_sha256": {
            "analyzer": sha256(Path(__file__)),
            "runner": sha256(runner) if runner.exists() else None,
        },
        "natural_stop_gsm8k_health": natural_stop_health(summary, len(details)),
    }
    write_json(args.output_dir / "provenance.json", provenance)
    print(json.dumps(provenance, sort_keys=True))


def checked_result(
    path: Path,
    provenance: dict,
    provenance_sha: str,
    run_label: str,
    workload: str,
    concurrency: int,
) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    count = min(provenance["num_requests"], max(20, min(3 * concurrency, 512)))
    output_tokens = provenance["fixed_output_tokens"]
    expected_inputs = provenance["prompt_token_lengths"][:count]
    expected_kind = f"{workload}_fixed_token_timing"
    expected_dataset_sha = provenance["datasets"][workload]["sha256"]
    expected = {
        "run_label": run_label,
        "workload_kind": expected_kind,
        "provenance_sha256": provenance_sha,
        "dataset_sha256": expected_dataset_sha,
        "source_details_sha256": provenance["source_details_sha256"],
        "ignore_eos": "true",
        "fixed_output_tokens": str(output_tokens),
        "temperature": "0",
        "num_warmups": str(2 * concurrency),
        "timing_prompt_count": str(count),
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"{path}: {key}={result.get(key)!r}, expected {value!r}")
    if result.get("max_concurrency") != concurrency:
        raise ValueError(f"{path}: concurrency metadata differs")
    if result.get("num_prompts") != count or result.get("completed") != count:
        raise ValueError(f"{path}: incomplete request set")
    if result.get("failed", 0):
        raise ValueError(f"{path}: failed requests present")
    if result.get("input_lens") != expected_inputs:
        raise ValueError(f"{path}: input token vector differs from provenance")
    if result.get("output_lens") != [output_tokens] * count:
        raise ValueError(f"{path}: fixed output token vector differs")
    for key in ("generated_texts", "errors", "ttfts", "itls"):
        if len(result.get(key, [])) != count:
            raise ValueError(f"{path}: --save-detailed field {key} is incomplete")
    if result.get("output_throughput", 0) <= 0:
        raise ValueError(f"{path}: output throughput is not positive")
    return result


def sign(value: float) -> int:
    return (value > 0) - (value < 0)


def analyze_root(root: Path, baseline_variant: str) -> dict:
    provenance_path = root / "workload" / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance_sha = sha256(provenance_path)
    paths = sorted((root / "runs").glob("*/gsm8k_c*.json"))
    if not paths:
        raise ValueError("no fixed-token GSM8K timing results found")

    cells = []
    pairs = []
    seen = set()
    for gsm_path in paths:
        run_label = gsm_path.parent.name
        concurrency = int(gsm_path.stem.removeprefix("gsm8k_c"))
        random_path = gsm_path.with_name(f"random_c{concurrency}.json")
        if not random_path.exists():
            raise ValueError(f"missing random twin for {gsm_path}")
        gsm = checked_result(
            gsm_path,
            provenance,
            provenance_sha,
            run_label,
            "gsm8k",
            concurrency,
        )
        random = checked_result(
            random_path,
            provenance,
            provenance_sha,
            run_label,
            "random",
            concurrency,
        )
        if gsm["variant"] != random["variant"]:
            raise ValueError(f"{gsm_path}: twin variants differ")
        if gsm.get("run_index") != random.get("run_index"):
            raise ValueError(f"{gsm_path}: twin run indexes differ")
        run_index = int(gsm["run_index"]) if gsm.get("run_index") else None
        pair = {
            "run_label": run_label,
            "run_index": run_index,
            "variant": gsm["variant"],
            "concurrency": concurrency,
            "gsm8k_output_tokens_per_second": gsm["output_throughput"],
            "gsm8k_questions_per_second": gsm["request_throughput"],
            "random_output_tokens_per_second": random["output_throughput"],
            "random_questions_per_second": random["request_throughput"],
            "gsm8k_to_random_tps_ratio": (
                gsm["output_throughput"] / random["output_throughput"]
            ),
            "input_token_vector_sha256": vector_sha256(gsm["input_lens"]),
            "output_token_vector_sha256": vector_sha256(gsm["output_lens"]),
        }
        pairs.append(pair)
        for workload, path, result in (
            ("gsm8k", gsm_path, gsm),
            ("random", random_path, random),
        ):
            cells.append(
                {
                    "run_label": run_label,
                    "run_index": run_index,
                    "variant": result["variant"],
                    "workload": workload,
                    "concurrency": concurrency,
                    "raw_result": str(path.resolve()),
                    "raw_result_sha256": sha256(path),
                    "completed": result["completed"],
                    "total_input_tokens": result["total_input_tokens"],
                    "total_output_tokens": result["total_output_tokens"],
                    "questions_per_second": result["request_throughput"],
                    "output_tokens_per_second": result["output_throughput"],
                }
            )
        seen.add(random_path)
    extras = set((root / "runs").glob("*/random_c*.json")) - seen
    if extras:
        raise ValueError(f"random twins without GSM8K results: {sorted(extras)}")

    tracking = []
    concurrencies = sorted({pair["concurrency"] for pair in pairs})
    for concurrency in concurrencies:
        at_concurrency = [pair for pair in pairs if pair["concurrency"] == concurrency]
        baselines = sorted(
            (
                pair
                for pair in at_concurrency
                if pair["variant"] == baseline_variant and pair["run_index"] is not None
            ),
            key=lambda pair: pair["run_index"],
        )
        for candidate in at_concurrency:
            if candidate["variant"] == baseline_variant:
                continue
            row = {
                "run_label": candidate["run_label"],
                "run_index": candidate["run_index"],
                "variant": candidate["variant"],
                "concurrency": concurrency,
                "fit_to_random_control": None,
            }
            if candidate["run_index"] is None:
                row["fit_reason"] = "candidate_run_index_missing"
                tracking.append(row)
                continue
            before = [
                pair for pair in baselines if pair["run_index"] < candidate["run_index"]
            ]
            after = [
                pair for pair in baselines if pair["run_index"] > candidate["run_index"]
            ]
            if not before or not after:
                row["fit_reason"] = "candidate_not_bracketed_by_A_runs"
                tracking.append(row)
                continue
            left, right = before[-1], after[0]
            gsm_reference = math.sqrt(
                left["gsm8k_output_tokens_per_second"]
                * right["gsm8k_output_tokens_per_second"]
            )
            random_reference = math.sqrt(
                left["random_output_tokens_per_second"]
                * right["random_output_tokens_per_second"]
            )
            gsm_gain = candidate["gsm8k_output_tokens_per_second"] / gsm_reference - 1
            random_gain = (
                candidate["random_output_tokens_per_second"] / random_reference - 1
            )
            residual_pp = 100 * (gsm_gain - random_gain)
            aa_gsm_drift = (
                right["gsm8k_output_tokens_per_second"]
                / left["gsm8k_output_tokens_per_second"]
                - 1
            )
            aa_random_drift = (
                right["random_output_tokens_per_second"]
                / left["random_output_tokens_per_second"]
                - 1
            )
            aa_residual_pp = 100 * (aa_gsm_drift - aa_random_drift)
            signs_match = sign(gsm_gain) == sign(random_gain)
            inside_band = abs(residual_pp) <= abs(aa_residual_pp)
            fit = signs_match and inside_band
            row.update(
                {
                    "baseline_before": left["run_label"],
                    "baseline_after": right["run_label"],
                    "baseline_reference": "geometric_mean_of_bracketing_A_runs",
                    "gsm8k_gain_pp": 100 * gsm_gain,
                    "random_gain_pp": 100 * random_gain,
                    "gsm8k_minus_random_gain_residual_pp": residual_pp,
                    "gain_signs_match": signs_match,
                    "aa_gsm_drift_pp": 100 * aa_gsm_drift,
                    "aa_random_drift_pp": 100 * aa_random_drift,
                    "aa_gain_residual_pp": aa_residual_pp,
                    "observed_aa_residual_band_pp": abs(aa_residual_pp),
                    "residual_inside_observed_aa_band": inside_band,
                    "fit_to_random_control": fit,
                    "fit_reason": (
                        "signs_match_and_residual_inside_observed_AA_band"
                        if fit
                        else (
                            "gain_sign_mismatch"
                            if not signs_match
                            else "residual_exceeds_observed_AA_band"
                        )
                    ),
                }
            )
            tracking.append(row)

    return {
        "schema_version": 1,
        "workload_kind": "gsm8k_fixed_token_timing_twin_analysis",
        "baseline_variant": baseline_variant,
        "fit_rule": (
            "A candidate fits only when its GSM8K and matched-random gains have "
            "the same sign and their residual is no larger than the residual "
            "observed between the bracketing A runs."
        ),
        "provenance": str(provenance_path.resolve()),
        "provenance_sha256": provenance_sha,
        "natural_stop_gsm8k_health": provenance["natural_stop_gsm8k_health"],
        "all_pairs_have_identical_input_output_vectors": True,
        "cells": cells,
        "pairs": pairs,
        "gain_tracking": tracking,
    }


def analyze(args: argparse.Namespace) -> None:
    result = analyze_root(args.result_root, args.baseline_variant)
    output = args.output or args.result_root / "summary.json"
    write_json(output, result)
    print(json.dumps(result, sort_keys=True))


def self_test() -> None:
    class FakeTokenizer:
        vocab_size = 32
        all_special_ids = [0]

        def num_special_tokens_to_add(self) -> int:
            return 1

        def decode(self, tokens: list[int]) -> str:
            return ",".join(map(str, tokens))

        def encode(self, prompt: str, add_special_tokens: bool = True) -> list[int]:
            tokens = [int(token) for token in prompt.split(",")]
            return ([0] if add_special_tokens else []) + tokens

        def __call__(self, prompt: str) -> argparse.Namespace:
            return argparse.Namespace(input_ids=self.encode(prompt))

    random_prompts = make_random_prompts(FakeTokenizer(), [4, 7], 0)
    assert [len(FakeTokenizer()(prompt).input_ids) for prompt in random_prompts] == [
        4,
        7,
    ]

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        workload = root / "workload"
        workload.mkdir()
        provenance = {
            "num_requests": 2,
            "fixed_output_tokens": 256,
            "prompt_token_lengths": [3, 4],
            "source_details_sha256": "details",
            "datasets": {
                "gsm8k": {"sha256": "gsm"},
                "random": {"sha256": "random"},
            },
            "natural_stop_gsm8k_health": {
                "interpretation": "descriptive_only_not_matched_token_throughput"
            },
        }
        provenance_path = workload / "provenance.json"
        write_json(provenance_path, provenance)
        provenance_sha = sha256(provenance_path)

        def add_run(
            label: str,
            index: int,
            variant: str,
            gsm_tps: float,
            random_tps: float,
        ) -> None:
            directory = root / "runs" / label
            directory.mkdir(parents=True)
            for workload_name, throughput, dataset_sha in (
                ("gsm8k", gsm_tps, "gsm"),
                ("random", random_tps, "random"),
            ):
                result = {
                    "run_label": label,
                    "run_index": str(index),
                    "variant": variant,
                    "workload_kind": f"{workload_name}_fixed_token_timing",
                    "provenance_sha256": provenance_sha,
                    "dataset_sha256": dataset_sha,
                    "source_details_sha256": "details",
                    "ignore_eos": "true",
                    "fixed_output_tokens": "256",
                    "temperature": "0",
                    "num_warmups": "16",
                    "timing_prompt_count": "2",
                    "max_concurrency": 8,
                    "num_prompts": 2,
                    "completed": 2,
                    "failed": 0,
                    "input_lens": [3, 4],
                    "output_lens": [256, 256],
                    "generated_texts": ["x", "y"],
                    "errors": ["", ""],
                    "ttfts": [0.1, 0.1],
                    "itls": [[], []],
                    "output_throughput": throughput,
                    "request_throughput": throughput / 256,
                    "total_input_tokens": 7,
                    "total_output_tokens": 512,
                }
                prefix = "gsm8k" if workload_name == "gsm8k" else "random"
                write_json(directory / f"{prefix}_c8.json", result)

        add_run("a1", 1, "A", 100, 100)
        add_run("b1", 2, "B", 120, 119)
        add_run("a2", 3, "A", 102, 101)
        result = analyze_root(root, "A")
        tracking = result["gain_tracking"][0]
        assert tracking["baseline_before"] == "a1"
        assert tracking["baseline_after"] == "a2"
        assert tracking["gain_signs_match"]
        assert tracking["fit_to_random_control"]
        random_path = root / "runs" / "b1" / "random_c8.json"
        random = json.loads(random_path.read_text())
        random["input_lens"] = [4, 3]
        write_json(random_path, random)
        try:
            analyze_root(root, "A")
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched twin token vectors were accepted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--details", type=Path, required=True)
    prepare_parser.add_argument("--accuracy-summary", type=Path)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--tokenizer", required=True)
    prepare_parser.add_argument("--tokenizer-mode", default="auto")
    prepare_parser.add_argument("--trust-remote-code", action="store_true")
    prepare_parser.add_argument("--model-revision", default="")
    prepare_parser.add_argument("--source-revision", default="")
    prepare_parser.add_argument("--output-tokens", type=int, default=256)
    prepare_parser.add_argument("--seed", type=int, default=0)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--result-root", type=Path, required=True)
    analyze_parser.add_argument("--baseline-variant", required=True)
    analyze_parser.add_argument("--output", type=Path)
    commands.add_parser("self-test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "analyze":
        analyze(args)
    else:
        self_test()


if __name__ == "__main__":
    main()
