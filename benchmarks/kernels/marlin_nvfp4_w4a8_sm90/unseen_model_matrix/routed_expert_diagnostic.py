#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import io
import json
import os
import tempfile
import urllib.request
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
from pathlib import Path

import numpy as np
import pybase64 as base64

LIMITATION = (
    "The API payload has no scheduler-step boundaries, runtime M, or selected "
    "runtime branch; this diagnostic cannot confirm exact P/R or a knee decision."
)


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.startswith("#"):
                yield json.loads(line)


def read_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_sha256(prompts: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def capture(
    *,
    url: str,
    model: str,
    rows: list[dict],
    output: Path,
    concurrency: int,
    timeout: float,
) -> None:
    endpoint = f"{url.rstrip('/')}/v1/completions"

    def send(item: tuple[int, dict]) -> dict:
        index, row = item
        payload = {
            "model": model,
            "prompt": row["prompt"],
            "temperature": 0,
            "max_tokens": 256,
            "seed": 42,
            "stop": ["Question", "Assistant:", "<|separator|>"],
            "stream": False,
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {
                "item": index,
                "question_id": row["question_id"],
                "response": json.load(response),
            }

    with (
        output.open("w", encoding="utf-8") as raw,
        ThreadPoolExecutor(max_workers=concurrency) as pool,
    ):
        items = iter(enumerate(rows))
        pending = {pool.submit(send, item) for item in islice(items, concurrency)}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                raw.write(json.dumps(future.result(), separators=(",", ":")) + "\n")
                if item := next(items, None):
                    pending.add(pool.submit(send, item))


def analyze(responses: Path, experts: int, expected_top_k: int) -> dict:
    aggregate_slots = None
    observed = None
    items = set()
    choices = 0
    tokens = 0
    for line_number, record in enumerate(iter_jsonl(responses), 1):
        item = record["item"]
        if item in items:
            raise ValueError(f"duplicate response item {item}")
        items.add(item)
        response = record["response"]
        for choice_index, choice in enumerate(response["choices"]):
            encoded = choice.get("routed_experts")
            if encoded is None:
                raise ValueError(
                    f"line {line_number}, choice {choice_index}: no routes"
                )
            routes = np.load(io.BytesIO(base64.b64decode(encoded)), allow_pickle=False)
            if routes.ndim != 3 or not np.issubdtype(routes.dtype, np.integer):
                raise ValueError(
                    f"line {line_number}, choice {choice_index}: expected integer "
                    f"[tokens,layers,top_k], got {routes.shape} {routes.dtype}"
                )
            if routes.shape[2] != expected_top_k:
                raise ValueError(
                    f"line {line_number}, choice {choice_index}: expected top-k "
                    f"{expected_top_k}, got {routes.shape[2]}"
                )
            if not routes.size or routes.min() < 0 or routes.max() >= experts:
                raise ValueError(
                    f"line {line_number}, choice {choice_index}: expert id outside "
                    f"[0,{experts}) or empty routes"
                )

            route_tokens, layers, top_k = routes.shape
            slots = np.empty((layers, top_k, experts), dtype=np.int64)
            for layer in range(layers):
                for rank in range(top_k):
                    slots[layer, rank] = np.bincount(
                        routes[:, layer, rank], minlength=experts
                    )
            if aggregate_slots is None:
                aggregate_slots = np.zeros_like(slots)
                observed = np.zeros(layers, dtype=np.bool_)
            if aggregate_slots.shape != slots.shape:
                raise ValueError("all responses must have the same layer/top-k shape")
            aggregate_slots += slots
            observed |= np.any(routes != 0, axis=(0, 2))
            choices += 1
            tokens += route_tokens

    if aggregate_slots is None or observed is None:
        raise ValueError("no routed-expert responses found")
    observed_layers = np.flatnonzero(observed).tolist()
    if not observed_layers:
        raise ValueError("no observed MoE layers found")
    layer_rows = []
    for layer in observed_layers:
        slots = aggregate_slots[layer]
        counts = slots.sum(axis=0)
        if counts.sum() != tokens * expected_top_k:
            raise AssertionError("route counts do not sum to tokens * top_k")
        layer_rows.append(
            {
                "layer": layer,
                "expert_counts": counts.tolist(),
                "topk_slot_expert_counts": slots.tolist(),
            }
        )
    return {
        "experts": experts,
        "top_k": expected_top_k,
        "num_api_responses": len(items),
        "num_choices": choices,
        "returned_route_tokens": tokens,
        "observed_moe_layer_indices": observed_layers,
        "zero_filled_layer_indices": np.flatnonzero(~observed).tolist(),
        "observed_layer_rule": (
            "include only layers with at least one nonzero returned expert ID"
        ),
        "layers": layer_rows,
        "limitations": [LIMITATION],
    }


def total_variation(left: list[int], right: list[int]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if (
        left_array.shape != right_array.shape
        or not left_array.sum()
        or not right_array.sum()
    ):
        raise ValueError("distribution counts must have matching nonempty shapes")
    return float(
        0.5
        * np.abs(left_array / left_array.sum() - right_array / right_array.sum()).sum()
    )


def distribution_drift(baseline: dict, candidate: dict) -> dict:
    if baseline["variant"] != "native_reference":
        raise ValueError("route-distribution baseline must be native_reference (A)")
    comparable = (
        "model",
        "model_revision",
        "retained_details_sha256",
        "prompt_corpus_sha256",
        "experts",
        "top_k",
    )
    if any(baseline[key] != candidate[key] for key in comparable):
        raise ValueError("route summaries do not describe the same prompt/shape")
    baseline_layers = {row["layer"]: row for row in baseline["layers"]}
    candidate_layers = {row["layer"]: row for row in candidate["layers"]}
    if baseline_layers.keys() != candidate_layers.keys():
        raise ValueError("A and candidate observed different MoE layer sets")

    rows = []
    for layer, current in candidate_layers.items():
        control = baseline_layers[layer]
        slot_drift = [
            total_variation(left, right)
            for left, right in zip(
                control["topk_slot_expert_counts"],
                current["topk_slot_expert_counts"],
                strict=True,
            )
        ]
        rows.append(
            {
                "layer": layer,
                "expert_distribution_total_variation": total_variation(
                    control["expert_counts"], current["expert_counts"]
                ),
                "topk_slot_total_variation": slot_drift,
            }
        )
    expert_drift = [row["expert_distribution_total_variation"] for row in rows]
    slot_drift = [value for row in rows for value in row["topk_slot_total_variation"]]
    return {
        "A_variant": "native_reference",
        "metric": "total variation distance on normalized expert counts",
        "per_layer": rows,
        "mean_layer_expert_total_variation": float(np.mean(expert_drift)),
        "max_layer_expert_total_variation": max(expert_drift),
        "max_layer_topk_slot_total_variation": max(slot_drift),
    }


def self_test() -> None:
    routes = np.array(
        [
            [[0, 0], [0, 1], [1, 2]],
            [[0, 0], [1, 2], [2, 0]],
        ],
        dtype=np.uint8,
    )
    buffer = io.BytesIO()
    np.save(buffer, routes)
    record = {
        "item": 0,
        "response": {
            "choices": [
                {"routed_experts": base64.b64encode(buffer.getvalue()).decode()}
            ]
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "raw.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        result = analyze(path, experts=3, expected_top_k=2)
    assert result["observed_moe_layer_indices"] == [1, 2]
    assert result["zero_filled_layer_indices"] == [0]
    assert result["layers"][0]["expert_counts"] == [1, 2, 1]
    summary = {
        **result,
        "model": "model",
        "model_revision": "revision",
        "variant": "native_reference",
        "retained_details_sha256": "details",
        "prompt_corpus_sha256": "corpus",
    }
    assert distribution_drift(summary, summary)["max_layer_expert_total_variation"] == 0
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay retained GSM8K prompts and summarize returned MoE routes."
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--url")
    parser.add_argument("--model")
    parser.add_argument("--model-revision", default="")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="retained native_reference GSM8K details.jsonl",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--variant")
    parser.add_argument("--num-questions", type=int, default=1319)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--baseline-details",
        type=Path,
        help="native_reference route summary.json used as A",
    )
    parser.add_argument("--experts", type=int, default=os.environ.get("ROUTE_EXPERTS"))
    parser.add_argument("--top-k", type=int, default=os.environ.get("ROUTE_TOP_K"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    required = ("url", "model", "data_dir", "output_dir", "variant", "experts", "top_k")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"missing arguments: {', '.join(missing)}")
    if args.num_questions < 1 or args.max_concurrency < 1:
        raise SystemExit("num-questions and max-concurrency must be positive")
    if args.experts < 1 or args.top_k < 1:
        raise SystemExit("experts and top-k must be positive")
    if args.variant != "native_reference" and not args.baseline_details:
        raise SystemExit("non-reference diagnostics require --baseline-details")

    rows = read_jsonl(args.data_dir)
    if len(rows) != args.num_questions:
        raise ValueError(
            f"retained corpus has {len(rows)} rows, expected {args.num_questions}"
        )
    if any("question_id" not in row or "prompt" not in row for row in rows):
        raise ValueError("retained corpus rows require question_id and prompt")
    prompts = [row["prompt"] for row in rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = args.output_dir / "raw_responses.jsonl"
    capture(
        url=args.url,
        model=args.model,
        rows=rows,
        output=raw,
        concurrency=min(args.max_concurrency, len(rows)),
        timeout=args.timeout,
    )
    summary = {
        "model": args.model,
        "model_revision": args.model_revision or None,
        "variant": args.variant,
        "workload_kind": "gsm8k_route_diagnostic_untimed",
        "matched_timing": False,
        "route_scope": (
            "retained prompt plus generated continuation; distributions are "
            "normalized because continuation lengths may differ"
        ),
        "max_concurrency": args.max_concurrency,
        "num_prompts": len(rows),
        "retained_details_sha256": sha256_file(args.data_dir),
        "prompt_corpus_sha256": prompt_sha256(prompts),
        "raw_responses_sha256": sha256_file(raw),
        **analyze(raw, args.experts, args.top_k),
        "distribution_drift_vs_A": None,
    }
    if args.baseline_details:
        baseline = json.loads(args.baseline_details.read_text(encoding="utf-8"))
        summary["distribution_drift_vs_A"] = distribution_drift(baseline, summary)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("UNSEEN_ROUTE_DIAGNOSTIC " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
