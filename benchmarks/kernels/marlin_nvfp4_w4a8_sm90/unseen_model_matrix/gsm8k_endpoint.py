#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from functools import partial
from pathlib import Path

import regex as re

INVALID = -9_999_999


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if not line.startswith("#")]


def answer_value(text: str) -> int:
    numbers = re.findall(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?", text)
    if not numbers:
        return INVALID
    try:
        value = Decimal(numbers[-1].replace(",", ""))
        return int(value) if value == value.to_integral_value() else INVALID
    except InvalidOperation:
        return INVALID


def exact_mcnemar(n01: int, n10: int) -> float:
    discordant = n01 + n10
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(n01, n10) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_summary(baseline: list[dict], candidate: list[dict]) -> dict:
    identity = ("question_id", "question", "prompt", "gold_answer")
    baseline_items = [tuple(row[key] for key in identity) for row in baseline]
    candidate_items = [tuple(row[key] for key in identity) for row in candidate]
    if baseline_items != candidate_items:
        raise ValueError("baseline and candidate GSM8K items differ")
    n01 = sum(
        (not left["correct"]) and right["correct"]
        for left, right in zip(baseline, candidate)
    )
    n10 = sum(
        left["correct"] and (not right["correct"])
        for left, right in zip(baseline, candidate)
    )
    total = len(baseline)
    return {
        "n": total,
        "n01_baseline_wrong_candidate_right": n01,
        "n10_baseline_right_candidate_wrong": n10,
        "baseline_correct": sum(row["correct"] for row in baseline),
        "candidate_correct": sum(row["correct"] for row in candidate),
        "delta_percentage_points": 100.0 * (n01 - n10) / total,
        "exact_two_sided_mcnemar_p": exact_mcnemar(n01, n10),
    }


def build_prompts(
    train: list[dict], test: list[dict], num_questions: int
) -> tuple[list[str], list[int]]:
    examples = "".join(
        f"Question: {row['question']}\nAnswer: {row['answer']}\n\n" for row in train[:5]
    )
    selected = test[:num_questions]
    prompts = [examples + f"Question: {row['question']}\nAnswer:" for row in selected]
    labels = [answer_value(row["answer"]) for row in selected]
    assert all(label != INVALID for label in labels)
    return prompts, labels


def request_completion(prompt: str, *, url: str, model: str, timeout: float) -> dict:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": 256,
            "seed": 42,
            "stop": ["Question", "Assistant:", "<|separator|>"],
        }
    ).encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    choice = result["choices"][0]
    return {
        "generated_answer": choice.get("text") or "",
        "completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
        "finish_reason": choice.get("finish_reason"),
        "request_seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--url")
    parser.add_argument("--model")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--variant")
    parser.add_argument("--num-questions", type=int, default=1319)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--baseline-details", type=Path)
    return parser.parse_args()


def self_test() -> None:
    assert answer_value("The answer is 1,234.") == 1234
    assert answer_value("The answer is -1,234.0.") == -1234
    assert answer_value("The answer is 12.5.") == INVALID
    assert answer_value("none") == INVALID
    assert exact_mcnemar(0, 0) == 1.0
    baseline = [
        {
            "question_id": "0",
            "question": "q0",
            "prompt": "p0",
            "gold_answer": 0,
            "correct": False,
        },
        {
            "question_id": "1",
            "question": "q1",
            "prompt": "p1",
            "gold_answer": 1,
            "correct": True,
        },
    ]
    candidate = [
        {**baseline[0], "correct": True},
        {**baseline[1], "correct": True},
    ]
    assert paired_summary(baseline, candidate)["delta_percentage_points"] == 50
    candidate[0]["prompt"] = "different"
    try:
        paired_summary(baseline, candidate)
    except ValueError:
        pass
    else:
        raise AssertionError("paired summary accepted different GSM8K items")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    required = (
        "url",
        "model",
        "data_dir",
        "output_dir",
        "variant",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"missing arguments: {', '.join(missing)}")

    train = read_jsonl(args.data_dir / "train.jsonl")
    test = read_jsonl(args.data_dir / "test.jsonl")
    prompts, labels = build_prompts(train, test, args.num_questions)
    call = partial(
        request_completion,
        url=args.url,
        model=args.model,
        timeout=args.timeout,
    )
    started = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=min(args.max_concurrency, len(prompts))
    ) as pool:
        responses = list(pool.map(call, prompts))
    latency = time.perf_counter() - started

    details = []
    for index, (prompt, label, response) in enumerate(zip(prompts, labels, responses)):
        prediction = answer_value(response["generated_answer"])
        details.append(
            {
                "question_id": f"gsm8k-test-{index:04d}",
                "question": test[index]["question"],
                "prompt": prompt,
                "gold_answer": label,
                "parsed_answer": None if prediction == INVALID else prediction,
                "correct": prediction == label,
                "error": "unparsable_answer" if prediction == INVALID else None,
                **response,
            }
        )

    correct = sum(row["correct"] for row in details)
    invalid = sum(row["parsed_answer"] is None for row in details)
    output_tokens = sum(row["completion_tokens"] for row in details)
    summary = {
        "model": args.model,
        "model_revision": args.model_revision or None,
        "variant": args.variant,
        "num_questions": len(details),
        "correct": correct,
        "accuracy": correct / len(details),
        "invalid": invalid,
        "invalid_rate": invalid / len(details),
        "latency": latency,
        "total_output_tokens": output_tokens,
        "tokens_per_second": output_tokens / latency,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / "details.jsonl"
    details_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in details),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("UNSEEN_GSM8K " + json.dumps(summary, sort_keys=True), flush=True)

    if args.baseline_details:
        baseline = read_jsonl(args.baseline_details)
        paired = paired_summary(baseline, details)
        paired["model"] = args.model
        (args.output_dir.parent / "paired.summary.json").write_text(
            json.dumps(paired, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "UNSEEN_GSM8K_PAIRED " + json.dumps(paired, sort_keys=True),
            flush=True,
        )


if __name__ == "__main__":
    main()
