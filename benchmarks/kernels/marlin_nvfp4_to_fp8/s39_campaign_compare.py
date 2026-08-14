# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import itertools
import json
import os
from pathlib import Path

MODELS = ("llama", "q36d", "q36m", "q3m")
VARIANTS = ("marlin", "adaptive", "r1", "sqrt6", "r6", "adaptive_prod")
POLICY = {
    "marlin": "baseline_control",
    "adaptive": "aligned_universal512_factor_stress",
    "r1": "experimental_below_fixed768_floor",
    "sqrt6": "experimental_below_fixed768_floor",
    "r6": "required_fixed768_floor",
    "adaptive_prod": "aligned_universal512_production_normal",
}
results_dir = Path(os.environ["S39_CAMPAIGN_RESULTS"])


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path):
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line
    ]
    if len(rows) != 1319:
        raise RuntimeError(f"{path}: expected 1319 rows, got {len(rows)}")
    if [row["index"] for row in rows] != list(range(1319)):
        raise RuntimeError(f"{path}: non-canonical indices")
    return rows


def pair_counts(left, right):
    counts = {
        "both_correct": 0,
        "both_wrong": 0,
        "left_correct_right_wrong": 0,
        "right_correct_left_wrong": 0,
        "same_prediction": 0,
        "different_prediction": 0,
        "same_response": 0,
        "different_response": 0,
    }
    for a, b in zip(left, right):
        if a["correct"] and b["correct"]:
            counts["both_correct"] += 1
        elif a["correct"]:
            counts["left_correct_right_wrong"] += 1
        elif b["correct"]:
            counts["right_correct_left_wrong"] += 1
        else:
            counts["both_wrong"] += 1
        key = (
            "same_prediction"
            if a["prediction"] == b["prediction"]
            else "different_prediction"
        )
        counts[key] += 1
        key = (
            "same_response"
            if a["response_sha256"] == b["response_sha256"]
            else "different_response"
        )
        counts[key] += 1
    return counts


campaign = {
    "schema_version": 1,
    "policy": POLICY,
    "models": {},
}
for model in MODELS:
    rows = {}
    inputs = {}
    for variant in VARIANTS:
        path = results_dir / f"{model}__{variant}.jsonl"
        rows[variant] = load_jsonl(path)
        inputs[variant] = {
            "path": str(path),
            "sha256": sha256(path),
        }

    reference = rows["marlin"]
    for variant in VARIANTS[1:]:
        candidate = rows[variant]
        for index, (a, b) in enumerate(zip(reference, candidate)):
            if (
                a["index"] != b["index"]
                or a["label"] != b["label"]
                or a["prompt_sha256"] != b["prompt_sha256"]
            ):
                raise RuntimeError(
                    f"{model}/{variant}: item invariant failed at {index}"
                )

    variant_counts = {
        variant: {
            "correct": sum(row["correct"] for row in variant_rows),
            "accuracy": sum(row["correct"] for row in variant_rows) / len(variant_rows),
            "invalid": sum(row["invalid"] for row in variant_rows),
            "completion_tokens": sum(row["completion_tokens"] for row in variant_rows),
            "policy": POLICY[variant],
        }
        for variant, variant_rows in rows.items()
    }
    pairs = {}
    for left, right in itertools.combinations(VARIANTS, 2):
        pairs[f"{left}__vs__{right}"] = {
            "left": left,
            "right": right,
            **pair_counts(rows[left], rows[right]),
        }

    flips = []
    for index in range(1319):
        correctness = {
            variant: bool(rows[variant][index]["correct"]) for variant in VARIANTS
        }
        if len(set(correctness.values())) == 1:
            continue
        reference_row = rows["marlin"][index]
        flips.append(
            {
                "index": index,
                "label": reference_row["label"],
                "prompt_sha256": reference_row["prompt_sha256"],
                "variants": {
                    variant: {
                        "prediction": rows[variant][index]["prediction"],
                        "correct": rows[variant][index]["correct"],
                        "invalid": rows[variant][index]["invalid"],
                        "completion_tokens": rows[variant][index]["completion_tokens"],
                        "response_sha256": rows[variant][index]["response_sha256"],
                    }
                    for variant in VARIANTS
                },
            }
        )

    flip_path = results_dir / f"paired_flips__{model}.jsonl"
    flip_payload = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in flips
    )
    flip_path.write_text(flip_payload, encoding="utf-8")
    model_summary = {
        "model": model,
        "n": 1319,
        "inputs": inputs,
        "variant_counts": variant_counts,
        "pairs": pairs,
        "items_with_any_correctness_flip": len(flips),
        "flip_artifact": {
            "path": str(flip_path),
            "sha256": hashlib.sha256(flip_payload.encode()).hexdigest(),
        },
        "item_invariants": {
            "indices_identical": True,
            "labels_identical": True,
            "prompt_sha256_identical_per_item": True,
        },
    }
    summary_path = results_dir / f"paired_summary__{model}.json"
    summary_path.write_text(
        json.dumps(model_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model_summary["summary_artifact"] = {
        "path": str(summary_path),
        "sha256": sha256(summary_path),
    }
    campaign["models"][model] = model_summary
    print(
        "S39_PAIR_ACCOUNTING "
        + json.dumps(
            {
                "model": model,
                "variant_counts": variant_counts,
                "pairs": pairs,
                "flip_artifact": model_summary["flip_artifact"],
                "summary_artifact": model_summary["summary_artifact"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

campaign_path = results_dir / "campaign_pair_accounting.json"
campaign_path.write_text(
    json.dumps(campaign, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(
    "S39_CAMPAIGN_ACCOUNTING "
    + json.dumps(
        {
            "path": str(campaign_path),
            "sha256": sha256(campaign_path),
        },
        sort_keys=True,
    ),
    flush=True,
)
