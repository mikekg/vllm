# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ruff: noqa: E402

import ast
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

VARIANTS = {
    "marlin": {
        "mode": "marlin",
        "policy": "baseline_control",
        "selector": "marlin",
        "target_r": None,
        "target_divisor": None,
        "code": None,
    },
    "adaptive": {
        "mode": "aligned_adaptive",
        "policy": "aligned_universal512_factor_stress",
        "selector": "universal_512",
        "target_r": None,
        "target_divisor": None,
        "code": None,
    },
    "r1": {
        "mode": "fixed",
        "policy": "experimental_below_fixed768_floor",
        "selector": "universal_512",
        "target_r": 1.0,
        "target_divisor": 128.0,
        "code": 0xB0,
    },
    "sqrt6": {
        "mode": "fixed",
        "policy": "experimental_below_fixed768_floor",
        "selector": "universal_512",
        "target_r": 6.0**0.5,
        "target_divisor": 128.0 * 6.0**0.5,
        "code": 0xBA,
    },
    "r6": {
        "mode": "fixed",
        "policy": "required_fixed768_floor",
        "selector": "universal_512",
        "target_r": 6.0,
        "target_divisor": 768.0,
        "code": 0xC4,
    },
    "adaptive_prod": {
        "mode": "aligned_adaptive",
        "policy": "aligned_universal512_production_normal",
        "selector": "universal_512_production_normal",
        "target_r": None,
        "target_divisor": None,
        "code": None,
    },
}
INVALID = -9999999
variant_name = os.environ["S39_VARIANT"]
variant = VARIANTS[variant_name]

if variant_name == "marlin":
    os.environ["VLLM_TEST_FORCE_FP8_MARLIN"] = "1"
    disabled = [
        value
        for value in os.environ.get("VLLM_DISABLED_KERNELS", "").split(",")
        if value
    ]
    if "MarlinNvFp4ToFp8LinearKernel" not in disabled:
        disabled.append("MarlinNvFp4ToFp8LinearKernel")
    os.environ["VLLM_DISABLED_KERNELS"] = ",".join(disabled)

import torch

so_path = Path(os.environ["S39_SO"])
expected_so_sha = os.environ["S39_SO_SHA256"]
with so_path.open("rb") as file:
    so_sha = hashlib.file_digest(file, "sha256").hexdigest()
if so_sha != expected_so_sha:
    raise RuntimeError(f"SO SHA mismatch: {so_sha} != {expected_so_sha}")

campaign_manifest_path = Path(os.environ["S39_MANIFEST"])
campaign_manifest = json.loads(campaign_manifest_path.read_text(encoding="utf-8"))
source_root = Path(os.environ["S39_ROOT"])
for recorded_path, expected_sha in campaign_manifest["source_files"].items():
    source_path = source_root / Path(recorded_path).name
    with source_path.open("rb") as file:
        source_sha = hashlib.file_digest(file, "sha256").hexdigest()
    if source_sha != expected_sha:
        raise RuntimeError(
            f"source SHA mismatch for {source_path}: {source_sha} != {expected_sha}"
        )

converter_sha = campaign_manifest["runtime"]["converter_so"]["sha256"]
if so_sha != converter_sha:
    raise RuntimeError(f"converter SHA mismatch: {so_sha} != {converter_sha}")
stable_libtorch = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so"
)
with stable_libtorch.open("rb") as file:
    stable_sha = hashlib.file_digest(file, "sha256").hexdigest()
expected_stable_sha = campaign_manifest["runtime"]["abi_so"]["sha256"]
if stable_sha != expected_stable_sha:
    raise RuntimeError(
        f"stable libtorch SHA mismatch: {stable_sha} != {expected_stable_sha}"
    )
if not hasattr(torch.ops._C, "marlin_nvfp4_to_fp8"):
    torch.ops.load_library(str(so_path))

import vllm
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4


def restore_marlin_baseline():
    import vllm.model_executor.kernels.linear as linear
    from vllm.model_executor.layers.fused_moe.oracle import nvfp4 as oracle
    from vllm.model_executor.layers.quantization import modelopt

    candidate_init = linear.init_nvfp4_linear_kernel
    original_init = candidate_init.__globals__.get("original_init")
    if not callable(original_init):
        raise RuntimeError("compatibility bootstrap did not expose original init")
    linear.init_nvfp4_linear_kernel = original_init
    for name in candidate_init.__globals__["dense_alias_modules"]:
        module = sys.modules.get(name)
        if (
            module is not None
            and getattr(module, "init_nvfp4_linear_kernel", None) is candidate_init
        ):
            module.init_nvfp4_linear_kernel = original_init

    patched_method_init = modelopt.ModelOptNvFp4W4A16LinearMethod.__init__
    original_method_init = patched_method_init.__globals__.get(
        "original_modelopt_w4a16_init"
    )
    if not callable(original_method_init):
        raise RuntimeError("compatibility bootstrap did not expose ModelOpt init")
    modelopt.ModelOptNvFp4W4A16LinearMethod.__init__ = original_method_init

    compat_make_moe_kernel = oracle.make_nvfp4_moe_kernel
    original_make_moe_kernel = compat_make_moe_kernel.__globals__.get(
        "current_make_moe_kernel"
    )
    if not callable(original_make_moe_kernel):
        raise RuntimeError("compatibility bootstrap did not expose MoE factory")
    oracle.make_nvfp4_moe_kernel = original_make_moe_kernel
    for name in compat_make_moe_kernel.__globals__["moe_alias_modules"]:
        module = sys.modules.get(name)
        if (
            module is not None
            and getattr(module, "make_nvfp4_moe_kernel", None) is compat_make_moe_kernel
        ):
            module.make_nvfp4_moe_kernel = original_make_moe_kernel


def restore_production_knees():
    from vllm.model_executor.kernels.linear.nvfp4 import marlin_fp8
    from vllm.model_executor.layers.fused_moe.experts import nvfp4_bycopy_moe

    for module, name, original_name in (
        (
            marlin_fp8,
            "_lookup_dense_m_knee",
            "original_dense_lookup",
        ),
        (
            nvfp4_bycopy_moe,
            "_lookup_moe_m_knee",
            "original_moe_lookup",
        ),
    ):
        measured_lookup = getattr(module, name)
        original_lookup = measured_lookup.__globals__.get(original_name)
        if not callable(original_lookup):
            raise RuntimeError(f"compatibility bootstrap did not expose {name}")
        setattr(module, name, original_lookup)


if variant_name == "marlin":
    restore_marlin_baseline()
    restore_production_knees()
elif variant["selector"] == "universal_512_production_normal":
    restore_production_knees()

metadata = {
    "calls": 0,
    "code_histogram": Counter(),
    "total_codes": 0,
}
original_codes = marlin_utils_fp4._nvfp4_tile_scale_divisor_codes


def record_codes(codes):
    histogram = torch.bincount(codes.reshape(-1).to(torch.int64), minlength=256).cpu()
    metadata["calls"] += 1
    metadata["total_codes"] += codes.numel()
    metadata["code_histogram"].update(
        {code: count for code, count in enumerate(histogram.tolist()) if count}
    )
    return codes


if variant["mode"] == "fixed":
    fixed_code = variant["code"]

    def campaign_codes(packed_weight, block_scales, scale_factor):
        del block_scales, scale_factor
        n = packed_weight.size(-2)
        k = packed_weight.size(-1) * 2
        shape = (
            *packed_weight.shape[:-2],
            (n + 127) // 128,
            (k + 127) // 128,
        )
        return record_codes(
            torch.full(
                shape,
                fixed_code,
                dtype=torch.uint8,
                device=packed_weight.device,
            )
        )

elif variant["mode"] == "aligned_adaptive":

    def campaign_codes(packed_weight, block_scales, scale_factor):
        codes = original_codes(packed_weight, block_scales, scale_factor)
        if torch.any((codes != 0x78) & ((codes & 7) != 4)):
            raise RuntimeError("source overlay did not emit exponent-aligned codes")
        return record_codes(codes)


else:

    def campaign_codes(packed_weight, block_scales, scale_factor):
        return record_codes(original_codes(packed_weight, block_scales, scale_factor))


marlin_utils_fp4._nvfp4_tile_scale_divisor_codes = campaign_codes

if os.environ.get("S39_BOOTSTRAP_ONLY") == "1":
    print(f"S39_CAMPAIGN_BOOTSTRAP_PASS {variant_name}", flush=True)
    raise SystemExit(0)


def decode_code(code):
    if code is None:
        return None
    bits = torch.tensor([code << 7], dtype=torch.int16)
    return float(bits.view(torch.float16).item())


decoded_divisor = decode_code(variant["code"])
decoded_r = None if decoded_divisor is None else decoded_divisor / 128.0
if variant_name == "r1":
    assert (decoded_divisor, decoded_r) == (128.0, 1.0)
elif variant_name == "sqrt6":
    assert (decoded_divisor, decoded_r) == (320.0, 2.5)
elif variant_name == "r6":
    assert (decoded_divisor, decoded_r) == (768.0, 6.0)


def read_jsonl(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def answer_value(text):
    numbers = re.findall(r"\d+", text.replace(",", ""))
    if not numbers:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID


data_dir = Path(os.environ["S39_DATA"])
train = read_jsonl(data_dir / "train.jsonl")
start_index = int(os.environ.get("S39_START_INDEX", "0"))
stop_index = int(os.environ.get("S39_STOP_INDEX", "1319"))
test = read_jsonl(data_dir / "test.jsonl")[start_index:stop_index]
gen_prefix = (
    " <think>\n\n</think>\n" if os.environ.get("S39_GEN_PREFIX") == "qwen36" else ""
)
examples = "".join(
    "Question: {}\nAnswer:{} {}\n\n".format(row["question"], gen_prefix, row["answer"])
    for row in train[:5]
)
prompts = [
    examples + "Question: {}\nAnswer:{}".format(row["question"], gen_prefix)
    for row in test
]
labels = [answer_value(row["answer"]) for row in test]
max_num_seqs = int(os.environ.get("S39_MAX_NUM_SEQS", "64"))
max_tokens = int(os.environ.get("S39_MAX_TOKENS", "256"))
max_model_len = int(os.environ.get("S39_MAX_MODEL_LEN", "2048"))

print(
    "S39_CAMPAIGN_CONFIG "
    + json.dumps(
        {
            "variant": variant_name,
            "variant_config": variant,
            "decoded_divisor": decoded_divisor,
            "decoded_r": decoded_r,
            "model": os.environ["S39_MODEL"],
            "model_id": os.environ["S39_MODEL_ID"],
            "model_revision": os.environ["S39_MODEL_REVISION"],
            "num_questions": len(prompts),
            "start_index": start_index,
            "stop_index": stop_index,
            "max_num_seqs": max_num_seqs,
            "max_tokens": max_tokens,
            "max_model_len": max_model_len,
            "engine_seed": 0,
            "sampling_seed": 42,
            "temperature": 0.0,
            "run_id": os.environ["S39_RUN_ID"],
            "source_tag": os.environ["S39_SOURCE_TAG"],
            "vllm_cache_root": os.environ["VLLM_CACHE_ROOT"],
            "so_sha256": so_sha,
        },
        sort_keys=True,
    ),
    flush=True,
)

llm = LLM(
    model=os.environ["S39_MODEL"],
    dtype="bfloat16",
    max_model_len=max_model_len,
    max_num_seqs=max_num_seqs,
    gpu_memory_utilization=0.92,
    disable_log_stats=True,
    seed=0,
)
tokenizer = llm.get_tokenizer()
prompt_lengths = [len(tokenizer.encode(prompt)) for prompt in prompts]
if prompt_lengths and max(prompt_lengths) + max_tokens > max_model_len:
    raise RuntimeError(
        (
            max(prompt_lengths),
            max_tokens,
            max_model_len,
        )
    )
prompt_aggregate_sha = hashlib.sha256("".join(prompts).encode()).hexdigest()
print(
    "S39_CAMPAIGN_PROMPTS "
    + json.dumps(
        {
            "aggregate_sha256": prompt_aggregate_sha,
            "count": len(prompts),
            "min_tokens": min(prompt_lengths),
            "max_tokens": max(prompt_lengths),
            "mean_tokens": sum(prompt_lengths) / len(prompt_lengths),
        },
        sort_keys=True,
    ),
    flush=True,
)

params = SamplingParams(
    temperature=0,
    max_tokens=max_tokens,
    seed=42,
    stop=["Question", "Assistant:", "<|separator|>"],
)
profile = os.environ.get("S39_PROFILE", "0") == "1"
native_events = {}
torch.cuda.synchronize()
start = time.perf_counter()
if profile:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        outputs = llm.generate(prompts, params, use_tqdm=True)
    native_events = {
        event.key: event.count
        for event in profiler.key_averages()
        if "marlin_nvfp4_to_fp8" in event.key
    }
else:
    outputs = llm.generate(prompts, params, use_tqdm=True)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

records = []
for offset, (prompt, label, output) in enumerate(zip(prompts, labels, outputs)):
    text = output.outputs[0].text
    prediction = answer_value(text)
    records.append(
        {
            "index": start_index + offset,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "label": label,
            "prediction": prediction,
            "correct": prediction == label,
            "invalid": prediction == INVALID,
            "completion_tokens": len(output.outputs[0].token_ids),
            "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "response": text,
        }
    )

artifact = Path(os.environ["S39_OUTPUT"])
artifact.parent.mkdir(parents=True, exist_ok=True)
payload = "".join(
    json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n" for record in records
)
artifact.write_text(payload, encoding="utf-8")
artifact_sha = hashlib.sha256(payload.encode()).hexdigest()
correct = sum(record["correct"] for record in records)
invalid = sum(record["invalid"] for record in records)
total_output_tokens = sum(record["completion_tokens"] for record in records)
manifest_path = Path(os.environ["S39_MANIFEST"])
with manifest_path.open("rb") as file:
    manifest_sha = hashlib.file_digest(file, "sha256").hexdigest()
histogram = {
    f"0x{code:02x}": count for code, count in sorted(metadata["code_histogram"].items())
}
result = {
    "result": "pass",
    "variant": variant_name,
    "variant_config": variant,
    "decoded_divisor": decoded_divisor,
    "decoded_r": decoded_r,
    "model": os.environ["S39_MODEL"],
    "model_id": os.environ["S39_MODEL_ID"],
    "model_revision": os.environ["S39_MODEL_REVISION"],
    "num_questions": len(records),
    "correct": correct,
    "accuracy": correct / len(records),
    "invalid": invalid,
    "invalid_rate": invalid / len(records),
    "total_output_tokens": total_output_tokens,
    "elapsed_seconds": elapsed,
    "questions_per_second": len(records) / elapsed,
    "tokens_per_second": total_output_tokens / elapsed,
    "num_shots": 5,
    "max_num_seqs": max_num_seqs,
    "max_tokens": max_tokens,
    "max_model_len": max_model_len,
    "engine_seed": 0,
    "sampling_seed": 42,
    "temperature": 0,
    "run_id": os.environ["S39_RUN_ID"],
    "source_tag": os.environ["S39_SOURCE_TAG"],
    "vllm_cache_root": os.environ["VLLM_CACHE_ROOT"],
    "gen_prefix": gen_prefix,
    "prompt_aggregate_sha256": prompt_aggregate_sha,
    "prompt_tokens": {
        "min": min(prompt_lengths),
        "max": max(prompt_lengths),
        "mean": sum(prompt_lengths) / len(prompt_lengths),
    },
    "metadata_calls": metadata["calls"],
    "metadata_total_codes": metadata["total_codes"],
    "metadata_code_histogram": histogram,
    "native_events": native_events,
    "artifact": str(artifact),
    "artifact_sha256": artifact_sha,
    "manifest_sha256": manifest_sha,
    "so_sha256": so_sha,
    "torch": torch.__version__,
    "vllm": vllm.__version__,
}
summary = Path(os.environ["S39_SUMMARY"])
summary.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
summary_sha = hashlib.sha256(summary.read_bytes()).hexdigest()
print(
    "S39_CAMPAIGN_RESULT " + json.dumps(result, sort_keys=True),
    flush=True,
)
print(
    "S39_CAMPAIGN_ARTIFACTS "
    + json.dumps(
        {
            "items": str(artifact),
            "items_sha256": artifact_sha,
            "summary": str(summary),
            "summary_sha256": summary_sha,
        },
        sort_keys=True,
    ),
    flush=True,
)

if profile:
    if len(records) != 1:
        raise RuntimeError("smoke must contain exactly one prompt")
    if records[0]["invalid"] or not records[0]["response"].strip():
        raise RuntimeError(f"incoherent smoke: {records[0]!r}")
    if variant_name == "marlin" and native_events:
        raise RuntimeError(f"Marlin baseline invoked converter: {native_events}")
    if variant_name != "marlin" and not native_events:
        raise RuntimeError(f"{variant_name} did not invoke native converter")
if variant["mode"] == "fixed" and set(histogram) != {f"0x{variant['code']:02x}"}:
    raise RuntimeError(f"fixed-code override was not exclusive: {histogram}")
