# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ruff: noqa: E402

import hashlib
import importlib.util
import json
import os
import runpy
import sys

import torch

root = os.environ["S39_ROOT"]
bootstrap_path = os.path.realpath(__file__)
with open(bootstrap_path, "rb") as bootstrap_file:
    bootstrap_sha256 = hashlib.sha256(bootstrap_file.read()).hexdigest()
print(
    "S39_BOOTSTRAP_PROVENANCE "
    + json.dumps(
        {
            "path": bootstrap_path,
            "sha256": bootstrap_sha256,
            "vllm_cache_root": os.environ.get("VLLM_CACHE_ROOT"),
        },
        sort_keys=True,
    ),
    flush=True,
)
from vllm import _custom_ops as ops

stable_libtorch = (
    "/usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so"
)
with open(stable_libtorch, "rb") as stable_file:
    stable_sha256 = hashlib.sha256(stable_file.read()).hexdigest()
print(
    "S39_STABLE_LIBTORCH_PROVENANCE "
    + json.dumps({"path": stable_libtorch, "sha256": stable_sha256}, sort_keys=True),
    flush=True,
)
if not hasattr(torch.ops._C, "marlin_nvfp4_to_fp8"):
    torch.ops.load_library(root + "/marlin_nvfp4_to_fp8_final.so")
ops.marlin_nvfp4_to_fp8 = torch.ops._C.marlin_nvfp4_to_fp8

from vllm.model_executor.layers.quantization.utils import marlin_utils

old_workspace = marlin_utils.marlin_make_workspace_new


def compat_workspace(device, max_blocks_per_sm=1, existing=None):
    if existing is None:
        return old_workspace(device, max_blocks_per_sm)
    size = marlin_utils.num_compute_units(device.index) * max_blocks_per_sm
    if (
        existing.device != device
        or existing.dtype != torch.int
        or existing.numel() != size
    ):
        raise ValueError("incompatible existing Marlin workspace")
    return existing.zero_()


marlin_utils.marlin_make_workspace_new = compat_workspace


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, root + "/" + filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


import importlib

old_workspace_module = importlib.import_module("vllm.v1.worker.workspace")
workspace_names = (
    "is_workspace_manager_initialized",
    "current_workspace_manager",
    "init_workspace_manager",
    "lock_workspace",
    "reserve_workspace_for_all_ubatches",
    "unlock_workspace",
    "reset_workspace_manager",
)
old_workspace_functions = {
    name: getattr(old_workspace_module, name)
    for name in workspace_names
    if hasattr(old_workspace_module, name)
}
current_workspace_module = load("vllm.v1.worker.workspace", "workspace.py")
for imported_module in list(sys.modules.values()):
    namespace = getattr(imported_module, "__dict__", None)
    if namespace is None:
        continue
    for name, old_function in old_workspace_functions.items():
        if namespace.get(name) is old_function:
            namespace[name] = getattr(current_workspace_module, name)
load("vllm.model_executor.layers.quantization.utils.nvfp4_utils", "nvfp4_utils.py")
load(
    "vllm.model_executor.layers.quantization.utils.marlin_utils_fp4",
    "marlin_utils_fp4.py",
)
import vllm.model_executor.kernels.linear as linear

load("vllm.model_executor.kernels.linear.nvfp4.base", "base.py")
load("vllm.model_executor.kernels.linear.nvfp4.marlin", "marlin.py")
dense = load("vllm.model_executor.kernels.linear.nvfp4.marlin_fp8", "marlin_fp8.py")
moe = load(
    "vllm.model_executor.layers.fused_moe.experts.nvfp4_bycopy_moe",
    "nvfp4_bycopy_moe.py",
)
load(
    "vllm.compilation.passes.utility.fix_functionalization", "fix_functionalization.py"
)

from vllm.model_executor.layers.quantization.utils import flashinfer_fp4_moe


def nvfp4_swizzled_scale_to_cutedsl_mma_view(scale):
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    num_experts, m_padded, k_sf_padded = scale.shape
    view = convert_sf_to_mma_layout(
        scale.reshape(num_experts * m_padded, k_sf_padded),
        m=m_padded,
        k=k_sf_padded * 16,
        num_groups=num_experts,
        sf_vec_size=16,
    )
    assert view.data_ptr() == scale.data_ptr()
    return view


flashinfer_fp4_moe.nvfp4_swizzled_scale_to_cutedsl_mma_view = (
    nvfp4_swizzled_scale_to_cutedsl_mma_view
)

oracle = load("vllm.model_executor.layers.fused_moe.oracle.nvfp4", "nvfp4.py")
current_make_moe_kernel = oracle.make_nvfp4_moe_kernel


def compat_make_moe_kernel(
    moe_quant_config, moe_config, experts_cls, routing_tables=None
):
    if not issubclass(experts_cls, moe.NvFp4ByCopyExperts):
        raise RuntimeError("campaign expected MARLIN_FP8_BYCOPY experts")
    return current_make_moe_kernel(
        moe_quant_config,
        moe_config,
        experts_cls,
        oracle.NvFp4MoeBackend.MARLIN_FP8_BYCOPY,
        routing_tables,
    )


oracle.make_nvfp4_moe_kernel = compat_make_moe_kernel
load("vllm.model_executor.layers.fused_moe.runner.shared_experts", "shared_experts.py")

original_init = linear.init_nvfp4_linear_kernel


def candidate_init(use_a16=False):
    if use_a16:
        supported, _ = dense.MarlinNvFp4ToFp8LinearKernel.is_supported()
        if supported:
            return dense.MarlinNvFp4ToFp8LinearKernel(dense.NvFp4LinearLayerConfig())
    return original_init(use_a16)


linear.MarlinNvFp4ToFp8LinearKernel = dense.MarlinNvFp4ToFp8LinearKernel
linear.init_nvfp4_linear_kernel = candidate_init

branch_counts = {"dense": {}, "moe": {}}
layer_type_counts = {}


def record_lookup(kind, dtype, shape, branch):
    key = (str(dtype), tuple(shape), branch)
    branch_counts[kind][key] = branch_counts[kind].get(key, 0) + 1


original_dense_lookup = dense._lookup_dense_m_knee


def universal_dense_lookup(device_name, dtype, shape):
    knee = original_dense_lookup(device_name, dtype, shape)
    assert knee == 512
    record_lookup("dense", dtype, shape, "universal")
    return knee


dense._lookup_dense_m_knee = universal_dense_lookup

original_layer_eligibility = dense._is_nvfp4_bycopy_layer


def record_layer_eligibility(layer):
    eligible = original_layer_eligibility(layer)
    key = (f"{type(layer).__module__}.{type(layer).__qualname__}", eligible)
    layer_type_counts[key] = layer_type_counts.get(key, 0) + 1
    return eligible


dense._is_nvfp4_bycopy_layer = record_layer_eligibility

original_moe_lookup = moe._lookup_moe_m_knee


def universal_moe_lookup(device_name, dtype, shape):
    knee = original_moe_lookup(device_name, dtype, shape)
    assert knee == 512
    record_lookup("moe", dtype, shape, "universal")
    return knee


moe._lookup_moe_m_knee = universal_moe_lookup

dense_alias_modules = (
    "vllm.model_executor.layers.quantization.modelopt",
    "vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4",
    "vllm.model_executor.layers.quantization.quark.schemes.quark_nvfp4",
)
for name in dense_alias_modules:
    module = sys.modules.get(name)
    if module is not None and hasattr(module, "init_nvfp4_linear_kernel"):
        module.init_nvfp4_linear_kernel = candidate_init

moe_alias_modules = (
    "vllm.model_executor.layers.quantization.modelopt",
    "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4",
    "vllm.model_executor.layers.quantization.quark.quark_moe",
    "vllm.model_executor.layers.quantization.online.nvfp4",
)
oracle_names = (
    "NvFp4MoeBackend",
    "select_nvfp4_moe_backend",
    "convert_to_nvfp4_moe_kernel_format",
    "is_global_sf_supported_for_nvfp4_backend",
    "make_nvfp4_moe_quant_config",
    "make_nvfp4_moe_kernel",
)
for name in moe_alias_modules:
    module = sys.modules.get(name)
    if module is not None:
        for attr in oracle_names:
            if hasattr(module, attr):
                setattr(module, attr, getattr(oracle, attr))

modelopt_module = importlib.import_module(
    "vllm.model_executor.layers.quantization.modelopt"
)
compressed_dense_module = importlib.import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4"
)
compressed_moe_module = importlib.import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4"
)
modelopt_module.init_nvfp4_linear_kernel = candidate_init
compressed_dense_module.init_nvfp4_linear_kernel = candidate_init
for module in (modelopt_module, compressed_moe_module):
    for attr in oracle_names:
        if hasattr(module, attr):
            setattr(module, attr, getattr(oracle, attr))

original_modelopt_w4a16_init = modelopt_module.ModelOptNvFp4W4A16LinearMethod.__init__


def campaign_modelopt_w4a16_init(self, *args, **kwargs):
    original_modelopt_w4a16_init(self, *args, **kwargs)
    self.kernel = candidate_init(True)


modelopt_module.ModelOptNvFp4W4A16LinearMethod.__init__ = campaign_modelopt_w4a16_init

manifest_files = (
    "marlin_nvfp4_to_fp8_final.so",
    "workspace.py",
    "nvfp4_utils.py",
    "marlin_utils_fp4.py",
    "base.py",
    "marlin.py",
    "marlin_fp8.py",
    "nvfp4_bycopy_moe.py",
    "fix_functionalization.py",
    "nvfp4.py",
    "shared_experts.py",
)
manifest = {}
for filename in manifest_files:
    with open(root + "/" + filename, "rb") as file:
        manifest[filename] = hashlib.sha256(file.read()).hexdigest()
print("S39_CURRENT_MANIFEST " + json.dumps(manifest, sort_keys=True), flush=True)
print("S39_OVERLAY_IMPORT_PASS", flush=True)
try:
    runpy.run_path(os.environ["S39_RUNNER"], run_name="__main__")
finally:
    serialized_counts = {
        kind: [
            {
                "dtype": dtype,
                "shape": list(shape),
                "branch": branch,
                "count": count,
            }
            for (dtype, shape, branch), count in sorted(counts.items())
        ]
        for kind, counts in branch_counts.items()
    }
    print(
        "S39_BRANCH_COUNTS " + json.dumps(serialized_counts, sort_keys=True),
        flush=True,
    )
    serialized_layer_types = [
        {"type": layer_type, "eligible": eligible, "count": count}
        for (layer_type, eligible), count in sorted(layer_type_counts.items())
    ]
    print(
        "S39_LAYER_TYPE_COUNTS " + json.dumps(serialized_layer_types, sort_keys=True),
        flush=True,
    )
