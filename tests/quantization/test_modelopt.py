# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test ModelOpt quantization method setup and weight loading.

Run `pytest tests/quantization/test_modelopt.py`.
"""

import os
from typing import Any, NoReturn
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from tests.quantization.utils import (
    is_quant_method_supported,
    load_model_without_vllm_runner,
)
from vllm import envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.model import ModelConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.kernels.linear import (
    FlashInferCuteDslNvFp4W4A16LinearKernel,
    HummingNvFp4LinearKernel,
    MarlinNvFp4LinearKernel,
    MarlinNvFp4ToFp8LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4 import marlin_fp8
from vllm.model_executor.kernels.linear.nvfp4.marlin_fp8 import (
    _is_nvfp4_bycopy_layer,
    _lookup_dense_m_knee,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.modelopt import (
    LINEAR_ALGOS,
    ModelOptFp8Config,
    ModelOptLinearMethod,
    ModelOptMixedPrecisionConfig,
    ModelOptMxFp8Config,
    ModelOptNvFp4Config,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_pad_dim,
    marlin_permute_bias,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    kFp8StaticTensorSym,
    kMxfp8Dynamic,
    kMxfp8Static,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform


@pytest.fixture(scope="function", autouse=True)
def enable_pickle(monkeypatch):
    """`LLM.apply_model` requires pickling a function."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def _skip(msg: str) -> NoReturn:
    pytest.skip(msg)
    raise RuntimeError(msg)


def _snapshot_download_or_skip(model_id: str) -> str:
    try:
        from vllm.transformers_utils.repo_utils import hf_api
    except Exception as e:  # pragma: no cover
        _skip(f"huggingface_hub is required to download {model_id}: {e}")

    try:
        return hf_api().snapshot_download(
            repo_id=model_id,
            repo_type="model",
            # These checkpoints are already small; download full repo for simplicity.
            allow_patterns=["*"],
        )
    except Exception as e:
        _skip(f"Failed to download {model_id} from the HF Hub: {e}")


def _mock_lm_head() -> Mock:
    lm_head = Mock(spec=ParallelLMHead)
    lm_head.__class__ = ParallelLMHead
    return lm_head


def _mixed_precision_config(quantized_layers: dict) -> ModelOptMixedPrecisionConfig:
    return ModelOptMixedPrecisionConfig(
        kv_cache_quant_method=None,
        exclude_modules=[],
        quantized_layers=quantized_layers,
        fp8_config=ModelOptFp8Config(
            quant_method="FP8",
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=None,
            exclude_modules=[],
        ),
        nvfp4_config=ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        ),
        w4a16_nvfp4_config=ModelOptNvFp4Config(
            quant_method="W4A16_NVFP4",
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        ),
        mxfp8_config=ModelOptMxFp8Config(
            is_checkpoint_mxfp8_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        ),
    )


def test_modelopt_nvfp4_quantizes_parallel_lm_head():
    config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )

    method = config.get_quant_method(_mock_lm_head(), prefix="lm_head")

    assert isinstance(method, ModelOptLinearMethod)
    assert method.spec.weight is kNvfp4Static
    assert method.spec.activation is kNvfp4Dynamic


def test_modelopt_fp8_updates_weight_dims_after_transpose():
    """Humming reads weight.input_dim/output_dim. Swapping the
    ModelWeightParameter for a plain Parameter drops them, so the per-tensor
    FP8 scheme must restore them for the transposed [in, out] layout.
    """
    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.layers.quantization.modelopt import (
        SCHEME_FOR,
        CkptCtx,
        FormatScheme,
    )

    layer = torch.nn.Module()
    layer.register_parameter(
        "weight", torch.nn.Parameter(torch.empty(3, 2), requires_grad=False)
    )
    layer.register_parameter(
        "weight_scale", torch.nn.Parameter(torch.ones(1), requires_grad=False)
    )
    layer.register_parameter(
        "input_scale", torch.nn.Parameter(torch.ones(1), requires_grad=False)
    )
    layer.logical_widths = [3]

    method = ModelOptLinearMethod.__new__(ModelOptLinearMethod)
    method.spec = QuantSpec(weight=kFp8StaticTensorSym, activation=kFp8StaticTensorSym)
    method.ctx = CkptCtx()
    method.fmt = FormatScheme()
    method.wkey = SCHEME_FOR[kFp8StaticTensorSym]
    method.akey = SCHEME_FOR[kFp8StaticTensorSym]
    method.kernel = Mock()
    method.process_weights_after_loading(layer)

    assert layer.weight.shape == (2, 3)
    assert layer.weight.input_dim == 0
    assert layer.weight.output_dim == 1
    method.kernel.process_weights_after_loading.assert_called_once_with(layer)


def test_modelopt_linear_algos_table_matches_resolve():
    """LINEAR_ALGOS is the single source of truth for supported linear algos.

    Every entry must be dispatchable by resolve(), and every config's
    validation list must be derived from it -- so adding a format is one row
    here plus one row in resolve(), with nothing else to keep in sync.
    """
    from vllm.model_executor.layers.quantization.modelopt import (
        QUANT_ALGOS,
        algos_owned_by,
        resolve,
    )

    class _Cfg:
        group_size = 16

    for algo in LINEAR_ALGOS:
        spec, _, _ = resolve(algo, _Cfg(), "layer")
        assert spec.weight is not None, algo

    assert list(QUANT_ALGOS) == [*LINEAR_ALGOS, "MIXED_PRECISION"]
    assert algos_owned_by("modelopt") == (
        "FP8",
        "FP8_PER_CHANNEL_PER_TOKEN",
        "FP8_PB_WO",
    )
    assert algos_owned_by("modelopt_fp4") == ("NVFP4", "W4A16_NVFP4")
    assert algos_owned_by("modelopt_mxfp8") == ("MXFP8",)


@pytest.mark.parametrize("algo", list(LINEAR_ALGOS))
def test_modelopt_mixed_precision_dispatches_every_linear_algo(algo):
    """Mixed precision must route every algo in LINEAR_ALGOS through the
    generic method. FP8_PER_CHANNEL_PER_TOKEN and FP8_PB_WO used to fall
    through to UnquantizedLinearMethod, which loses the checkpoint's scales.
    """
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization import modelopt as m

    config = m.ModelOptMixedPrecisionConfig.from_config(
        {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": None,
                "exclude_modules": [],
                "group_size": 16,
                "quantized_layers": {
                    "model.layers.0.mlp.down_proj": {"quant_algo": algo}
                },
            }
        }
    )
    method = config.get_quant_method(
        MagicMock(spec=LinearBase), "model.layers.0.mlp.down_proj"
    )

    assert isinstance(method, ModelOptLinearMethod), (algo, type(method).__name__)


def test_modelopt_nvfp4_leaves_excluded_parallel_lm_head_unquantized():
    config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=["lm_head"],
    )

    method = config.get_quant_method(_mock_lm_head(), prefix="lm_head")

    assert isinstance(method, UnquantizedLinearMethod)


def test_modelopt_mixed_precision_quantizes_parallel_lm_head():
    config = _mixed_precision_config(
        {"lm_head": {"quant_algo": "NVFP4", "group_size": 16}}
    )

    method = config.get_quant_method(_mock_lm_head(), prefix="lm_head")

    assert isinstance(method, ModelOptLinearMethod)
    assert method.spec.weight is kNvfp4Static
    assert method.spec.activation is kNvfp4Dynamic


def test_modelopt_mixed_precision_resolves_declared_packed_projection():
    config = _mixed_precision_config(
        {
            "model.layers.0.self_attn.q_proj": {"quant_algo": "MXFP8"},
            "model.layers.0.self_attn.k_proj": {"quant_algo": "MXFP8"},
            "model.layers.0.self_attn.v_proj": {"quant_algo": "MXFP8"},
        }
    )
    config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    assert config._resolve_quant_algo("model.layers.0.self_attn.qkv_proj") == "MXFP8"


def test_modelopt_mixed_precision_does_not_quantize_unlisted_fused_sibling():
    config = _mixed_precision_config(
        {
            "model.layers.0.linear_attn.in_proj_qkv": {"quant_algo": "FP8"},
            "model.layers.0.linear_attn.in_proj_z": {"quant_algo": "FP8"},
            "model.layers.0.linear_attn.out_proj": {"quant_algo": "FP8"},
        }
    )
    config.packed_modules_mapping = {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    assert (
        config._resolve_quant_algo("model.layers.0.linear_attn.in_proj_qkvz") == "FP8"
    )
    assert config._resolve_quant_algo("model.layers.0.linear_attn.in_proj_ba") is None


def test_modelopt_mixed_precision_composes_gemma4_mappers():
    from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
    from vllm.model_executor.models.gemma4_mm import (
        Gemma4ForConditionalGeneration,
    )

    config = _mixed_precision_config(
        {
            "model.language_model.layers.0.experts": {
                "quant_algo": "NVFP4",
                "group_size": 16,
            },
            "model.language_model.layers.1.moe.experts.gate_up_proj": {
                "quant_algo": "NVFP4",
                "group_size": 16,
            },
        }
    )

    config.apply_vllm_mapper(
        Gemma4ForConditionalGeneration.hf_to_vllm_mapper.get_rename_mapper()
    )
    config.apply_vllm_mapper(Gemma4ForCausalLM.hf_to_vllm_mapper.get_rename_mapper())

    expected_prefix = "language_model.model.layers.0.moe.experts"
    assert set(config.quantized_layers) == {
        expected_prefix,
        "language_model.model.layers.1.moe.gate_up_proj",
    }
    assert config._resolve_quant_algo(expected_prefix) == "NVFP4"


def test_modelopt_mixed_precision_infers_fused_gate_up_projection():
    from vllm.model_executor.layers.linear import LinearBase

    config = _mixed_precision_config(
        {
            "model.layers.0.mlp.gate_proj": {"quant_algo": "NVFP4"},
            "model.layers.0.mlp.up_proj": {"quant_algo": "NVFP4"},
        }
    )

    fake_layer = MagicMock(spec=LinearBase)
    method = config.get_quant_method(fake_layer, "model.layers.0.mlp.gate_up_proj")

    assert isinstance(method, ModelOptLinearMethod)
    assert method.spec.weight is kNvfp4Static
    assert method.spec.activation is kNvfp4Dynamic


@pytest.mark.parametrize(
    ("quantized_prefix", "missing_prefix"),
    [
        ("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.down_proj"),
        ("model.layers.0.self_attn.o_proj", "model.layers.0.self_attn.qkv_proj"),
    ],
)
def test_modelopt_mixed_precision_does_not_infer_missing_sibling_linear(
    quantized_prefix, missing_prefix
):
    from vllm.model_executor.layers.linear import LinearBase

    config = _mixed_precision_config(
        {
            quantized_prefix: {"quant_algo": "NVFP4"},
        }
    )

    fake_layer = MagicMock(spec=LinearBase)
    method = config.get_quant_method(fake_layer, missing_prefix)

    assert isinstance(method, UnquantizedLinearMethod)


def test_vocab_parallel_embedding_weight_loader_accepts_scalar_scale():
    holder = Mock()
    scale = torch.nn.Parameter(torch.empty(1))
    loaded_scale = torch.tensor(2.0)

    VocabParallelEmbedding.weight_loader(holder, scale, loaded_scale)

    assert torch.equal(scale, loaded_scale.reshape(1))


@pytest.mark.skipif(
    not is_quant_method_supported("modelopt"),
    reason="ModelOpt FP8 is not supported on this GPU type.",
)
def test_modelopt_fp8_checkpoint_setup(default_vllm_config, vllm_runner):
    """Test ModelOpt FP8 checkpoint loading and structure validation."""
    # TODO: provide a small publicly available test checkpoint
    model_path = (
        "/home/scratch.omniml_data_1/zhiyu/ckpts/test_ckpts/"
        "TinyLlama-1.1B-Chat-v1.0-fp8-0710"
    )

    # Skip test if checkpoint doesn't exist
    if not os.path.exists(model_path):
        pytest.skip(
            f"Test checkpoint not found at {model_path}. "
            "This test requires a local ModelOpt FP8 checkpoint."
        )

    # Set model config as model_config.dtype is required in ModelOptLinearMethod.
    default_vllm_config.model_config = ModelConfig()
    with vllm_runner(model_path, quantization="modelopt", enforce_eager=True) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj
            o_proj = layer.self_attn.o_proj
            gate_up_proj = layer.mlp.gate_up_proj
            down_proj = layer.mlp.down_proj

            # Check that ModelOpt quantization method is properly applied
            assert isinstance(qkv_proj.quant_method, ModelOptLinearMethod)
            assert isinstance(o_proj.quant_method, ModelOptLinearMethod)
            assert isinstance(gate_up_proj.quant_method, ModelOptLinearMethod)
            assert isinstance(down_proj.quant_method, ModelOptLinearMethod)

            # Check weight dtype is FP8
            assert qkv_proj.weight.dtype == torch.float8_e4m3fn
            assert o_proj.weight.dtype == torch.float8_e4m3fn
            assert gate_up_proj.weight.dtype == torch.float8_e4m3fn
            assert down_proj.weight.dtype == torch.float8_e4m3fn

            # Check scales are present and have correct dtype
            assert hasattr(qkv_proj, "weight_scale")
            assert hasattr(qkv_proj, "input_scale")
            assert qkv_proj.weight_scale.dtype == torch.float32
            assert qkv_proj.input_scale.dtype == torch.float32

            assert hasattr(o_proj, "weight_scale")
            assert hasattr(o_proj, "input_scale")
            assert o_proj.weight_scale.dtype == torch.float32
            assert o_proj.input_scale.dtype == torch.float32

            assert hasattr(gate_up_proj, "weight_scale")
            assert hasattr(gate_up_proj, "input_scale")
            assert gate_up_proj.weight_scale.dtype == torch.float32
            assert gate_up_proj.input_scale.dtype == torch.float32

            assert hasattr(down_proj, "weight_scale")
            assert hasattr(down_proj, "input_scale")
            assert down_proj.weight_scale.dtype == torch.float32
            assert down_proj.input_scale.dtype == torch.float32

        llm.apply_model(check_model)

        # Run a simple generation test to ensure the model works
        output = llm.generate_greedy(["Hello my name is"], max_tokens=4)
        assert output
        print(f"ModelOpt FP8 output: {output}")


@pytest.mark.skipif(
    not is_quant_method_supported("modelopt"),
    reason="ModelOpt FP8 is not supported on this GPU type.",
)
def test_modelopt_fp8_pc_pt_checkpoint_setup(monkeypatch, dist_init, workspace_init):
    """Test ModelOpt FP8_PER_CHANNEL_PER_TOKEN checkpoint setup."""
    model_id = "CedricHwang/qwen2.5-0.5b-modelopt-fp8-pc-pt"
    model_path = _snapshot_download_or_skip(model_id)

    model, vllm_config = load_model_without_vllm_runner(
        model_path,
        quantization="modelopt",
    )
    layer = model.model.layers[0]

    qkv_proj = layer.self_attn.qkv_proj
    o_proj = layer.self_attn.o_proj
    gate_up_proj = layer.mlp.gate_up_proj
    down_proj = layer.mlp.down_proj

    assert isinstance(qkv_proj.quant_method, ModelOptLinearMethod)
    assert isinstance(o_proj.quant_method, ModelOptLinearMethod)
    assert isinstance(gate_up_proj.quant_method, ModelOptLinearMethod)
    assert isinstance(down_proj.quant_method, ModelOptLinearMethod)

    fp8_dtype = current_platform.fp8_dtype()
    assert qkv_proj.weight.dtype == fp8_dtype
    assert o_proj.weight.dtype == fp8_dtype
    assert gate_up_proj.weight.dtype == fp8_dtype
    assert down_proj.weight.dtype == fp8_dtype

    # Per-channel scales; activations are dynamically scaled per token.
    for projection in (qkv_proj, o_proj, gate_up_proj, down_proj):
        assert hasattr(projection, "weight_scale")
        assert projection.weight_scale.dtype == torch.float32
        assert projection.weight_scale.dim() == 1
        assert not hasattr(projection, "input_scale")

    monkeypatch.setattr(Attention, "forward", lambda _, q, k, v: q.contiguous())
    input_ids = torch.tensor([1, 2, 3, 4], device=current_platform.device_type)
    positions = torch.arange(input_ids.numel(), device=current_platform.device_type)
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(None, vllm_config, num_tokens=input_ids.numel()),
    ):
        hidden_states = model(input_ids, positions, None)
        logits = model.compute_logits(hidden_states)
    assert torch.isfinite(logits).all()


@pytest.mark.skipif(
    not is_quant_method_supported("modelopt"),
    reason="ModelOpt FP8 is not supported on this GPU type.",
)
def test_modelopt_fp8_pb_wo_checkpoint_setup(monkeypatch, dist_init, workspace_init):
    """Test ModelOpt FP8_PB_WO checkpoint setup."""
    model_id = "CedricHwang/qwen2.5-0.5b-modelopt-fp8-pb-wo"
    model_path = _snapshot_download_or_skip(model_id)

    model, vllm_config = load_model_without_vllm_runner(
        model_path,
        quantization="modelopt",
    )
    layer = model.model.layers[0]

    qkv_proj = layer.self_attn.qkv_proj
    o_proj = layer.self_attn.o_proj
    gate_up_proj = layer.mlp.gate_up_proj
    down_proj = layer.mlp.down_proj

    assert isinstance(qkv_proj.quant_method, ModelOptLinearMethod)
    assert isinstance(o_proj.quant_method, ModelOptLinearMethod)
    assert isinstance(gate_up_proj.quant_method, ModelOptLinearMethod)
    assert isinstance(down_proj.quant_method, ModelOptLinearMethod)

    fp8_dtype = current_platform.fp8_dtype()
    assert qkv_proj.weight.dtype == fp8_dtype
    assert o_proj.weight.dtype == fp8_dtype
    assert gate_up_proj.weight.dtype == fp8_dtype
    assert down_proj.weight.dtype == fp8_dtype

    # Block scales are materialized as a 2D [out_blk, in_blk] tensor.
    for projection in (qkv_proj, o_proj, gate_up_proj, down_proj):
        assert hasattr(projection, "weight_scale")
        assert projection.weight_scale.dtype == torch.float32
        assert projection.weight_scale.dim() == 2

    monkeypatch.setattr(Attention, "forward", lambda _, q, k, v: q.contiguous())
    input_ids = torch.tensor([1, 2, 3, 4], device=current_platform.device_type)
    positions = torch.arange(input_ids.numel(), device=current_platform.device_type)
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(None, vllm_config, num_tokens=input_ids.numel()),
    ):
        hidden_states = model(input_ids, positions, None)
        logits = model.compute_logits(hidden_states)
    assert torch.isfinite(logits).all()


def test_modelopt_nvfp4_config_dispatches_w4a4_method():
    """``quant_method="NVFP4"`` (W4A4) resolves to a
    ``(kNvfp4Static, kNvfp4Dynamic)`` QuantSpec under the generic
    ``ModelOptLinearMethod``."""
    from vllm.model_executor.layers.linear import LinearBase

    config = ModelOptNvFp4Config(
        quant_method="NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    assert config.quant_method == "NVFP4"

    method = config.get_quant_method(
        MagicMock(spec=LinearBase), "model.layers.0.fake_proj"
    )
    assert isinstance(method, ModelOptLinearMethod)
    assert method.spec.weight is kNvfp4Static
    assert method.spec.activation is kNvfp4Dynamic


def test_modelopt_nvfp4_config_dispatches_w4a16_method():
    """``quant_method="W4A16_NVFP4"`` resolves to a weight-only QuantSpec
    (``activation=None``) — distinct from the W4A4 sibling.

    A regression here would mean a W4A16 NVFP4 checkpoint silently loaded
    with a dynamic fp4 activation key, registering an ``input_scale`` and
    routing to the cutlass W4A4 NVFP4 GEMM instead of FP4 Marlin.
    """
    from vllm.model_executor.layers.linear import LinearBase

    config = ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    assert config.quant_method == "W4A16_NVFP4"

    method = config.get_quant_method(
        MagicMock(spec=LinearBase), "model.layers.0.fake_proj"
    )
    assert isinstance(method, ModelOptLinearMethod)
    assert method.spec.weight is kNvfp4Static
    assert method.spec.activation is None


def test_modelopt_linear_method_builder_registry_override(monkeypatch):
    """The bespoke-method escape hatch: a format registered in
    ``LINEAR_METHOD_BUILDERS`` routes that algo to its own method instead of the
    generic ``ModelOptLinearMethod``. This is how a format that cannot be a
    ``(weight, activation)`` key pair plugs into dispatch."""
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization import modelopt as m

    sentinel = object()
    monkeypatch.setitem(m.LINEAR_METHOD_BUILDERS, "NVFP4", lambda cfg, prefix: sentinel)

    config = ModelOptNvFp4Config(
        quant_method="NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    method = config.get_quant_method(
        MagicMock(spec=LinearBase), "model.layers.0.fake_proj"
    )
    assert method is sentinel  # bespoke builder wins over the generic path


@pytest.mark.parametrize(
    ("linear_backend", "kernel_cls"),
    [
        ("auto", MarlinNvFp4ToFp8LinearKernel),
        ("marlin", MarlinNvFp4LinearKernel),
        ("humming", HummingNvFp4LinearKernel),
        ("flashinfer_cutedsl", FlashInferCuteDslNvFp4W4A16LinearKernel),
    ],
)
@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
def test_modelopt_w4a16_respects_linear_backend(linear_backend, kernel_cls):
    """W4A16 (`activation=None`) kernel selection honors ``--linear-backend``:
    ``use_a16=True`` defaults to Marlin, but an explicit backend wins. The
    generic method routes this through ``select_linear_kernel``."""
    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.layers.quantization.modelopt import (
        RuntimeDtypes,
        select_linear_kernel,
    )

    if linear_backend == "auto":
        cc = current_platform.get_device_capability()
        compute_capability = cc.to_int() if cc is not None else None
        cutedsl_ok, _ = FlashInferCuteDslNvFp4W4A16LinearKernel.is_supported(
            compute_capability
        )
        if compute_capability in (100, 103) and cutedsl_ok:
            kernel_cls = FlashInferCuteDslNvFp4W4A16LinearKernel
        else:
            bycopy_ok, _ = MarlinNvFp4ToFp8LinearKernel.is_supported()
            if (
                not bycopy_ok
                or MarlinNvFp4ToFp8LinearKernel.__name__
                in envs.VLLM_DISABLED_KERNELS
            ):
                kernel_cls = MarlinNvFp4LinearKernel
    else:
        is_supported, reason = kernel_cls.is_supported()
        if not is_supported:
            pytest.skip(reason)
    vllm_config = VllmConfig()
    vllm_config.kernel_config.linear_backend = linear_backend
    spec = QuantSpec(weight=kNvfp4Static, activation=None)
    rt = RuntimeDtypes(torch.bfloat16, torch.bfloat16)
    with set_current_vllm_config(vllm_config):
        kernel = select_linear_kernel(spec, MagicMock(), rt)
    assert isinstance(kernel, kernel_cls)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
def test_modelopt_linear_exposes_humming_layer_attrs(dist_init, monkeypatch):
    """``prepare_humming_linear_layer_config`` reads ``output_partition_sizes``
    and ``has_bias`` straight off the layer, so ``--linear-backend=humming``
    needs create_weights to leave both there. Nothing else sets
    ``output_partition_sizes``; ``LinearBase`` sets ``has_bias`` but
    ``ParallelLMHead`` does not.
    """
    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.layers.quantization import modelopt as mo

    monkeypatch.setattr(mo, "select_linear_kernel", lambda spec, layer, rt: Mock())
    monkeypatch.setattr(mo, "expose_input_quant_key", lambda layer, kernel: None)

    def build(layer):
        method = ModelOptLinearMethod.__new__(ModelOptLinearMethod)
        method.spec = QuantSpec(weight=kNvfp4Static, activation=None)
        method.ctx = mo.CkptCtx(group_size=16)
        method.fmt = mo.FormatScheme()
        method.wkey = mo.SCHEME_FOR[kNvfp4Static]
        method.akey = None
        method.input_dtype = method.out_dtype = torch.bfloat16
        method.marlin_input_dtype = None
        method.create_weights(
            layer, 64, [32, 32], 64, 64, torch.bfloat16, weight_loader=Mock()
        )

    # ParallelLMHead-style: a bias slot but no has_bias attribute.
    lm_head = torch.nn.Module()
    lm_head.register_parameter("bias", None)
    build(lm_head)
    assert lm_head.output_partition_sizes == [32, 32]
    assert lm_head.has_bias is False

    # LinearBase already decided has_bias; we must not overwrite it.
    linear = torch.nn.Module()
    linear.has_bias = True
    build(linear)
    assert linear.has_bias is True


def test_nvfp4_bycopy_knee_is_universal_except_embedding_types():
    assert _lookup_dense_m_knee("unknown GPU", torch.float16, (123, 456)) == 512
    assert _lookup_dense_m_knee("another GPU", torch.bfloat16, (789, 64)) == 512

    embedding = Mock(spec=VocabParallelEmbedding)
    embedding.__class__ = VocabParallelEmbedding
    assert _is_nvfp4_bycopy_layer(Mock(spec=[]))
    assert not _is_nvfp4_bycopy_layer(embedding)
    assert not _is_nvfp4_bycopy_layer(_mock_lm_head())


def test_nvfp4_bycopy_production_selection_stays_hopper_only():
    assert MarlinNvFp4ToFp8LinearKernel.is_supported(89) == (
        False,
        "Marlin NVFP4-to-FP8 requires Hopper",
    )


def test_nvfp4_bycopy_k64_pads_scratch_without_fallback(monkeypatch):
    with (
        patch.object(
            MarlinNvFp4ToFp8LinearKernel,
            "is_supported",
            return_value=(True, None),
        ),
        patch.object(marlin_fp8, "QuantFP8"),
    ):
        kernel = MarlinNvFp4ToFp8LinearKernel(Mock())

    layer = Mock(spec=[])
    layer.output_size_per_partition = 128
    layer.input_size_per_partition = 64
    layer.weight = torch.empty((4, 256), dtype=torch.int32)
    layer.params_dtype = torch.bfloat16
    kernel.marlin.process_weights_after_loading = Mock()
    reserve = Mock()
    monkeypatch.setattr(marlin_fp8, "marlin_repacked_nk", lambda *args: (128, 64))
    monkeypatch.setattr(marlin_fp8.current_platform, "get_device_name", Mock())
    monkeypatch.setattr(marlin_fp8, "reserve_workspace_for_all_ubatches", reserve)

    kernel.process_weights_after_loading(layer)

    assert kernel.resident_k == 64
    assert kernel.padded_k == 128
    assert kernel.m_knee == 512
    reserve.assert_called_once_with(128 * 128 + 256)


def test_nvfp4_bycopy_dense_uses_block_fp8_contract(monkeypatch):
    with (
        patch.object(
            MarlinNvFp4ToFp8LinearKernel,
            "is_supported",
            return_value=(True, None),
        ),
        patch.object(
            MarlinNvFp4LinearKernel, "is_supported", return_value=(True, None)
        ),
        patch.object(marlin_fp8, "QuantFP8") as quant_cls,
    ):
        kernel = MarlinNvFp4ToFp8LinearKernel(Mock())

    quant_cls.assert_called_once_with(
        static=False,
        group_shape=GroupShape(1, 128),
        column_major_scales=True,
        use_ue8m0=False,
    )

    kernel.m_knee = 1
    kernel.logical_n, kernel.logical_k = 64, 192
    kernel.padded_n, kernel.padded_k = 128, 256
    fp8_weight = torch.empty((128, 256), dtype=torch.float8_e4m3fn)
    weight_scale = torch.empty((1, 2), dtype=torch.float32)
    workspace = Mock()
    workspace.get_simultaneous.return_value = (fp8_weight, weight_scale)
    monkeypatch.setattr(marlin_fp8, "current_workspace_manager", lambda: workspace)

    quantized = torch.empty((2, 256), dtype=torch.float8_e4m3fn)
    activation_scale = torch.empty((2, 2), dtype=torch.float32)
    kernel.quant_fp8.return_value = (quantized, activation_scale)
    composite = Mock(return_value=torch.zeros((2, 128), dtype=torch.bfloat16))
    monkeypatch.setattr(
        torch.ops.vllm,
        "marlin_nvfp4_to_fp8_block_scaled_mm",
        composite,
    )

    layer = Mock(spec=[])
    layer.weight = torch.empty((16, 256), dtype=torch.int32)
    layer.weight_scale = torch.empty((16, 128), dtype=torch.float8_e4m3fn)
    layer.weight_global_scale = torch.ones((), dtype=torch.float32)
    layer.weight_fp8_scale_divisor_code = torch.ones((1, 2), dtype=torch.uint8)
    layer.params_dtype = torch.bfloat16
    x = torch.ones((2, 192), dtype=torch.bfloat16)

    bias = torch.arange(64, dtype=torch.bfloat16)
    output = kernel.apply_weights(layer, x, bias)

    workspace.get_simultaneous.assert_called_once_with(
        ((128, 256), torch.float8_e4m3fn), ((1, 2), torch.float32)
    )
    padded_x = kernel.quant_fp8.call_args.args[0]
    assert padded_x.shape == (2, 256)
    assert torch.equal(padded_x[:, :192], x)
    assert not torch.count_nonzero(padded_x[:, 192:])
    composite.assert_called_once_with(
        fp8_weight,
        weight_scale,
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        layer.weight_fp8_scale_divisor_code,
        torch.bfloat16,
        quantized,
        activation_scale,
        torch.bfloat16,
    )
    assert output.shape == (2, 64)
    assert torch.equal(output, bias.expand_as(output))


def test_nvfp4_bycopy_preserves_public_skip_bias_add(monkeypatch, default_vllm_config):
    from vllm.model_executor.layers.linear import ReplicatedLinear

    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    layer = ReplicatedLinear(
        192,
        64,
        bias=True,
        skip_bias_add=True,
        params_dtype=torch.bfloat16,
        disable_tp=True,
    )
    layer.input_size_per_partition = 192
    layer.output_size_per_partition = 64
    logical_bias = torch.arange(64, dtype=torch.bfloat16)
    layer.bias.data.copy_(logical_bias)

    with patch.object(
        MarlinNvFp4ToFp8LinearKernel, "is_supported", return_value=(True, None)
    ):
        kernel = MarlinNvFp4ToFp8LinearKernel(Mock())

    def prepare_marlin(candidate):
        candidate.bias = torch.nn.Parameter(
            marlin_permute_bias(marlin_pad_dim(candidate.bias, 64, 128)),
            requires_grad=False,
        )

    kernel.marlin.process_weights_after_loading = prepare_marlin
    monkeypatch.setattr(marlin_fp8, "marlin_repacked_nk", lambda *args: (128, 256))
    monkeypatch.setattr(
        marlin_fp8, "reserve_workspace_for_all_ubatches", lambda *args: None
    )
    kernel.process_weights_after_loading(layer)

    class SkipBiasMethod:
        @staticmethod
        def apply(candidate, x, bias):
            assert candidate is layer
            assert bias is None
            return torch.zeros((*x.shape[:-1], 64), dtype=x.dtype)

    layer.quant_method = SkipBiasMethod()
    output, output_bias = layer(torch.ones((2, 192), dtype=torch.bfloat16))

    assert not torch.count_nonzero(output)
    assert output_bias is layer.bias
    assert torch.equal(output_bias, logical_bias)

    kernel.m_knee = 512
    kernel.marlin.apply_weights = Mock(return_value=output)
    x = torch.ones((1, 192), dtype=torch.bfloat16)
    kernel.apply_weights(layer, x, layer.bias)
    cached_marlin_bias = layer._marlin_nvfp4_bias
    kernel.apply_weights(layer, x, layer.bias)
    expected_marlin_bias = marlin_permute_bias(marlin_pad_dim(logical_bias, 64, 128))
    assert kernel.marlin.apply_weights.call_count == 2
    for call in kernel.marlin.apply_weights.call_args_list:
        assert call.args[2] is cached_marlin_bias
        assert torch.equal(call.args[2], expected_marlin_bias)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
def test_modelopt_w4a16_can_disable_nvfp4_bycopy(monkeypatch):
    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.layers.quantization.modelopt import (
        RuntimeDtypes,
        select_linear_kernel,
    )

    monkeypatch.setattr(
        "vllm.envs.VLLM_DISABLED_KERNELS",
        [MarlinNvFp4ToFp8LinearKernel.__name__],
    )
    with set_current_vllm_config(VllmConfig()):
        kernel = select_linear_kernel(
            QuantSpec(weight=kNvfp4Static, activation=None),
            MagicMock(),
            RuntimeDtypes(torch.bfloat16, torch.bfloat16),
        )
    assert isinstance(kernel, MarlinNvFp4LinearKernel)


@pytest.mark.parametrize(
    "quant_method, expected_use_a16, act_key_is_none",
    [
        ("NVFP4", False, False),  # W4A4 default
        ("W4A16_NVFP4", True, True),  # native W4A16 ckpt
    ],
)
def test_modelopt_nvfp4_moe_dispatches_to_marlin_when_w4a16(
    quant_method, expected_use_a16, act_key_is_none
):
    """``ModelOptNvFp4FusedMoE``: when the ckpt's ``quant_method`` is
    ``W4A16_NVFP4``, the MoE class must pass ``activation_key=None`` to
    ``select_nvfp4_moe_backend``. That filters out every W4A4 backend
    (their ``_supports_quant_scheme`` requires
    ``(kNvfp4Static, kNvfp4Dynamic)`` exactly); Marlin survives because
    it only checks ``weight_key``. A regression here would mean a W4A16
    ckpt silently went to the cutlass W4A4 path.
    """
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
        ModelOptNvFp4FusedMoE,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kNvfp4Dynamic,
        kNvfp4Static,
    )

    config = ModelOptNvFp4Config(
        quant_method=quant_method,
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
        group_size=16,
    )

    mock_select = MagicMock(return_value=(MagicMock(), MagicMock()))
    with (
        patch(
            "vllm.model_executor.layers.quantization.modelopt.select_nvfp4_moe_backend",
            mock_select,
        ),
        patch(
            "vllm.model_executor.layers.quantization.modelopt."
            "is_global_sf_supported_for_nvfp4_backend",
            return_value=False,
        ),
    ):
        moe = ModelOptNvFp4FusedMoE(config, MagicMock())

    assert moe.use_a16 is expected_use_a16
    _, kwargs = mock_select.call_args
    assert kwargs["weight_key"] is kNvfp4Static
    if act_key_is_none:
        assert kwargs["activation_key"] is None
    else:
        assert kwargs["activation_key"] is kNvfp4Dynamic


@pytest.mark.parametrize(
    "per_layer_algo, expected_weight, expected_activation",
    [
        ("NVFP4", kNvfp4Static, kNvfp4Dynamic),
        ("W4A16_NVFP4", kNvfp4Static, None),
        ("FP8", kFp8StaticTensorSym, kFp8StaticTensorSym),
        ("MXFP8", kMxfp8Static, kMxfp8Dynamic),
    ],
)
def test_modelopt_mixed_precision_dispatches_linear_layer(
    per_layer_algo, expected_weight, expected_activation
):
    """``ModelOptMixedPrecisionConfig.get_quant_method`` routes a Linear layer
    to the generic ``ModelOptLinearMethod`` with the ``QuantSpec`` resolved
    from its per-layer ``quant_algo`` entry in ``quantized_layers``. A
    regression here would mean a layer got the wrong ``(weight, activation)``
    key pair or fell through to ``UnquantizedLinearMethod`` — e.g. a W4A16
    layer picking up a dynamic fp4 activation key (cutlass W4A4 path) instead
    of the weight-only Marlin path.
    """
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization import modelopt as m

    hf_quant_config: dict[str, Any] = {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "exclude_modules": [],
            "group_size": 16,
            "quantized_layers": {
                "model.layers.0.fake_proj": {"quant_algo": per_layer_algo},
            },
        }
    }
    config = m.ModelOptMixedPrecisionConfig.from_config(hf_quant_config)

    fake_layer = MagicMock(spec=LinearBase)
    method = config.get_quant_method(fake_layer, "model.layers.0.fake_proj")

    assert isinstance(method, m.ModelOptLinearMethod)
    assert method.spec.weight is expected_weight
    assert method.spec.activation is expected_activation


def test_modelopt_mixed_precision_builds_w4a16_sibling_config():
    """Sanity: ``ModelOptMixedPrecisionConfig._from_config`` builds **two**
    NVFP4 sub-configs — one for W4A4 (default) and one tagged
    ``quant_method='W4A16_NVFP4'`` — so per-layer dispatch can hand
    Marlin-bound layers the right config without re-instantiating it on
    every call.
    """
    from vllm.model_executor.layers.quantization import modelopt as m

    hf_quant_config: dict[str, Any] = {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "exclude_modules": [],
            "group_size": 16,
            "quantized_layers": {
                "model.layers.0.a": {"quant_algo": "NVFP4"},
                "model.layers.0.b": {"quant_algo": "W4A16_NVFP4"},
            },
        }
    }
    config = m.ModelOptMixedPrecisionConfig.from_config(hf_quant_config)

    assert config.nvfp4_config.quant_method == "NVFP4"
    assert config.w4a16_nvfp4_config.quant_method == "W4A16_NVFP4"


def test_modelopt_fp8_pb_wo_hides_output_padding(monkeypatch):
    """FP8_PB_WO output width that is not a multiple of 128 (a partial trailing
    block) is padded up to a block boundary before the kernel post-load, the
    GEMM runs on the padded weight, and the output is trimmed back to the
    logical width with bias added after. Faithful port of wei-zhao #53132's
    test_modelopt_fp8_pb_wo_hides_output_padding for the generic method +
    _Fp8PbWoPartialBlock FormatScheme.

    Width is the real motivating case -- GLM's fused qkv_a_proj, q_a 2048 +
    kv_a 576 = 2624 (replicated, so no TP degree makes it a 128-multiple),
    padded to 2688 = 21 * 128.
    """
    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.layers.quantization import modelopt as mo
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
        kFp8Static128BlockSym,
    )

    kernel = Mock()
    monkeypatch.setattr(mo, "select_linear_kernel", lambda spec, layer, rt: kernel)
    monkeypatch.setattr(mo, "expose_input_quant_key", lambda layer, k: None)

    method = ModelOptLinearMethod.__new__(ModelOptLinearMethod)
    method.spec = QuantSpec(weight=kFp8Static128BlockSym, activation=kFp8Dynamic128Sym)
    method.ctx = mo.CkptCtx()
    method.fmt = mo._PB_WO_PARTIAL_BLOCK
    method.wkey = mo.SCHEME_FOR[kFp8Static128BlockSym]
    method.akey = mo.SCHEME_FOR[kFp8Dynamic128Sym]
    method.input_dtype = method.out_dtype = torch.bfloat16
    method.marlin_input_dtype = None

    layer = torch.nn.Module()
    with (
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
    ):
        # output 2624 = 2048 + 576, input 128
        method.create_weights(
            layer, 128, [2048, 576], 128, 2624, torch.bfloat16, weight_loader=Mock()
        )

    # loaded at logical size; scale is cdiv(2624, 128) = 21 block rows
    assert layer.weight.shape == (2624, 128)
    assert layer.weight_scale.shape == (21, 1, 1, 1)

    layer.weight.data.fill_(1)
    method.process_weights_after_loading(layer)

    # weight padded to the block boundary; the pad rows are zero
    assert layer.weight.shape == (2688, 128)
    assert torch.count_nonzero(layer.weight[2624:].float()) == 0
    kernel.process_weights_after_loading.assert_called_once_with(layer)

    # apply: GEMM on padded weight (bias=None), output trimmed + bias added
    physical_output = torch.randn(4, 2688, dtype=torch.bfloat16)
    kernel.apply_weights.return_value = physical_output
    bias = torch.randn(2624, dtype=torch.bfloat16)
    output = method.apply(layer, torch.randn(4, 128), bias)

    torch.testing.assert_close(output, physical_output[:, :2624] + bias)
    assert output.shape == (4, 2624)
    assert output.is_contiguous()
    kernel.apply_weights.assert_called_once()
    assert kernel.apply_weights.call_args.kwargs["bias"] is None


def test_modelopt_fp8_pb_wo_rejects_non_128_input():
    """Input width must still be a multiple of 128 (same as #53132, which only
    pads the output). A partial input block is refused loudly rather than
    silently loading wrong scales."""
    from vllm.model_executor.layers.quantization import modelopt as mo
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Static128BlockSym,
    )

    scheme = mo.SCHEME_FOR[kFp8Static128BlockSym]
    shapes = mo.Shapes([128], 100, torch.bfloat16)  # input 100 not divisible by 128
    with pytest.raises(ValueError, match="in divisible by 128"):
        scheme.create_weights(
            torch.nn.Module(), mo.WEIGHT, mo.CkptCtx(), shapes, Mock()
        )
