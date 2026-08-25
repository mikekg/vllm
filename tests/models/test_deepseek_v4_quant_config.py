# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe import config as fused_moe_config
from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

pytestmark = pytest.mark.skip_global_cleanup


def test_missing_expert_dtype_uses_fp8_with_tp8_dp4_ep(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        fused_moe_config,
        "get_dp_group",
        lambda: SimpleNamespace(rank_in_group=0),
    )
    monkeypatch.setattr(
        fused_moe_config,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    parallel = FusedMoEParallelConfig.make(
        8,
        1,
        4,
        1,
        SimpleNamespace(
            enable_expert_parallel=True,
            all2all_backend="allgather_reducescatter",
            enable_eplb=False,
        ),
    )
    assert (parallel.tp_size, parallel.ep_size) == (1, 32)

    hf_config = DeepseekV4Config(hidden_size=7168, moe_intermediate_size=3072)
    assert hf_config.expert_dtype == "fp8"
    assert DeepseekV4Config(expert_dtype="fp4").expert_dtype == "fp4"

    vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config))
    quant_config = DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    with set_current_vllm_config(vllm_config):
        assert quant_config.expert_dtype == "fp8"
        assert not quant_config.is_scale_e8m0
