# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.runner import shared_experts as module
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)


class _SharedLayer(torch.nn.Module):
    def __init__(self, predicate, compressed_tensors=False):
        super().__init__()
        self.proj = torch.nn.Module()
        owner = SimpleNamespace(
            kernel=SimpleNamespace(requires_single_stream_workspace=predicate)
        )
        if compressed_tensors:
            self.proj.scheme = owner
            self.proj.quant_method = SimpleNamespace()
        else:
            self.proj.quant_method = owner


def _make_shared_experts(
    monkeypatch, predicate, mk_overlap=True, compressed_tensors=False
):
    monkeypatch.setattr(module, "aux_stream", lambda: None)
    parallel_config = SimpleNamespace(
        enable_eplb=False,
        use_fi_nvl_two_sided_kernels=False,
        dp_size=1,
    )
    return SharedExperts(
        _SharedLayer(predicate, compressed_tensors),
        SimpleNamespace(moe_parallel_config=parallel_config),
        enable_dbo=False,
        mk_can_overlap_shared_experts=lambda: mk_overlap,
        is_multistream_safe=lambda: True,
    )


def test_bycopy_shared_workspace_disables_overlap(monkeypatch):
    shared_experts = _make_shared_experts(monkeypatch, lambda m: m >= 4)

    assert (
        shared_experts._determine_shared_experts_order(torch.empty(3, 8))
        == SharedExpertsOrder.MK_INTERNAL_OVERLAPPED
    )
    assert (
        shared_experts._determine_shared_experts_order(torch.empty(4, 8))
        == SharedExpertsOrder.NO_OVERLAP
    )


def test_bycopy_shared_workspace_disables_aux_stream(monkeypatch):
    shared_experts = _make_shared_experts(
        monkeypatch, lambda m: m >= 4, mk_overlap=False
    )
    shared_experts._stream = object()

    assert (
        shared_experts._determine_shared_experts_order(torch.empty(3, 8))
        == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
    )
    assert (
        shared_experts._determine_shared_experts_order(torch.empty(4, 8))
        == SharedExpertsOrder.NO_OVERLAP
    )


def test_symbolic_m_preserves_bycopy_fallback(monkeypatch):
    calls = []
    shared_experts = _make_shared_experts(monkeypatch, lambda m: calls.append(m))
    symbolic_input = SimpleNamespace(shape=(object(), 8))

    assert (
        shared_experts._determine_shared_experts_order(symbolic_input)
        == SharedExpertsOrder.MK_INTERNAL_OVERLAPPED
    )
    assert calls == []


def test_compressed_tensors_bycopy_workspace_disables_overlap(monkeypatch):
    shared_experts = _make_shared_experts(
        monkeypatch, lambda m: m >= 4, compressed_tensors=True
    )

    assert (
        shared_experts._determine_shared_experts_order(torch.empty(4, 8))
        == SharedExpertsOrder.NO_OVERLAP
    )
