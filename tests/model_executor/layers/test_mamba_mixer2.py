# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    mamba_v2_sharded_weight_loader,
)


@pytest.mark.parametrize("tp_rank", range(32))
def test_weight_loader_replicates_groups_across_tp_ranks(tp_rank: int):
    loaded_weight = torch.cat(
        (torch.arange(64), 100 + torch.arange(8), 200 + torch.arange(8))
    )
    param = torch.empty(4, dtype=loaded_weight.dtype)

    mamba_v2_sharded_weight_loader([(64, 0, 1), (32, 24, 4), (32, 24, 4)], 32, tp_rank)(
        param, loaded_weight
    )

    expected = torch.tensor(
        [2 * tp_rank, 2 * tp_rank + 1, 100 + tp_rank // 4, 200 + tp_rank // 4]
    )
    torch.testing.assert_close(param, expected)
