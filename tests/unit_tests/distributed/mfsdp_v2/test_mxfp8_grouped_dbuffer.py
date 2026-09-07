# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""End-to-end MXFP8 coverage for experimental MFSDP grouped buffers."""

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

te = pytest.importorskip("transformer_engine")

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (  # noqa: E402
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.mxfp8_grouped_dbuffer import (  # noqa: E402
    MXFP8GroupedDBuffer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import (  # noqa: E402
    BlockAtomic,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import (  # noqa: E402
    MixedPrecisionPolicy,
)


def test_mxfp8_linear_training_step_uses_grouped_dbuffer(distributed_setup):
    """A bias-free TE MXFP8 Linear completes a ZeRO-3 training step on two ranks."""
    if distributed_setup.world_size != 2:
        pytest.skip("MXFP8 grouped DBuffer coverage requires exactly two ranks.")
    if torch.cuda.get_device_capability(distributed_setup.device)[0] < 10:
        pytest.skip("MXFP8 requires Blackwell-or-newer CUDA hardware.")

    recipe = te.common.recipe.MXFP8BlockScaling(fp8_format=te.common.recipe.Format.HYBRID)
    with te.pytorch.quantized_model_init(recipe=recipe, preserve_high_precision_init_val=True):
        linear = te.pytorch.Linear(
            64, 64, bias=False, params_dtype=torch.bfloat16, device=distributed_setup.device
        )

    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    with fully_shard_context(device=distributed_setup.device):
        fully_shard(
            linear,
            mesh=mesh,
            placements=placements,
            mixed_precision_policy=MixedPrecisionPolicy(main_params_dtype=torch.float32),
        )

    parameter_group = linear.parameter_groups[0]
    assert isinstance(parameter_group.mxfp8_model_weight, MXFP8GroupedDBuffer)
    assert parameter_group.main_weight.placements == (BlockAtomic(32),)

    optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)
    fully_shard_optimizer(optimizer)
    main_weight_before = parameter_group.main_weight.local_buffer.detach().clone()

    x = torch.randn(32, 64, dtype=torch.bfloat16, device=distributed_setup.device)
    optimizer.zero_grad(set_to_none=True)
    with te.pytorch.autocast(recipe=recipe):
        loss = linear(x).float().square().mean()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert not torch.equal(main_weight_before, parameter_group.main_weight.local_buffer)
    grouped = parameter_group.mxfp8_model_weight.grouped
    assert grouped.plane_names == (
        "rowwise_data",
        "columnwise_data",
        "rowwise_scale",
        "columnwise_scale",
    )
