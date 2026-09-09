# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for Transformer Engine MXFP8 experimental MFSDP grouped buffers."""

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.grouped_dbuffer import (
    GroupedDBuffer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import (
    BlockAtomic,
    Flat,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy


def _grouped(mesh, placements, device: torch.device) -> GroupedDBuffer:
    return GroupedDBuffer(mesh, placements, [(64, 64)], device, block_size=32)


def test_grouped_dbuffer_allgathers_every_plane(distributed_setup):
    """Every materialized MXFP8-like plane follows the same collective transition."""
    if distributed_setup.world_size < 2:
        pytest.skip("GroupedDBuffer all-gather requires at least two ranks.")

    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    grouped = _grouped(mesh, [Flat()], distributed_setup.device)
    for plane in grouped.planes.values():
        plane.local_buffer.fill_(mesh.get_local_rank())

    result = grouped.allgather(0)

    assert result.placements == (Replicate(),)
    assert result.plane_names == (
        "rowwise_data",
        "columnwise_data",
        "rowwise_scale",
        "columnwise_scale",
    )
    for plane in result.planes.values():
        assert plane.local_buffer.view(mesh.size(), -1)[0].eq(0).all()
        assert plane.local_buffer.view(mesh.size(), -1)[1].eq(1).all()


def test_grouped_dbuffer_redistributes_into_matching_destinations(distributed_setup):
    """A preallocated grouped destination preserves every plane allocation."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    source = _grouped(mesh, [Replicate()], distributed_setup.device)
    for plane in source.planes.values():
        plane.local_buffer.copy_(torch.arange(plane.local_buffer.numel(), device=plane.device))
    destination = _grouped(mesh, [Flat()], distributed_setup.device)
    data_ptrs = {name: plane.local_buffer.data_ptr() for name, plane in destination.planes.items()}

    result = source.redistribute([Flat()], out=destination)

    assert result is destination
    for name, plane in destination.planes.items():
        assert plane.local_buffer.data_ptr() == data_ptrs[name]
        torch.testing.assert_close(
            result.plane(name).allgather(0).local_buffer, source.plane(name).local_buffer
        )


def test_mxfp8_linear_training_step_uses_grouped_dbuffer(distributed_setup):
    """A bias-free TE MXFP8 Linear completes a ZeRO-3 training step on two ranks."""
    te = pytest.importorskip("transformer_engine")
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
    assert isinstance(parameter_group.model_weight, GroupedDBuffer)
    assert parameter_group.post_optimizer_model_weight is parameter_group.model_weight
    assert parameter_group.main_weight.placements == (BlockAtomic(32),)
    assert parameter_group.model_weight.get_local_tensor(0).shape == (32, 64)
    assert isinstance(parameter_group._unsharded_model_weight, GroupedDBuffer)
    assert parameter_group._unsharded_model_weight.get_local_tensor(0).shape == (64, 64)

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
    assert parameter_group.model_weight.plane_names == (
        "rowwise_data",
        "columnwise_data",
        "rowwise_scale",
        "columnwise_scale",
    )
