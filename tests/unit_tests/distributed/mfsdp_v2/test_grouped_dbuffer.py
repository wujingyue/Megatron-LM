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
    for plane in grouped.planes:
        plane.local_buffer.fill_(mesh.get_local_rank())

    result = grouped.allgather(0)

    assert result.rowwise_data.placements == (Replicate(),)
    for plane in result.planes:
        assert plane.local_buffer.view(mesh.size(), -1)[0].eq(0).all()
        assert plane.local_buffer.view(mesh.size(), -1)[1].eq(1).all()


def test_grouped_dbuffer_redistributes_into_matching_destinations(distributed_setup):
    """A preallocated grouped destination receives every plane."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    source = _grouped(mesh, [Replicate()], distributed_setup.device)
    for plane in source.planes:
        plane.local_buffer.copy_(torch.arange(plane.local_buffer.numel(), device=plane.device))
    destination = _grouped(mesh, [Flat()], distributed_setup.device)
    result = source.redistribute([Flat()], out=destination)

    assert result is destination
    for result_plane, source_plane in zip(result.planes, source.planes):
        torch.testing.assert_close(
            result_plane.allgather(0).local_buffer, source_plane.local_buffer
        )


@pytest.mark.parametrize("preserve_high_precision_init_val", [True, False])
def test_mxfp8_linear_training_step_uses_grouped_dbuffer(
    distributed_setup, preserve_high_precision_init_val
):
    """A bias-free TE MXFP8 Linear completes a ZeRO-3 training step on two ranks."""
    te = pytest.importorskip("transformer_engine")
    if distributed_setup.world_size != 2:
        pytest.skip("MXFP8 grouped DBuffer coverage requires exactly two ranks.")
    if torch.cuda.get_device_capability(distributed_setup.device)[0] < 10:
        pytest.skip("MXFP8 requires Blackwell-or-newer CUDA hardware.")

    recipe = te.common.recipe.MXFP8BlockScaling(fp8_format=te.common.recipe.Format.HYBRID)
    with te.pytorch.quantized_model_init(
        recipe=recipe, preserve_high_precision_init_val=preserve_high_precision_init_val
    ):
        linear = te.pytorch.Linear(
            64, 256, bias=False, params_dtype=torch.bfloat16, device=distributed_setup.device
        )
        reference = te.pytorch.Linear(
            64, 256, bias=False, params_dtype=torch.bfloat16, device=distributed_setup.device
        )
    get_high_precision_init_val = getattr(linear.weight, "get_high_precision_init_val", None)
    initial_value = get_high_precision_init_val() if get_high_precision_init_val else None
    reference_main_weight = torch.nn.Parameter(
        (initial_value if initial_value is not None else linear.weight).to(
            device=distributed_setup.device, dtype=torch.float32
        )
    )
    with torch.no_grad():
        reference.weight.quantize_(reference_main_weight)
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

    optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)
    fully_shard_optimizer(optimizer)
    reference_optimizer = torch.optim.SGD([reference_main_weight], lr=0.1)

    torch.manual_seed(1234)
    x = torch.randn(32, 64, dtype=torch.bfloat16, device=distributed_setup.device)
    optimizer.zero_grad(set_to_none=True)
    reference_optimizer.zero_grad(set_to_none=True)
    with te.pytorch.autocast(recipe=recipe):
        loss = linear(x).float().square().mean()
        reference_loss = reference(x).float().square().mean()
        torch.testing.assert_close(loss, reference_loss, rtol=5e-2, atol=5e-2)
    loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(
        parameter_group.main_grad.get_local_tensor(0),
        reference.weight.grad[distributed_setup.rank * 128 : (distributed_setup.rank + 1) * 128],
        rtol=5e-2,
        atol=5e-2,
    )
    reference_main_weight.grad = reference.weight.grad.float()
    reference.weight.grad = None
    optimizer.step()
    reference_optimizer.step()
    torch.testing.assert_close(
        parameter_group.main_weight.get_local_tensor(0),
        reference_main_weight[distributed_setup.rank * 128 : (distributed_setup.rank + 1) * 128],
        rtol=0,
        atol=0,
    )
    assert torch.isfinite(loss)
    with torch.no_grad(), te.pytorch.autocast(recipe=recipe):
        assert torch.isfinite(linear(x)).all()
