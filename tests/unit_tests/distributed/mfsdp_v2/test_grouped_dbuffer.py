# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for coordinated multi-plane experimental MFSDP buffers."""

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer import DBuffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.grouped_dbuffer import (
    GroupedDBuffer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import Flat


def _planes(device: torch.device) -> dict[str, list[torch.Tensor]]:
    return {
        "rowwise_data": [
            torch.arange(64, dtype=torch.uint8, device=device).reshape(4, 16),
            torch.arange(64, 128, dtype=torch.uint8, device=device).reshape(4, 16),
        ],
        "rowwise_scale": [
            torch.arange(16, dtype=torch.uint8, device=device).reshape(4, 4),
            torch.arange(16, 32, dtype=torch.uint8, device=device).reshape(4, 4),
        ],
        "columnwise_data": [
            torch.arange(32, dtype=torch.uint8, device=device).reshape(2, 16),
            torch.arange(32, 64, dtype=torch.uint8, device=device).reshape(2, 16),
        ],
    }


def test_grouped_dbuffer_allgathers_every_plane(distributed_setup):
    """Every materialized MXFP8-like plane follows the same collective transition."""
    if distributed_setup.world_size < 2:
        pytest.skip("GroupedDBuffer all-gather requires at least two ranks.")

    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    expected = _planes(distributed_setup.device)
    grouped = GroupedDBuffer(
        {
            name: DBuffer.distribute_tensors(tensors, mesh, [Flat()])
            for name, tensors in expected.items()
        }
    )

    result = grouped.allgather(0)

    assert result.placements == (Replicate(),)
    assert result.plane_names == tuple(expected)
    for name, tensors in expected.items():
        for index, tensor in enumerate(tensors):
            torch.testing.assert_close(result.plane(name).get_local_tensor(index), tensor)


def test_grouped_dbuffer_requires_matching_collective_state(distributed_setup):
    """A composed buffer rejects planes that would take different collectives."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    tensors = _planes(distributed_setup.device)
    with pytest.raises(ValueError, match="placements"):
        GroupedDBuffer(
            {
                "rowwise_data": DBuffer.distribute_tensors(tensors["rowwise_data"], mesh, [Flat()]),
                "rowwise_scale": DBuffer.distribute_tensors(
                    tensors["rowwise_scale"], mesh, [Replicate()]
                ),
            }
        )


def test_grouped_dbuffer_redistributes_into_matching_destinations(distributed_setup):
    """A preallocated grouped destination preserves every plane allocation."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    expected = _planes(distributed_setup.device)
    source = GroupedDBuffer(
        {
            name: DBuffer.distribute_tensors(tensors, mesh, [Replicate()])
            for name, tensors in expected.items()
        }
    )
    destination = GroupedDBuffer(
        {
            name: DBuffer(
                mesh=mesh,
                placements=[Flat()],
                tensor_shapes=plane.layout.tensor_shapes,
                dtype=plane.dtype,
                device=plane.device,
                block_size=plane.layout.block_size,
            )
            for name, plane in source.planes.items()
        }
    )
    data_ptrs = {name: plane.local_buffer.data_ptr() for name, plane in destination.planes.items()}

    result = source.redistribute([Flat()], out=destination)

    assert result is destination
    for name, plane in destination.planes.items():
        assert plane.local_buffer.data_ptr() == data_ptrs[name]
        torch.testing.assert_close(
            result.plane(name).allgather(0).local_buffer, source.plane(name).local_buffer
        )
