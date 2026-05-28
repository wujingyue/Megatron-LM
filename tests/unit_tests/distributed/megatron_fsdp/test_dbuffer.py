# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Megatron-FSDP DBuffer."""

import dataclasses
import os
from collections.abc import Iterable, Iterator

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer import (
    DBuffer,
    Flat,
    Partial,
    Replicate,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    Bucket,
    DataParallelBuffer,
    _get_dp_buffer_shard_bucket_index,
    build_data_parallel_buffer_index,
)


@dataclasses.dataclass(frozen=True)
class DistributedSetup:
    """Per-rank distributed test setup."""

    rank: int
    world_size: int
    device: torch.device


@pytest.fixture(scope="module")
def setup() -> Iterator[DistributedSetup]:
    """Read torchrun rank state and set up this rank's local device."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Not running under torchrun.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    yield DistributedSetup(rank=rank, world_size=world_size, device=device)

    if dist.is_initialized():
        dist.destroy_process_group()


def _same_tensors_on_all_ranks(device: torch.device) -> list[torch.Tensor]:
    return [
        torch.arange(21, dtype=torch.float32, device=device).reshape(7, 3),
        torch.arange(10, dtype=torch.float32, device=device).reshape(2, 5) + 100,
        torch.arange(7, dtype=torch.float32, device=device) + 200,
    ]


def _assert_dbuffer_contains_tensors(buffer: DBuffer, expected: Iterable[torch.Tensor]) -> None:
    for index, tensor in enumerate(expected):
        torch.testing.assert_close(buffer.get_tensor(index), tensor)


def _dbuffer_ddp_config() -> DistributedDataParallelConfig:
    return DistributedDataParallelConfig(data_parallel_sharding_strategy="optim_grads_params")


@pytest.mark.distributed
def test_dbuffer_layout_pads_to_lcm_times_dp_size_and_fills_gaps(setup: DistributedSetup):
    """DBuffer layout returns element offsets and pads to LCM * DP size."""
    if setup.world_size < 2:
        pytest.skip("DBuffer layout test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (2,))
    shapes = [torch.Size((5, 4)), torch.Size((2, 6)), torch.Size((3,))]

    buffer = DBuffer(
        mesh=mesh,
        placements=[Replicate()],
        tensor_shapes=shapes,
        dtype=torch.float32,
        device=setup.device,
    )

    assert buffer.layout.tensor_shapes == tuple(shapes)
    assert buffer.layout.tensor_to_offset == (0, 24, 20)
    assert buffer.layout.size == 48


@pytest.mark.distributed
def test_dbuffer_layout_aligns_fragment_offsets_to_rows(setup: DistributedSetup):
    """DBuffer layout keeps small tensors aligned to their non-leading dimensions."""
    if setup.world_size < 2:
        pytest.skip("DBuffer layout test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (2,))
    shapes = [torch.Size((4, 4)), torch.Size((1, 6))]

    buffer = DBuffer(
        mesh=mesh,
        placements=[Replicate()],
        tensor_shapes=shapes,
        dtype=torch.float32,
        device=setup.device,
    )

    assert buffer.layout.tensor_to_offset == (0, 18)
    assert buffer.layout.size == 24


@pytest.mark.distributed
def test_compute_layout_fills_lcm_padding_gaps(setup: DistributedSetup):
    """LCM packing fills row-aligned padding gaps on a 5-rank flat-sharded mesh."""
    if setup.world_size < 5:
        pytest.skip("LCM padding-gap layout test requires at least 5 ranks.")

    # P0-P4 are zero-based logical tensor names matching tensor indices.
    shapes = [
        torch.Size((2, 6)),  # P0
        torch.Size((4, 4)),  # P1
        torch.Size((4, 4)),  # P2
        torch.Size((1, 2)),  # P3
        torch.Size((1, 6)),  # P4
    ]

    mesh = init_device_mesh(setup.device.type, (5,))
    if mesh.get_coordinate() is None:
        pytest.skip("Rank is outside the 5-rank DBuffer mesh.")

    buffer = DBuffer(
        mesh=mesh,
        placements=[Flat()],
        tensor_shapes=shapes,
        dtype=torch.float32,
        device=setup.device,
    )
    layout = buffer.layout
    expected_local_shapes_by_rank = [
        [(2, 6), (0, 4), (0, 4), (0, 2), (0, 6)],
        [(0, 6), (3, 4), (0, 4), (0, 2), (0, 6)],
        [(0, 6), (1, 4), (1, 4), (1, 2), (0, 6)],
        [(0, 6), (0, 4), (3, 4), (0, 2), (0, 6)],
        [(0, 6), (0, 4), (0, 4), (0, 2), (1, 6)],
    ]

    assert layout.tensor_shapes == tuple(shapes)
    assert layout.tensor_to_offset == (0, 12, 32, 28, 48)
    assert layout.size == 60
    expected_local_shapes = expected_local_shapes_by_rank[mesh.get_local_rank(0)]
    for index, expected_shape in enumerate(expected_local_shapes):
        assert buffer.get_tensor(index).shape == torch.Size(expected_shape)
        assert buffer.get_dtensor(index).shape == shapes[index]


@pytest.mark.distributed
def test_data_parallel_buffer_index_uses_dbuffer_layout(setup: DistributedSetup):
    """MFSDP DataParallelBuffer indexing follows the DBuffer row-aligned layout."""
    if setup.world_size < 2:
        pytest.skip("DataParallelBuffer DBuffer layout test requires at least 2 ranks.")

    shapes = [torch.Size((4, 4)), torch.Size((1, 6))]

    item_index_map, bucket_index, shard_bucket_index = build_data_parallel_buffer_index(
        shapes,
        data_parallel_rank=setup.rank,
        data_parallel_world_size=setup.world_size,
        is_data_distributed=True,
        ddp_config=_dbuffer_ddp_config(),
        bucket_id=3,
        chunk_size_factor=12,
    )

    assert item_index_map[0].global_data_index == 0
    assert item_index_map[1].global_data_index == 18
    assert bucket_index.size % (12 * setup.world_size) == 0
    assert shard_bucket_index.size == bucket_index.size // setup.world_size


@pytest.mark.distributed
def test_constructor_allocates_local_buffer(setup: DistributedSetup):
    """DBuffer allocates local storage from shape, mesh, placement, dtype, and device."""
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    tensor_shapes = [torch.Size((7, 3)), torch.Size((2, 5)), torch.Size((7,))]
    mesh_size = mesh.size()

    replicated_buffer = DBuffer(
        mesh=mesh,
        placements=[Replicate()],
        tensor_shapes=tensor_shapes,
        dtype=torch.float32,
        device=setup.device,
    )
    sharded_buffer = DBuffer(
        mesh=mesh,
        placements=[Flat()],
        tensor_shapes=tensor_shapes,
        dtype=torch.float32,
        device=setup.device,
    )

    assert replicated_buffer.layout == sharded_buffer.layout
    assert replicated_buffer.layout.tensor_shapes == tuple(tensor_shapes)
    assert replicated_buffer.offset == 0
    expected_sharded_local_numel = replicated_buffer.layout.size // setup.world_size
    assert sharded_buffer.offset == setup.rank * expected_sharded_local_numel
    assert replicated_buffer.local_buffer.numel() == replicated_buffer.layout.size
    assert sharded_buffer.local_buffer.numel() == replicated_buffer.layout.size // setup.world_size
    assert sharded_buffer.layout.size % (15 * mesh_size) == 0
    assert replicated_buffer.dtype == torch.float32
    assert sharded_buffer.local_buffer.device == setup.device


@pytest.mark.distributed
def test_from_local_reuses_required_local_buffer(setup: DistributedSetup):
    """DBuffer.from_local reuses caller-provided local storage without allocation."""
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    tensors = _same_tensors_on_all_ranks(setup.device)
    replicated_buffer = DBuffer.distribute_tensors(tensors, mesh, [Replicate()])
    local_numel = replicated_buffer.layout.size // setup.world_size
    offset = setup.rank * local_numel
    local_buffer = replicated_buffer.local_buffer.narrow(0, offset, local_numel)

    sharded_buffer = DBuffer.from_local(
        local_buffer, mesh, iter([Flat()]), replicated_buffer.layout.tensor_shapes
    )

    assert sharded_buffer.placements == (Flat(),)
    assert sharded_buffer.layout == replicated_buffer.layout
    assert sharded_buffer.offset == offset
    assert sharded_buffer.local_buffer.data_ptr() == local_buffer.data_ptr()
    _assert_dbuffer_contains_tensors(sharded_buffer.allgather(0), tensors)


@pytest.mark.distributed
def test_replicate_get_tensor_and_dtensor(setup: DistributedSetup):
    """Replicated DBuffer returns full local tensors and replicated DTensors."""
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    tensors = _same_tensors_on_all_ranks(setup.device)

    buffer = DBuffer.distribute_tensors(tensors, mesh, [Replicate()])

    _assert_dbuffer_contains_tensors(buffer, tensors)
    dtensor = buffer.get_dtensor(0)
    torch.testing.assert_close(dtensor.to_local(), tensors[0], rtol=0, atol=0)


@pytest.mark.distributed
def test_distribute_tensors_moves_inputs_to_mesh_device(setup: DistributedSetup):
    """distribute_tensors moves full input tensors to the mesh device type."""
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    tensors = _same_tensors_on_all_ranks(torch.device("cpu"))

    buffer = DBuffer.distribute_tensors(tensors, mesh, [Replicate()])

    assert buffer.local_buffer.device == setup.device
    _assert_dbuffer_contains_tensors(buffer, [tensor.to(setup.device) for tensor in tensors])


@pytest.mark.distributed
def test_data_parallel_buffer_wraps_persistent_storage_with_dbuffer(setup: DistributedSetup):
    """Sharded MFSDP DataParallelBuffer uses DBuffer for persistent item storage."""
    if setup.world_size < 2:
        pytest.skip("DataParallelBuffer DBuffer storage test requires at least 2 ranks.")

    tensors = [
        torch.arange(16, dtype=torch.float32, device=setup.device).reshape(4, 4),
        torch.arange(6, dtype=torch.float32, device=setup.device).reshape(1, 6) + 100,
    ]
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    params = [torch.nn.Parameter(tensor.clone()) for tensor in tensors]
    buffer = DataParallelBuffer(
        _dbuffer_ddp_config(),
        params,
        is_data_distributed=True,
        bucket_id=0,
        device=setup.device,
        data_parallel_group=mesh.get_group(),
        chunk_size_factor=12,
    )
    buffer.init_data(torch.empty(buffer.data_size, dtype=torch.float32, device=setup.device))

    assert buffer.dbuffer is not None
    assert buffer.dbuffer.placements == (Flat(),)

    for item_id, tensor in enumerate(tensors):
        buffer.set_item(item_id, tensor)

    replicated_buffer = buffer.dbuffer.allgather(0)
    _assert_dbuffer_contains_tensors(replicated_buffer, tensors)
    assert buffer.get_shard_from_local_buffer().data_ptr() == buffer.dbuffer.local_buffer.data_ptr()


@pytest.mark.distributed
def test_data_parallel_buffer_uses_dbuffer_for_virtual_shards(setup: DistributedSetup):
    """Unsharded MFSDP buffers expose virtual DP shards through DBuffer scatter."""
    if setup.world_size < 2:
        pytest.skip("DataParallelBuffer virtual shard test requires at least 2 ranks.")

    tensors = [
        torch.arange(16, dtype=torch.float32, device=setup.device).reshape(4, 4),
        torch.arange(6, dtype=torch.float32, device=setup.device).reshape(1, 6) + 100,
    ]
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    params = [torch.nn.Parameter(tensor.clone()) for tensor in tensors]
    buffer = DataParallelBuffer(
        _dbuffer_ddp_config(),
        params,
        is_data_distributed=False,
        bucket_id=0,
        device=setup.device,
        data_parallel_group=mesh.get_group(),
        chunk_size_factor=12,
    )
    buffer.init_data(torch.empty(buffer.data_size, dtype=torch.float32, device=setup.device))

    assert buffer.dbuffer is not None
    assert buffer.dbuffer.placements == (Replicate(),)

    for item_id, tensor in enumerate(tensors):
        buffer.set_item(item_id, tensor)
        torch.testing.assert_close(buffer.get_item(item_id), tensor.flatten(), rtol=0, atol=0)

    flat_view = buffer.dbuffer.scatter(0, Flat())
    torch.testing.assert_close(buffer.get_shard_from_local_buffer(), flat_view.local_buffer)
    for item_id in range(len(tensors)):
        torch.testing.assert_close(
            buffer.get_item(item_id, only_shard=True), flat_view.get_tensor(item_id).view(-1)
        )


@pytest.mark.distributed
def test_data_parallel_buffer_all_gather_uses_dbuffer_primitive(setup: DistributedSetup):
    """MFSDP DataParallelBuffer all-gathers through DBuffer."""
    if setup.world_size < 2:
        pytest.skip("DataParallelBuffer DBuffer all-gather test requires at least 2 ranks.")

    tensors = _same_tensors_on_all_ranks(setup.device)
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    params = [torch.nn.Parameter(tensor.clone()) for tensor in tensors]
    buffer = DataParallelBuffer(
        _dbuffer_ddp_config(),
        params,
        is_data_distributed=True,
        bucket_id=0,
        device=setup.device,
        data_parallel_group=mesh.get_group(),
    )
    buffer.init_data(torch.empty(buffer.data_size, dtype=torch.float32, device=setup.device))
    for item_id, tensor in enumerate(tensors):
        buffer.set_item(item_id, tensor)
    bucket = buffer.allocate_bucket_storage(dtype=buffer.dtype, device=setup.device)

    gathered = buffer.all_gather_into_bucket(bucket, async_op=True)
    assert gathered is not None
    _, work = gathered
    assert work is not None
    work.wait()

    bucket_buffer = DBuffer.from_local(bucket.data, mesh, [Replicate()], [t.shape for t in tensors])
    _assert_dbuffer_contains_tensors(bucket_buffer, tensors)

    sync_bucket = buffer.allocate_bucket_storage(dtype=buffer.dtype, device=setup.device)
    gathered = buffer.all_gather_into_bucket(sync_bucket, async_op=False)
    assert gathered is not None
    gathered_data, work = gathered
    assert work is None
    assert gathered_data.data_ptr() == sync_bucket.data.data_ptr()
    sync_bucket_buffer = DBuffer.from_local(
        sync_bucket.data, mesh, [Replicate()], [t.shape for t in tensors]
    )
    _assert_dbuffer_contains_tensors(sync_bucket_buffer, tensors)


@pytest.mark.distributed
def test_data_parallel_buffer_reduce_scatter_uses_dbuffer_primitive(setup: DistributedSetup):
    """MFSDP DataParallelBuffer reduce-scatters through DBuffer."""
    if setup.world_size < 2:
        pytest.skip("DataParallelBuffer DBuffer reduce-scatter test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    rank_scale = float(setup.rank + 1)
    tensors = [
        torch.full((5, 3), rank_scale, dtype=torch.float32, device=setup.device),
        torch.full((4,), rank_scale * 10, dtype=torch.float32, device=setup.device),
    ]
    params = [torch.nn.Parameter(torch.empty_like(tensor)) for tensor in tensors]
    buffer = DataParallelBuffer(
        _dbuffer_ddp_config(),
        params,
        is_data_distributed=True,
        bucket_id=0,
        device=setup.device,
        data_parallel_group=mesh.get_group(),
    )
    buffer.init_data(torch.empty(buffer.data_size, dtype=torch.float32, device=setup.device))
    bucket = buffer.allocate_bucket_storage(dtype=buffer.dtype, device=setup.device)
    source_buffer = DBuffer.distribute_tensors(tensors, mesh, [Replicate()])
    bucket.data.copy_(source_buffer.local_buffer)

    reduced = buffer.reduce_scatter_bucket_into_shard(
        bucket, reduce_op=dist.ReduceOp.SUM, async_op=True
    )
    assert reduced is not None
    shard, work = reduced
    assert work is not None
    work.wait()

    sharded_buffer = DBuffer.from_local(shard, mesh, [Flat()], [tensor.shape for tensor in tensors])
    replicated_buffer = sharded_buffer.allgather(0)
    scale_sum = float(setup.world_size * (setup.world_size + 1) // 2)
    expected = [
        torch.full((5, 3), scale_sum, dtype=torch.float32, device=setup.device),
        torch.full((4,), scale_sum * 10, dtype=torch.float32, device=setup.device),
    ]
    _assert_dbuffer_contains_tensors(replicated_buffer, expected)


@pytest.mark.distributed
def test_data_parallel_buffer_all_reduce_uses_dbuffer_primitive(setup: DistributedSetup):
    """MFSDP DataParallelBuffer all-reduces through DBuffer."""
    if setup.world_size < 2:
        pytest.skip("DataParallelBuffer DBuffer all-reduce test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    rank_scale = float(setup.rank + 1)
    tensors = [
        torch.full((5, 3), rank_scale, dtype=torch.float32, device=setup.device),
        torch.full((4,), rank_scale * 10, dtype=torch.float32, device=setup.device),
    ]
    params = [torch.nn.Parameter(torch.empty_like(tensor)) for tensor in tensors]
    buffer = DataParallelBuffer(
        _dbuffer_ddp_config(),
        params,
        is_data_distributed=False,
        bucket_id=0,
        device=setup.device,
        data_parallel_group=mesh.get_group(),
    )
    buffer.init_data(torch.empty(buffer.data_size, dtype=torch.float32, device=setup.device))
    source_buffer = DBuffer.distribute_tensors(tensors, mesh, [Replicate()])
    buffer.data.copy_(source_buffer.local_buffer)

    all_reduced = buffer.all_reduce_bucket(
        Bucket(data=buffer.data), reduce_op=dist.ReduceOp.SUM, async_op=True
    )
    assert all_reduced is not None
    _, work = all_reduced
    assert work is not None
    work.wait()

    scale_sum = float(setup.world_size * (setup.world_size + 1) // 2)
    expected = [
        torch.full((5, 3), scale_sum, dtype=torch.float32, device=setup.device),
        torch.full((4,), scale_sum * 10, dtype=torch.float32, device=setup.device),
    ]
    _assert_dbuffer_contains_tensors(buffer.dbuffer, expected)

    buffer.data.copy_(source_buffer.local_buffer)
    all_reduced = buffer.all_reduce_bucket(
        Bucket(data=buffer.data), reduce_op=dist.ReduceOp.SUM, async_op=False
    )
    assert all_reduced is not None
    reduced_data, work = all_reduced
    assert work is None
    assert reduced_data.data_ptr() == buffer.data.data_ptr()
    _assert_dbuffer_contains_tensors(buffer.dbuffer, expected)


@pytest.mark.distributed
def test_data_parallel_buffer_hfsdp_helper_uses_2d_mesh_inner_axis(
    setup: DistributedSetup,
):
    """HFSDP helper buffers use the full DP mesh and communicate along inner DP."""
    if setup.world_size < 4 or setup.world_size % 2 != 0:
        pytest.skip("HFSDP helper DBuffer test requires an even world size of at least 4.")

    tensors = [
        torch.arange(16, dtype=torch.float32, device=setup.device).reshape(4, 4),
        torch.arange(6, dtype=torch.float32, device=setup.device).reshape(1, 6) + 100,
    ]
    shapes = [tensor.shape for tensor in tensors]
    mesh = init_device_mesh(
        setup.device.type, (2, setup.world_size // 2), mesh_dim_names=("dp_outer", "dp_inner")
    )
    item_index_map, bucket_index, _ = build_data_parallel_buffer_index(
        shapes,
        data_parallel_rank=0,
        data_parallel_world_size=mesh.size(),
        is_data_distributed=True,
        ddp_config=_dbuffer_ddp_config(),
        bucket_id=0,
        chunk_size_factor=12,
    )
    shard_bucket_index = _get_dp_buffer_shard_bucket_index(
        bucket_index,
        is_data_distributed=True,
        data_parallel_world_size=mesh.size("dp_inner"),
        data_parallel_rank=mesh.get_local_rank("dp_inner"),
    )
    params = [torch.nn.Parameter(tensor.clone()) for tensor in tensors]
    buffer = DataParallelBuffer(
        _dbuffer_ddp_config(),
        params,
        is_data_distributed=True,
        bucket_id=0,
        device=setup.device,
        data_parallel_group=mesh.get_group("dp_inner"),
        chunk_size_factor=12,
        item_index_map=item_index_map,
        bucket_index=bucket_index,
        shard_bucket_index=shard_bucket_index,
        dbuffer_mesh=mesh,
        dbuffer_mesh_axis="dp_inner",
        dbuffer_placements=(Replicate(), Flat()),
    )
    buffer.init_data(torch.empty(buffer.data_size, dtype=torch.float32, device=setup.device))

    assert buffer.dbuffer is not None
    assert buffer.dbuffer.mesh.ndim == 2
    assert buffer.dbuffer.placements == (Replicate(), Flat())
    assert buffer.dbuffer.layout.size == bucket_index.size
    assert buffer.data_size == bucket_index.size // mesh.size("dp_inner")
    assert buffer.dbuffer.local_buffer.data_ptr() == buffer.data.data_ptr()
    for item_id, tensor in enumerate(tensors):
        buffer.set_item(item_id, tensor)

    bucket = buffer.allocate_bucket_storage(dtype=buffer.dtype, device=setup.device)
    gathered = buffer.all_gather_into_bucket(bucket, async_op=True)
    assert gathered is not None
    _, work = gathered
    assert work is not None
    work.wait()

    bucket_buffer = DBuffer.from_local(bucket.data, mesh, [Replicate(), Replicate()], shapes)
    _assert_dbuffer_contains_tensors(bucket_buffer, tensors)


@pytest.mark.distributed
def test_data_parallel_buffer_full_dp_uses_2d_mesh_logical_rank(
    setup: DistributedSetup,
):
    """Full-DP HFSDP buffers derive logical rank from 2D mesh coordinates."""
    if setup.world_size < 4 or setup.world_size % 2 != 0:
        pytest.skip("HFSDP full-DP DBuffer test requires an even world size of at least 4.")

    tensors = _same_tensors_on_all_ranks(setup.device)
    mesh = init_device_mesh(
        setup.device.type, (2, setup.world_size // 2), mesh_dim_names=("dp_outer", "dp_inner")
    )
    params = [torch.nn.Parameter(tensor.clone()) for tensor in tensors]
    logical_dp_rank = (
        mesh.get_local_rank("dp_inner") * mesh.size("dp_outer")
        + mesh.get_local_rank("dp_outer")
    )
    buffer = DataParallelBuffer(
        _dbuffer_ddp_config(),
        params,
        is_data_distributed=True,
        bucket_id=0,
        device=setup.device,
        data_parallel_group=dist.group.WORLD,
        dp_rank=logical_dp_rank,
        dbuffer_mesh=mesh,
        dbuffer_mesh_axis="dp_outer",
        dbuffer_placements=(Flat(), Flat()),
    )
    buffer.init_data(torch.empty(buffer.data_size, dtype=torch.float32, device=setup.device))

    assert buffer.dbuffer is not None
    assert buffer.dbuffer.placements == (Flat(), Flat())
    assert buffer.dbuffer.offset == buffer.shard_bucket_index.bucket_data_index

    for item_id, tensor in enumerate(tensors):
        buffer.set_item(item_id, tensor)

    replicated_buffer = buffer.dbuffer.allgather("dp_outer").allgather("dp_inner")
    _assert_dbuffer_contains_tensors(replicated_buffer, tensors)


@pytest.mark.distributed
def test_sharded_allgather_round_trip(setup: DistributedSetup):
    """Sharded buffers round-trip through all-gather as contiguous tensor fragments."""
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    tensors = _same_tensors_on_all_ranks(setup.device)

    sharded_buffer = DBuffer.distribute_tensors(tensors, mesh, [Flat()])
    layout = sharded_buffer.layout
    for index, tensor in enumerate(tensors):
        local_tensor = sharded_buffer.get_tensor(index)
        assert local_tensor.shape[1:] == tensor.shape[1:]
        assert local_tensor.is_contiguous()

    replicated_buffer = sharded_buffer.allgather(0)

    assert replicated_buffer.layout == layout
    _assert_dbuffer_contains_tensors(replicated_buffer, tensors)


@pytest.mark.distributed
def test_sharded_allgather_into_existing_buffer(setup: DistributedSetup):
    """Sharded buffers can all-gather directly into a preallocated replicated buffer."""
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    tensors = _same_tensors_on_all_ranks(setup.device)
    sharded_buffer = DBuffer.distribute_tensors(tensors, mesh, [Flat()])
    destination = DBuffer(
        mesh=mesh,
        placements=[Replicate()],
        tensor_shapes=sharded_buffer.layout.tensor_shapes,
        dtype=sharded_buffer.dtype,
        device=sharded_buffer.local_buffer.device,
    )
    destination_data_ptr = destination.local_buffer.data_ptr()

    result = sharded_buffer.allgather(0, out=destination)

    assert result is destination
    assert destination.local_buffer.data_ptr() == destination_data_ptr
    _assert_dbuffer_contains_tensors(destination, tensors)


@pytest.mark.distributed
def test_replicate_scatter_round_trip(setup: DistributedSetup):
    """Replicated buffers locally chunk into sharded buffers and all-gather back."""
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    tensors = _same_tensors_on_all_ranks(setup.device)

    replicated_buffer = DBuffer.distribute_tensors(tensors, mesh, [Replicate()])
    sharded_buffer = replicated_buffer.scatter(0, Flat())
    redistribute_destination = DBuffer(
        mesh=mesh,
        placements=[Flat()],
        tensor_shapes=replicated_buffer.layout.tensor_shapes,
        dtype=replicated_buffer.dtype,
        device=replicated_buffer.local_buffer.device,
    )
    redistributed_sharded_buffer = replicated_buffer.redistribute(
        [Flat()], out=redistribute_destination
    )

    assert sharded_buffer.placements == (Flat(),)
    assert redistributed_sharded_buffer is redistribute_destination
    assert redistributed_sharded_buffer.placements == (Flat(),)
    expected_sharded_local_numel = replicated_buffer.layout.size // setup.world_size
    assert sharded_buffer.offset == setup.rank * expected_sharded_local_numel
    source_slice = replicated_buffer.local_buffer.narrow(
        0, sharded_buffer.offset - replicated_buffer.offset, sharded_buffer.local_buffer.numel()
    )
    assert sharded_buffer.local_buffer.data_ptr() == source_slice.data_ptr()
    torch.testing.assert_close(
        sharded_buffer.local_buffer, redistributed_sharded_buffer.local_buffer, rtol=0, atol=0
    )
    _assert_dbuffer_contains_tensors(sharded_buffer.allgather(0), tensors)


@pytest.mark.distributed
def test_partial_allreduce(setup: DistributedSetup):
    """Partial buffers all-reduce into replicated buffers."""
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    rank_scale = float(setup.rank + 1)
    tensors = [
        torch.full((5, 3), rank_scale, dtype=torch.float32, device=setup.device),
        torch.full((4,), rank_scale * 10, dtype=torch.float32, device=setup.device),
    ]
    partial_buffer = DBuffer.distribute_tensors(tensors, mesh, [Partial()])

    replicated_buffer = partial_buffer.allreduce(0)

    scale_sum = float(setup.world_size * (setup.world_size + 1) // 2)
    expected = [
        torch.full((5, 3), scale_sum, dtype=torch.float32, device=setup.device),
        torch.full((4,), scale_sum * 10, dtype=torch.float32, device=setup.device),
    ]
    _assert_dbuffer_contains_tensors(replicated_buffer, expected)


@pytest.mark.distributed
def test_partial_allreduce_average(setup: DistributedSetup):
    """Partial buffers can all-reduce with AVG."""
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    rank_scale = float(setup.rank + 1)
    tensors = [
        torch.full((5, 3), rank_scale, dtype=torch.float32, device=setup.device),
        torch.full((4,), rank_scale * 10, dtype=torch.float32, device=setup.device),
    ]
    partial_buffer = DBuffer.distribute_tensors(
        tensors, mesh, [Partial(reduce_op=dist.ReduceOp.AVG)]
    )

    destination = DBuffer(
        mesh=mesh,
        placements=[Replicate()],
        tensor_shapes=partial_buffer.layout.tensor_shapes,
        dtype=partial_buffer.dtype,
        device=partial_buffer.local_buffer.device,
    )
    replicated_buffer = partial_buffer.allreduce(0, out=destination)

    assert replicated_buffer is destination
    scale_average = float(setup.world_size + 1) / 2.0
    expected = [
        torch.full((5, 3), scale_average, dtype=torch.float32, device=setup.device),
        torch.full((4,), scale_average * 10, dtype=torch.float32, device=setup.device),
    ]
    _assert_dbuffer_contains_tensors(replicated_buffer, expected)


@pytest.mark.distributed
def test_partial_reduce_scatter_to_flat(setup: DistributedSetup):
    """Partial buffers reduce-scatter into sharded buffers."""
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    rank_scale = float(setup.rank + 1)
    tensors = [
        torch.full((5, 3), rank_scale, dtype=torch.float32, device=setup.device),
        torch.full((4,), rank_scale * 10, dtype=torch.float32, device=setup.device),
    ]
    partial_buffer = DBuffer.distribute_tensors(tensors, mesh, [Partial()])
    layout = partial_buffer.layout

    destination = DBuffer(
        mesh=mesh,
        placements=[Flat()],
        tensor_shapes=partial_buffer.layout.tensor_shapes,
        dtype=partial_buffer.dtype,
        device=partial_buffer.local_buffer.device,
    )
    sharded_buffer = partial_buffer.reduce_scatter(0, Flat(), out=destination)
    replicated_buffer = sharded_buffer.allgather(0)

    assert sharded_buffer is destination
    assert sharded_buffer.placements == (Flat(),)
    assert sharded_buffer.layout == layout
    assert replicated_buffer.layout == layout
    scale_sum = float(setup.world_size * (setup.world_size + 1) // 2)
    expected_tensors = [
        torch.full((5, 3), scale_sum, dtype=torch.float32, device=setup.device),
        torch.full((4,), scale_sum * 10, dtype=torch.float32, device=setup.device),
    ]
    _assert_dbuffer_contains_tensors(replicated_buffer, expected_tensors)


@pytest.mark.distributed
def test_partial_reduce_scatter_to_flat_average(setup: DistributedSetup):
    """Partial buffers can reduce-scatter with AVG."""
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    rank_scale = float(setup.rank + 1)
    tensors = [
        torch.full((5, 3), rank_scale, dtype=torch.float32, device=setup.device),
        torch.full((4,), rank_scale * 10, dtype=torch.float32, device=setup.device),
    ]
    partial_buffer = DBuffer.distribute_tensors(
        tensors, mesh, [Partial(reduce_op=dist.ReduceOp.AVG)]
    )
    layout = partial_buffer.layout

    sharded_buffer = partial_buffer.reduce_scatter(0, Flat())
    replicated_buffer = sharded_buffer.allgather(0)

    assert sharded_buffer.placements == (Flat(),)
    assert sharded_buffer.layout == layout
    assert replicated_buffer.layout == layout
    scale_average = float(setup.world_size + 1) / 2.0
    expected_tensors = [
        torch.full((5, 3), scale_average, dtype=torch.float32, device=setup.device),
        torch.full((4,), scale_average * 10, dtype=torch.float32, device=setup.device),
    ]
    _assert_dbuffer_contains_tensors(replicated_buffer, expected_tensors)


@pytest.mark.distributed
def test_get_dtensor_from_sharded_buffer(setup: DistributedSetup):
    """Sharded DBuffer exposes per-tensor local shards as DTensors."""
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    tensors = _same_tensors_on_all_ranks(setup.device)
    sharded_buffer = DBuffer.distribute_tensors(tensors, mesh, [Flat()])

    dtensor = sharded_buffer.get_dtensor(0)

    torch.testing.assert_close(dtensor.to_local(), sharded_buffer.get_tensor(0), rtol=0, atol=0)
    assert dtensor.shape == tensors[0].shape


@pytest.mark.distributed
def test_2d_mesh_replicate_flat_round_trip(setup: DistributedSetup):
    """A 2D mesh can replicate on one axis and flat-shard on the other."""
    if setup.world_size < 4 or setup.world_size % 2 != 0:
        pytest.skip("2D DBuffer test requires an even world size of at least 4.")

    tensors = _same_tensors_on_all_ranks(setup.device)
    mesh = init_device_mesh(
        setup.device.type, (2, setup.world_size // 2), mesh_dim_names=("replicate", "flat")
    )

    sharded_buffer = DBuffer.distribute_tensors(tensors, mesh, [Replicate(), Flat()])
    replicated_buffer = sharded_buffer.allgather("flat")

    _assert_dbuffer_contains_tensors(replicated_buffer, tensors)


@pytest.mark.distributed
def test_2d_mesh_flat_before_replicate_is_rejected(setup: DistributedSetup):
    """Flat axes must be a suffix to keep every local buffer contiguous."""
    if setup.world_size < 4 or setup.world_size % 2 != 0:
        pytest.skip("2D DBuffer test requires an even world size of at least 4.")

    mesh = init_device_mesh(
        setup.device.type, (2, setup.world_size // 2), mesh_dim_names=("flat", "replicate")
    )

    with pytest.raises(ValueError, match="Flat placements must be a suffix"):
        DBuffer(
            mesh=mesh,
            placements=[Flat(), Replicate()],
            tensor_shapes=[torch.Size((6, 4))],
            dtype=torch.float32,
            device=setup.device,
        )


@pytest.mark.distributed
def test_2d_mesh_shards_across_all_ranks(setup: DistributedSetup):
    """Multiple Flat axes shard local storage by the product of their mesh sizes."""
    if setup.world_size < 4 or setup.world_size % 2 != 0:
        pytest.skip("2D DBuffer test requires an even world size of at least 4.")

    tensors = _same_tensors_on_all_ranks(setup.device)
    mesh = init_device_mesh(
        setup.device.type, (2, setup.world_size // 2), mesh_dim_names=("dp_outer", "dp_inner")
    )
    fully_sharded_buffer = DBuffer.distribute_tensors(tensors, mesh, [Flat(), Flat()])

    assert fully_sharded_buffer.layout.tensor_shapes == tuple(tensor.shape for tensor in tensors)
    expected_local_numel = fully_sharded_buffer.layout.size // mesh.size()
    expected_inner_axis_shard_numel = fully_sharded_buffer.layout.size // mesh.size(1)
    expected_offset = (
        mesh.get_local_rank("dp_inner") * expected_inner_axis_shard_numel
        + mesh.get_local_rank("dp_outer") * expected_local_numel
    )
    assert fully_sharded_buffer.offset == expected_offset
    assert (
        fully_sharded_buffer.local_buffer.numel() == fully_sharded_buffer.layout.size // mesh.size()
    )
    for index, _ in enumerate(tensors):
        assert fully_sharded_buffer.get_tensor(index).is_contiguous()


@pytest.mark.distributed
def test_2d_mesh_partial_flat_reduce_scatter_to_flat_flat(setup: DistributedSetup):
    """Partial+Flat reduce-scatter reduces the existing Flat local shard."""
    if setup.world_size < 4 or setup.world_size % 2 != 0:
        pytest.skip("2D DBuffer test requires an even world size of at least 4.")

    mesh = init_device_mesh(
        setup.device.type, (2, setup.world_size // 2), mesh_dim_names=("dp_outer", "dp_inner")
    )
    outer_scale = float(mesh.get_local_rank(0) + 1)
    tensors = [
        torch.full((6, 2), outer_scale, dtype=torch.float32, device=setup.device),
        torch.full((4,), outer_scale * 10, dtype=torch.float32, device=setup.device),
    ]

    partial_sharded_buffer = DBuffer.distribute_tensors(tensors, mesh, [Partial(), Flat()])
    fully_sharded_buffer = partial_sharded_buffer.reduce_scatter("dp_outer", Flat())
    replicated_buffer = fully_sharded_buffer.allgather("dp_outer").allgather("dp_inner")

    assert fully_sharded_buffer.placements == (Flat(), Flat())
    expected_local_numel = fully_sharded_buffer.layout.size // mesh.size()
    expected_inner_axis_shard_numel = fully_sharded_buffer.layout.size // mesh.size(1)
    expected_offset = (
        mesh.get_local_rank("dp_inner") * expected_inner_axis_shard_numel
        + mesh.get_local_rank("dp_outer") * expected_local_numel
    )
    assert fully_sharded_buffer.offset == expected_offset
    assert (
        fully_sharded_buffer.local_buffer.numel()
        == partial_sharded_buffer.local_buffer.numel() // 2
    )

    outer_scale_sum = float(mesh.size(0) * (mesh.size(0) + 1) // 2)
    expected = [
        torch.full((6, 2), outer_scale_sum, dtype=torch.float32, device=setup.device),
        torch.full((4,), outer_scale_sum * 10, dtype=torch.float32, device=setup.device),
    ]
    _assert_dbuffer_contains_tensors(replicated_buffer, expected)


@pytest.mark.distributed
def test_2d_mesh_replicate_flat_scatter_to_flat_flat(setup: DistributedSetup):
    """Replicate+Flat scatter chunks the existing Flat local shard."""
    if setup.world_size < 4 or setup.world_size % 2 != 0:
        pytest.skip("2D DBuffer test requires an even world size of at least 4.")

    tensors = _same_tensors_on_all_ranks(setup.device)
    mesh = init_device_mesh(
        setup.device.type, (2, setup.world_size // 2), mesh_dim_names=("dp_outer", "dp_inner")
    )

    replicated_sharded_buffer = DBuffer.distribute_tensors(tensors, mesh, [Replicate(), Flat()])
    fully_sharded_buffer = replicated_sharded_buffer.scatter("dp_outer", Flat())
    replicated_buffer = fully_sharded_buffer.allgather("dp_outer").allgather("dp_inner")

    assert fully_sharded_buffer.placements == (Flat(), Flat())
    expected_local_numel = fully_sharded_buffer.layout.size // mesh.size()
    expected_inner_axis_shard_numel = fully_sharded_buffer.layout.size // mesh.size(1)
    expected_offset = (
        mesh.get_local_rank("dp_inner") * expected_inner_axis_shard_numel
        + mesh.get_local_rank("dp_outer") * expected_local_numel
    )
    assert fully_sharded_buffer.offset == expected_offset
    assert (
        fully_sharded_buffer.local_buffer.numel()
        == replicated_sharded_buffer.local_buffer.numel() // 2
    )
    _assert_dbuffer_contains_tensors(replicated_buffer, tensors)
