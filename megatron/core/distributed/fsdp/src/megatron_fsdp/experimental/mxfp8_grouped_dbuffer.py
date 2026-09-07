# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Transformer Engine MXFP8 adapter built on :mod:`grouped_dbuffer`."""

from collections.abc import Sequence

import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.placement_types import Placement

from ..mixed_precision import HAVE_TE_MXFP8TENSOR
from .dbuffer import DBuffer
from .grouped_dbuffer import GroupedDBuffer

if HAVE_TE_MXFP8TENSOR:
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
else:
    MXFP8Tensor = None

_MXFP8_BLOCK_SIZE = 32


def is_mxfp8_tensor(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` is a TE MXFP8 tensor with both physical data planes."""
    return (
        HAVE_TE_MXFP8TENSOR
        and isinstance(tensor, MXFP8Tensor)
        and tensor._rowwise_data is not None
        and tensor._columnwise_data is not None
    )


class MXFP8GroupedDBuffer:
    """MFSDP storage adapter for an aligned TE :class:`MXFP8Tensor`.

    TE pads its scale tensors for GEMM kernels.  Those padded regions are local
    implementation details, so this adapter communicates only the meaningful
    scale tiles and reconstructs the padded tensors before installing a compute
    parameter.  The four communicated planes are composed by ``GroupedDBuffer``.
    """

    def __init__(
        self, tensors: Sequence[torch.Tensor], mesh: DeviceMesh, placements: Sequence[Placement]
    ) -> None:
        if len(tensors) != 1:
            raise NotImplementedError(
                "Experimental MXFP8 MFSDP currently supports one parameter per group."
            )
        tensor = tensors[0]
        if not is_mxfp8_tensor(tensor):
            raise TypeError("MXFP8GroupedDBuffer requires a materialized TE MXFP8Tensor.")
        if tensor.ndim != 2 or tensor.shape[0] % _MXFP8_BLOCK_SIZE:
            raise NotImplementedError(
                "Experimental MXFP8 MFSDP requires a 2D tensor whose dim 0 is divisible by 32."
            )
        if mesh.size() != 2 or tensor.shape[0] // mesh.size() % _MXFP8_BLOCK_SIZE:
            raise NotImplementedError(
                "Experimental MXFP8 MFSDP currently requires two equal dim-0 "
                "shards of 32-row blocks."
            )

        self.tensor = tensor
        self.mesh = mesh
        self.placements = tuple(placements)
        self._local_rows = tensor.shape[0] // mesh.size()
        self.grouped = GroupedDBuffer(
            {
                "rowwise_data": DBuffer.distribute_tensors(
                    [tensor._rowwise_data], mesh, self.placements, block_size=_MXFP8_BLOCK_SIZE
                ),
                "columnwise_data": DBuffer.distribute_tensors(
                    [tensor._columnwise_data], mesh, self.placements, block_size=_MXFP8_BLOCK_SIZE
                ),
                "rowwise_scale": DBuffer.distribute_tensors(
                    [self._compact_rowwise_scale(tensor)],
                    mesh,
                    self.placements,
                    block_size=_MXFP8_BLOCK_SIZE,
                ),
                # One columnwise-scale row represents one 32-row data block.
                # Its layout may therefore use a one-row atom while retaining the
                # same placement state as the raw data planes.
                "columnwise_scale": DBuffer.distribute_tensors(
                    [self._compact_columnwise_scale(tensor)], mesh, self.placements
                ),
            }
        )
        self.unsharded = GroupedDBuffer(
            {
                name: DBuffer(
                    mesh=mesh,
                    placements=[Replicate()] * mesh.ndim,
                    tensor_shapes=plane.layout.tensor_shapes,
                    dtype=plane.dtype,
                    device=plane.device,
                    block_size=plane.layout.block_size,
                )
                for name, plane in self.grouped.planes.items()
            }
        )
        # TE's split dispatch creates wrappers with correctly shaped padded scale
        # allocations.  Their data storage is immediately replaced by our DBuffer
        # views, while the scale allocations receive compact-tile unpacking.
        self._local_tensor = tensor.split(self._local_rows, dim=0)[mesh.get_local_rank()]
        self._bind_local_tensor()

    def _compact_rowwise_scale(self, tensor: torch.Tensor) -> torch.Tensor:
        rows, columns = (
            tensor.shape[0],
            (tensor.shape[1] + _MXFP8_BLOCK_SIZE - 1) // _MXFP8_BLOCK_SIZE,
        )
        return tensor._rowwise_scale_inv[:rows, :columns].contiguous()

    def _compact_columnwise_scale(self, tensor: torch.Tensor) -> torch.Tensor:
        rows = (tensor.shape[0] + _MXFP8_BLOCK_SIZE - 1) // _MXFP8_BLOCK_SIZE
        columns = tensor.shape[1]
        return tensor._columnwise_scale_inv[:rows, :columns].contiguous()

    @staticmethod
    def _unpack_scale(destination: torch.Tensor, source: torch.Tensor) -> None:
        destination.zero_()
        destination[: source.shape[0], : source.shape[1]].copy_(source)

    def _bind_local_tensor(self) -> None:
        self._local_tensor._rowwise_data = self.grouped.plane("rowwise_data").get_local_tensor(0)
        self._local_tensor._columnwise_data = self.grouped.plane(
            "columnwise_data"
        ).get_local_tensor(0)
        self._unpack_scale(
            self._local_tensor._rowwise_scale_inv,
            self.grouped.plane("rowwise_scale").get_local_tensor(0),
        )
        self._unpack_scale(
            self._local_tensor._columnwise_scale_inv,
            self.grouped.plane("columnwise_scale").get_local_tensor(0),
        )

    def sync_from_main(self, main_weight: DBuffer) -> None:
        """Quantize the local FP32 master shard into the grouped compute planes."""
        with torch.no_grad():
            self._local_tensor.quantize_(main_weight.get_local_tensor(0))
        self.grouped.plane("rowwise_scale").get_local_tensor(0).copy_(
            self._compact_rowwise_scale(self._local_tensor)
        )
        self.grouped.plane("columnwise_scale").get_local_tensor(0).copy_(
            self._compact_columnwise_scale(self._local_tensor)
        )

    def unshard_into_tensor(self) -> None:
        """All-gather every plane and bind the full TE wrapper for compute."""
        self.unsharded.reallocate_storage()
        self.grouped.redistribute([Replicate()] * self.mesh.ndim, out=self.unsharded)
        self.tensor._rowwise_data = self.unsharded.plane("rowwise_data").get_local_tensor(0)
        self.tensor._columnwise_data = self.unsharded.plane("columnwise_data").get_local_tensor(0)
        self._unpack_scale(
            self.tensor._rowwise_scale_inv,
            self.unsharded.plane("rowwise_scale").get_local_tensor(0),
        )
        self._unpack_scale(
            self.tensor._columnwise_scale_inv,
            self.unsharded.plane("columnwise_scale").get_local_tensor(0),
        )

    def release_unsharded_storage(self) -> None:
        """Release all-gathered full-plane storage after compute."""
        self.unsharded.release_storage()
