# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer Engine MXFP8 distributed buffers with coordinated physical planes.

``GroupedDBuffer`` owns an MXFP8 tensor's data and scale planes, their MFSDP
layouts, and the tensor-wrapper lifecycle required by Transformer Engine.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Self

import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.placement_types import Placement

from ..mixed_precision import HAVE_TE_MXFP8TENSOR
from .dbuffer import DBuffer

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


class GroupedDBuffer:
    """The MFSDP storage and lifecycle for one TE MXFP8 tensor.

    The data and scale planes can have different dtypes, logical shapes, and
    layouts, but share mesh and placement state.  The direct-plane constructor
    is retained for constructing the replicated staging storage internally.
    """

    mesh: DeviceMesh
    placements: tuple[Placement, ...]
    planes: dict[str, DBuffer]
    tensor: torch.Tensor | None
    unsharded: Self | None
    _local_tensor: torch.Tensor | None

    def __init__(self, planes: Mapping[str, DBuffer]) -> None:
        if not planes:
            raise ValueError("GroupedDBuffer requires at least one physical plane.")
        if any(not name for name in planes):
            raise ValueError("GroupedDBuffer plane names must be non-empty.")

        self.planes = dict(planes)
        self.tensor = None
        self.unsharded = None
        self._local_tensor = None
        first_plane = next(iter(self.planes.values()))
        self.mesh = first_plane.mesh
        self.placements = first_plane.placements
        for name, plane in self.planes.items():
            if plane.mesh != self.mesh:
                raise ValueError(
                    f"Plane {name!r} uses mesh {plane.mesh!r}, expected {self.mesh!r}."
                )
            if plane.placements != self.placements:
                raise ValueError(
                    f"Plane {name!r} uses placements {plane.placements!r}, "
                    f"expected {self.placements!r}."
                )

    @classmethod
    def from_mxfp8(
        cls, tensors: Sequence[torch.Tensor], mesh: DeviceMesh, placements: Sequence[Placement]
    ) -> Self:
        """Create MXFP8 grouped storage and retain its TE wrapper metadata."""
        if len(tensors) != 1:
            raise NotImplementedError("Experimental MXFP8 MFSDP supports one parameter per group.")
        tensor = tensors[0]
        if not is_mxfp8_tensor(tensor) or tensor.ndim != 2:
            raise TypeError("GroupedDBuffer.from_mxfp8() requires a 2D materialized MXFP8Tensor.")
        if tensor.shape[0] % _MXFP8_BLOCK_SIZE or mesh.size() != 2:
            raise NotImplementedError("MXFP8 MFSDP requires two equal 32-row-block shards.")
        local_rows = tensor.shape[0] // mesh.size()
        if local_rows % _MXFP8_BLOCK_SIZE:
            raise NotImplementedError("MXFP8 MFSDP requires two equal 32-row-block shards.")
        result = cls(
            {
                "rowwise_data": DBuffer.distribute_tensors(
                    [tensor._rowwise_data], mesh, placements, block_size=_MXFP8_BLOCK_SIZE
                ),
                "columnwise_data": DBuffer.distribute_tensors(
                    [tensor._columnwise_data], mesh, placements, block_size=_MXFP8_BLOCK_SIZE
                ),
                "rowwise_scale": DBuffer.distribute_tensors(
                    [cls._compact_rowwise_scale(tensor)],
                    mesh,
                    placements,
                    block_size=_MXFP8_BLOCK_SIZE,
                ),
                "columnwise_scale": DBuffer.distribute_tensors(
                    [cls._compact_columnwise_scale(tensor)], mesh, placements
                ),
            }
        )
        result.tensor = tensor
        result.unsharded = cls(
            {
                name: DBuffer(
                    mesh=mesh,
                    placements=[Replicate()] * mesh.ndim,
                    tensor_shapes=plane.layout.tensor_shapes,
                    dtype=plane.dtype,
                    device=plane.device,
                    block_size=plane.layout.block_size,
                )
                for name, plane in result.planes.items()
            }
        )
        result._local_tensor = tensor.split(local_rows, dim=0)[mesh.get_local_rank()]
        result._bind_local_tensor()
        return result

    @staticmethod
    def _compact_rowwise_scale(tensor: torch.Tensor) -> torch.Tensor:
        """Remove TE padding from a rowwise scale tensor."""
        return tensor._rowwise_scale_inv[: tensor.shape[0], : tensor.shape[1] // 32].contiguous()

    @staticmethod
    def _compact_columnwise_scale(tensor: torch.Tensor) -> torch.Tensor:
        """Remove TE padding from a columnwise scale tensor."""
        return tensor._columnwise_scale_inv[: tensor.shape[0] // 32, : tensor.shape[1]].contiguous()

    @staticmethod
    def _unpack_scale(destination: torch.Tensor, source: torch.Tensor) -> None:
        destination.zero_()
        destination[: source.shape[0], : source.shape[1]].copy_(source)

    def _bind_local_tensor(self) -> None:
        assert self._local_tensor is not None
        self._local_tensor._rowwise_data = self.plane("rowwise_data").get_local_tensor(0)
        self._local_tensor._columnwise_data = self.plane("columnwise_data").get_local_tensor(0)
        self._unpack_scale(
            self._local_tensor._rowwise_scale_inv, self.plane("rowwise_scale").get_local_tensor(0)
        )
        self._unpack_scale(
            self._local_tensor._columnwise_scale_inv,
            self.plane("columnwise_scale").get_local_tensor(0),
        )

    def sync_from_main(self, main_weight: DBuffer) -> None:
        """Quantize the local FP32 master shard into MXFP8 grouped planes."""
        assert self._local_tensor is not None
        with torch.no_grad():
            self._local_tensor.quantize_(main_weight.get_local_tensor(0))
        self.plane("rowwise_scale").get_local_tensor(0).copy_(
            self._compact_rowwise_scale(self._local_tensor)
        )
        self.plane("columnwise_scale").get_local_tensor(0).copy_(
            self._compact_columnwise_scale(self._local_tensor)
        )

    def unshard_into_tensor(self) -> None:
        """All-gather MXFP8 planes and bind them to the retained TE wrapper."""
        assert self.tensor is not None and self.unsharded is not None
        self.unsharded.reallocate_storage()
        self.redistribute([Replicate()] * self.mesh.ndim, out=self.unsharded)
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
        """Release the all-gathered MXFP8 plane storage."""
        assert self.unsharded is not None
        self.unsharded.release_storage()

    @property
    def plane_names(self) -> tuple[str, ...]:
        """Names of the physical planes in stable construction order."""
        return tuple(self.planes)

    @property
    def is_symmetric_memory(self) -> bool:
        """Whether every plane is backed by symmetric memory."""
        return all(plane.is_symmetric_memory for plane in self.planes.values())

    def plane(self, name: str) -> DBuffer:
        """Return the named physical DBuffer."""
        try:
            return self.planes[name]
        except KeyError as error:
            raise KeyError(f"Unknown GroupedDBuffer plane {name!r}.") from error

    def reallocate_storage(self) -> None:
        """Restore every plane's backing storage."""
        for plane in self.planes.values():
            plane.reallocate_storage()

    def release_storage(self) -> None:
        """Release every plane's backing storage while retaining aliases."""
        for plane in self.planes.values():
            plane.release_storage()

    def view(self, placements: Iterable[Placement]) -> Self:
        """Return a storage-sharing view of every physical plane."""
        return type(self)({name: plane.view(placements) for name, plane in self.planes.items()})

    def redistribute(self, new_placements: Iterable[Placement], *, out: Self | None = None) -> Self:
        """Redistribute every plane with the same placement transition."""
        new_placements = tuple(new_placements)
        if out is not None:
            if out.plane_names != self.plane_names:
                raise ValueError(
                    f"Expected out planes {self.plane_names!r}, got {out.plane_names!r}."
                )
            if out.mesh != self.mesh:
                raise ValueError(f"Expected out mesh {self.mesh!r}, got {out.mesh!r}.")
            if out.placements != new_placements:
                raise ValueError(
                    f"Expected out placements {new_placements!r}, got {out.placements!r}."
                )
        result_planes = {
            name: plane.redistribute(new_placements, out=None if out is None else out.planes[name])
            for name, plane in self.planes.items()
        }
        if out is not None:
            # DBuffer's out= contract preserves the destination allocation and
            # object identity.  Preserve that contract for the composite too.
            assert all(result_planes[name] is out.planes[name] for name in result_planes)
            return out
        return type(self)(result_planes)

    def allgather(self, mesh_axis: int, *, out: Self | None = None) -> Self:
        """All-gather every materialized physical plane on ``mesh_axis``."""
        if out is not None and out.plane_names != self.plane_names:
            raise ValueError(f"Expected out planes {self.plane_names!r}, got {out.plane_names!r}.")
        result_planes = {
            name: plane.allgather(mesh_axis, out=None if out is None else out.planes[name])
            for name, plane in self.planes.items()
        }
        if out is not None:
            assert all(result_planes[name] is out.planes[name] for name in result_planes)
            return out
        return type(self)(result_planes)
