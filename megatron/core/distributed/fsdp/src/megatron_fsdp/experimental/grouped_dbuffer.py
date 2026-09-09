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

"""Transformer Engine MXFP8 distributed buffers composed from physical planes."""

from collections.abc import Iterable
from typing import Self

import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor.placement_types import Placement

from ..mixed_precision import HAVE_TE_MXFP8TENSOR
from .dbuffer import DBuffer
from .placement import BlockAtomic, Flat

if not HAVE_TE_MXFP8TENSOR:
    raise ImportError("GroupedDBuffer requires Transformer Engine MXFP8 support.")

import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor

_MXFP8_BLOCK_SIZE = 32
_MXFP8_ROWWISE_SCALE_ALIGNMENT = 128
_MXFP8_COLUMNWISE_SCALE_ALIGNMENT = 128
_MXFP8_SCALE_ROW_ALIGNMENT = 4


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


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

    def __init__(
        self,
        mesh: DeviceMesh,
        placements: Iterable[Placement],
        tensor_shapes: Iterable[torch.Size],
        device: torch.device | str,
        *,
        block_size: int = 1,
    ) -> None:
        tensor_shapes = tuple(torch.Size(shape) for shape in tensor_shapes)
        if not tensor_shapes or any(len(shape) != 2 for shape in tensor_shapes):
            raise ValueError("GroupedDBuffer requires one or more 2D MXFP8 tensor shapes.")
        placements = tuple(placements)
        self._set_planes(
            {
                "rowwise_data": DBuffer(
                    mesh, placements, tensor_shapes, torch.uint8, device, block_size=block_size
                ),
                "columnwise_data": DBuffer(
                    mesh, placements, tensor_shapes, torch.uint8, device, block_size=block_size
                ),
                "rowwise_scale": DBuffer(
                    mesh,
                    placements,
                    (
                        torch.Size(
                            (
                                _round_up(shape[0], _MXFP8_ROWWISE_SCALE_ALIGNMENT),
                                _round_up(
                                    shape[1] // _MXFP8_BLOCK_SIZE, _MXFP8_SCALE_ROW_ALIGNMENT
                                ),
                            )
                        )
                        for shape in tensor_shapes
                    ),
                    torch.uint8,
                    device,
                    block_size=block_size,
                ),
                "columnwise_scale": DBuffer(
                    mesh,
                    self._plane_placements("columnwise_scale", placements),
                    (
                        torch.Size(
                            (
                                _round_up(
                                    shape[0] // _MXFP8_BLOCK_SIZE, _MXFP8_SCALE_ROW_ALIGNMENT
                                ),
                                _round_up(shape[1], _MXFP8_COLUMNWISE_SCALE_ALIGNMENT),
                            )
                        )
                        for shape in tensor_shapes
                    ),
                    torch.uint8,
                    device,
                ),
            }
        )

    @classmethod
    def _from_planes(cls, planes: dict[str, DBuffer]) -> Self:
        """Create a composed view from already-allocated plane DBuffers."""
        result = cls.__new__(cls)
        result._set_planes(planes)
        return result

    def _set_planes(self, planes: dict[str, DBuffer]) -> None:
        """Install internally constructed plane DBuffers with a shared mesh and placement."""
        self.planes = planes
        first_plane = next(iter(planes.values()))
        self.mesh = first_plane.mesh
        self.placements = first_plane.placements

    @staticmethod
    def _plane_placements(name: str, placements: Iterable[Placement]) -> tuple[Placement, ...]:
        """Map logical MXFP8 placements to one physical plane's coordinates."""
        placements = tuple(placements)
        if name != "columnwise_scale":
            return placements
        return tuple(
            Flat() if isinstance(placement, BlockAtomic) else placement for placement in placements
        )

    def get_local_tensor(self, index: int) -> torch.Tensor:
        """Construct a TE MXFP8 wrapper from this rank's four physical-plane views."""
        rowwise_data = self.plane("rowwise_data").get_local_tensor(index)
        fp8_dtype = tex.DType.kFloat8E4M3
        return MXFP8Tensor(
            shape=rowwise_data.shape,
            dtype=torch.bfloat16,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=self.plane("rowwise_scale").get_local_tensor(index),
            columnwise_data=self.plane("columnwise_data").get_local_tensor(index),
            columnwise_scale_inv=self.plane("columnwise_scale").get_local_tensor(index),
            fp8_dtype=fp8_dtype,
            quantizer=MXFP8Quantizer(fp8_dtype),
            with_gemm_swizzled_scales=False,
            device=rowwise_data.device,
            requires_grad=False,
        )

    def sync_from_main(self, main_weight: DBuffer) -> None:
        """Quantize the local FP32 master shard into MXFP8 grouped planes."""
        for index in range(len(self.plane("rowwise_data").layout.tensor_shapes)):
            tensor = self.get_local_tensor(index)
            with torch.no_grad():
                tensor.quantize_(main_weight.get_local_tensor(index))

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
        return type(self)._from_planes(
            {
                name: plane.view(self._plane_placements(name, placements))
                for name, plane in self.planes.items()
            }
        )

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
            name: plane.redistribute(
                self._plane_placements(name, new_placements),
                out=None if out is None else out.planes[name],
            )
            for name, plane in self.planes.items()
        }
        if out is not None:
            # DBuffer's out= contract preserves the destination allocation and
            # object identity.  Preserve that contract for the composite too.
            assert all(result_planes[name] is out.planes[name] for name in result_planes)
            return out
        return type(self)._from_planes(result_planes)

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
        return type(self)._from_planes(result_planes)
