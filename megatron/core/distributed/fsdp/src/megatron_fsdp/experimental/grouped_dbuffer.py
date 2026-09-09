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
_PLANE_NAMES = ("rowwise_data", "columnwise_data", "rowwise_scale", "columnwise_scale")


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def is_mxfp8_tensor(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` is a TE MXFP8 tensor with both physical data planes."""
    return (
        isinstance(tensor, MXFP8Tensor)
        and tensor._rowwise_data is not None
        and tensor._columnwise_data is not None
    )


class GroupedDBuffer:
    """The MFSDP storage and lifecycle for one TE MXFP8 tensor.

    The data and scale planes can have different dtypes, logical shapes, and
    layouts, but share mesh and placement state.
    """

    mesh: DeviceMesh
    placements: tuple[Placement, ...]
    rowwise_data: DBuffer
    columnwise_data: DBuffer
    rowwise_scale: DBuffer
    columnwise_scale: DBuffer

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
        plane_shapes = {
            "rowwise_data": tensor_shapes,
            "columnwise_data": tensor_shapes,
            "rowwise_scale": tuple(
                torch.Size(
                    (
                        _round_up(shape[0], _MXFP8_ROWWISE_SCALE_ALIGNMENT),
                        _round_up(shape[1] // _MXFP8_BLOCK_SIZE, _MXFP8_SCALE_ROW_ALIGNMENT),
                    )
                )
                for shape in tensor_shapes
            ),
            "columnwise_scale": tuple(
                torch.Size(
                    (
                        _round_up(shape[0] // _MXFP8_BLOCK_SIZE, _MXFP8_SCALE_ROW_ALIGNMENT),
                        _round_up(shape[1], _MXFP8_COLUMNWISE_SCALE_ALIGNMENT),
                    )
                )
                for shape in tensor_shapes
            ),
        }
        planes = tuple(
            DBuffer(
                mesh,
                self._plane_placements(name, placements),
                shapes,
                torch.uint8,
                device,
                block_size=block_size if name != "columnwise_scale" else 1,
            )
            for name, shapes in plane_shapes.items()
        )
        self._set_planes(*planes)

    @classmethod
    def _from_planes(
        cls,
        rowwise_data: DBuffer,
        columnwise_data: DBuffer,
        rowwise_scale: DBuffer,
        columnwise_scale: DBuffer,
    ) -> Self:
        """Create a composed view from already-allocated physical planes."""
        result = cls.__new__(cls)
        result._set_planes(rowwise_data, columnwise_data, rowwise_scale, columnwise_scale)
        return result

    def _set_planes(
        self,
        rowwise_data: DBuffer,
        columnwise_data: DBuffer,
        rowwise_scale: DBuffer,
        columnwise_scale: DBuffer,
    ) -> None:
        """Install physical planes with the rowwise data plane's logical layout."""
        self.rowwise_data = rowwise_data
        self.columnwise_data = columnwise_data
        self.rowwise_scale = rowwise_scale
        self.columnwise_scale = columnwise_scale
        self.mesh = rowwise_data.mesh
        self.placements = rowwise_data.placements

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
        rowwise_data = self.rowwise_data.get_local_tensor(index)
        fp8_dtype = tex.DType.kFloat8E4M3
        return MXFP8Tensor(
            shape=rowwise_data.shape,
            dtype=torch.bfloat16,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=self.rowwise_scale.get_local_tensor(index),
            columnwise_data=self.columnwise_data.get_local_tensor(index),
            columnwise_scale_inv=self.columnwise_scale.get_local_tensor(index),
            fp8_dtype=fp8_dtype,
            quantizer=MXFP8Quantizer(fp8_dtype),
            with_gemm_swizzled_scales=False,
            device=rowwise_data.device,
            requires_grad=False,
        )

    def sync_from_main(self, main_weight: DBuffer) -> None:
        """Quantize the local FP32 master shard into MXFP8 grouped planes."""
        for index in range(len(self.rowwise_data.layout.tensor_shapes)):
            tensor = self.get_local_tensor(index)
            with torch.no_grad():
                tensor.quantize_(main_weight.get_local_tensor(index))

    @property
    def planes(self) -> tuple[DBuffer, DBuffer, DBuffer, DBuffer]:
        """Physical planes in TE's rowwise-data-first order."""
        return self.rowwise_data, self.columnwise_data, self.rowwise_scale, self.columnwise_scale

    @property
    def is_symmetric_memory(self) -> bool:
        """Whether every plane is backed by symmetric memory."""
        return all(plane.is_symmetric_memory for plane in self.planes)

    def _result(self, planes: tuple[DBuffer, DBuffer, DBuffer, DBuffer], out: Self | None) -> Self:
        """Return constructed plane results or preserve a supplied destination."""
        if out is None:
            return type(self)._from_planes(*planes)
        assert all(result is destination for result, destination in zip(planes, out.planes))
        return out

    def reallocate_storage(self) -> None:
        """Restore every plane's backing storage."""
        for plane in self.planes:
            plane.reallocate_storage()

    def release_storage(self) -> None:
        """Release every plane's backing storage while retaining aliases."""
        for plane in self.planes:
            plane.release_storage()

    def view(self, placements: Iterable[Placement]) -> Self:
        """Return a storage-sharing view of every physical plane."""
        return type(self)._from_planes(
            *(
                plane.view(self._plane_placements(name, placements))
                for name, plane in zip(_PLANE_NAMES, self.planes)
            )
        )

    def redistribute(self, new_placements: Iterable[Placement], *, out: Self | None = None) -> Self:
        """Redistribute every plane with the same placement transition."""
        new_placements = tuple(new_placements)
        if out is not None:
            if out.mesh != self.mesh:
                raise ValueError(f"Expected out mesh {self.mesh!r}, got {out.mesh!r}.")
            if out.placements != new_placements:
                raise ValueError(
                    f"Expected out placements {new_placements!r}, got {out.placements!r}."
                )
        result_planes = tuple(
            plane.redistribute(
                self._plane_placements(name, new_placements),
                out=None if out is None else out.planes[index],
            )
            for index, (name, plane) in enumerate(zip(_PLANE_NAMES, self.planes))
        )
        return self._result(result_planes, out)

    def allgather(self, mesh_axis: int, *, out: Self | None = None) -> Self:
        """All-gather every materialized physical plane on ``mesh_axis``."""
        result_planes = tuple(
            plane.allgather(mesh_axis, out=None if out is None else out.planes[index])
            for index, plane in enumerate(self.planes)
        )
        return self._result(result_planes, out)
