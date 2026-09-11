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

import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor.placement_types import Placement

from ..mixed_precision import HAVE_TE_MXFP8TENSOR
from .dbuffer import DBuffer
from .layout import GlobalLayout
from .placement import BlockAtomic, Flat

if not HAVE_TE_MXFP8TENSOR:
    raise ImportError("GroupedDBuffer requires Transformer Engine MXFP8 support.")

import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor

_MXFP8_DTYPE = tex.DType.kFloat8E4M3
_MXFP8_QUANTIZER = MXFP8Quantizer(_MXFP8_DTYPE)
_MXFP8_BLOCK_SIZE = 32


def is_mxfp8_tensor(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` is a TE MXFP8 tensor with both physical data planes."""
    return (
        isinstance(tensor, MXFP8Tensor)
        and tensor._rowwise_data is not None
        and tensor._columnwise_data is not None
    )


def effective_dtype(tensor: torch.Tensor) -> torch.dtype:
    """Return MFSDP's storage dtype for a parameter."""
    return torch.uint8 if is_mxfp8_tensor(tensor) else tensor.dtype


class GroupedDBuffer:
    """The MFSDP storage and lifecycle for one TE MXFP8 tensor.

    The data layout owns the parameter-to-rank partition. Scale planes use its
    compact MXFP8 coordinates, so every plane on a rank describes the same
    local data rows. TE-only padding is added when materializing a wrapper for
    compute, not to distributed storage.
    """

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
        if any(
            shape[0] % _MXFP8_BLOCK_SIZE or shape[1] % _MXFP8_BLOCK_SIZE for shape in tensor_shapes
        ):
            raise ValueError(
                f"GroupedDBuffer requires dimensions divisible by {_MXFP8_BLOCK_SIZE}."
            )
        placements = tuple(placements)
        self.rowwise_data = DBuffer(
            mesh, placements, tensor_shapes, torch.uint8, device, block_size=block_size
        )
        self.columnwise_data = self._plane(mesh, placements, self.rowwise_data.layout, device)
        self.rowwise_scale = self._plane(mesh, placements, self._scale_layout(rowwise=True), device)
        self.columnwise_scale = self._plane(
            mesh,
            self._columnwise_scale_placements(placements),
            self._scale_layout(rowwise=False),
            device,
        )

    def _scale_layout(self, *, rowwise: bool) -> GlobalLayout:
        """Derive a compact scale layout from the data layout's coordinates."""
        data_layout = self.rowwise_data.layout
        tensor_shapes = tuple(
            torch.Size(
                (shape[0], shape[1] // _MXFP8_BLOCK_SIZE)
                if rowwise
                else (shape[0] // _MXFP8_BLOCK_SIZE, shape[1])
            )
            for shape in data_layout.tensor_shapes
        )
        return GlobalLayout(
            tensor_shapes=tensor_shapes,
            tensor_to_offset=tuple(
                offset // _MXFP8_BLOCK_SIZE for offset in data_layout.tensor_to_offset
            ),
            size=data_layout.size // _MXFP8_BLOCK_SIZE,
            block_size=data_layout.block_size if rowwise else 1,
        )

    @staticmethod
    def _plane(
        mesh: DeviceMesh,
        placements: Iterable[Placement],
        layout: GlobalLayout,
        device: torch.device | str,
    ) -> DBuffer:
        """Allocate one physical plane from a GroupedDBuffer-owned layout."""
        placements = tuple(placements)
        _, local_numel = layout.get_local_range(mesh, placements)
        return DBuffer.from_local(
            torch.empty(local_numel, dtype=torch.uint8, device=device), mesh, placements, layout
        )

    @property
    def mesh(self) -> DeviceMesh:
        """Device mesh shared by all physical planes."""
        return self.rowwise_data.mesh

    @property
    def placements(self) -> tuple[Placement, ...]:
        """Logical weight placements shared by the data planes."""
        return self.rowwise_data.placements

    @classmethod
    def _from_planes(
        cls,
        rowwise_data: DBuffer,
        columnwise_data: DBuffer,
        rowwise_scale: DBuffer,
        columnwise_scale: DBuffer,
    ) -> "GroupedDBuffer":
        """Create a composed view from already-allocated physical planes."""
        result = cls.__new__(cls)
        result.rowwise_data = rowwise_data
        result.columnwise_data = columnwise_data
        result.rowwise_scale = rowwise_scale
        result.columnwise_scale = columnwise_scale
        return result

    @staticmethod
    def _columnwise_scale_placements(placements: Iterable[Placement]) -> tuple[Placement, ...]:
        """Map weight placements to columnwise-scale coordinates.

        A columnwise scale row describes one 32-row weight block, so it uses
        Flat instead of BlockAtomic(32) to preserve the same shard boundaries.
        """
        placements = tuple(placements)
        return tuple(
            Flat() if isinstance(placement, BlockAtomic) else placement for placement in placements
        )

    def get_local_tensor(self, index: int) -> torch.Tensor:
        """Construct a TE MXFP8 wrapper from this rank's physical-plane views."""
        rowwise_data = self.rowwise_data.get_local_tensor(index)
        rowwise_scale = self.rowwise_scale.get_local_tensor(index)
        columnwise_scale = self.columnwise_scale.get_local_tensor(index)
        return MXFP8Tensor(
            shape=rowwise_data.shape,
            dtype=torch.bfloat16,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=self._pad_scale(rowwise_scale, rowwise=True),
            columnwise_data=self.columnwise_data.get_local_tensor(index),
            columnwise_scale_inv=self._pad_scale(columnwise_scale, rowwise=False),
            fp8_dtype=_MXFP8_DTYPE,
            quantizer=_MXFP8_QUANTIZER,
            with_gemm_swizzled_scales=False,
            device=rowwise_data.device,
            requires_grad=False,
        )

    @staticmethod
    def _pad_scale(scale: torch.Tensor, *, rowwise: bool) -> torch.Tensor:
        """Pad a compact scale plane to TE's GEMM allocation requirements."""
        padded_shape = _MXFP8_QUANTIZER.get_scale_shape(
            (
                scale.shape[0] if rowwise else scale.shape[0] * _MXFP8_BLOCK_SIZE,
                scale.shape[1] * _MXFP8_BLOCK_SIZE if rowwise else scale.shape[1],
            ),
            columnwise=not rowwise,
        )
        if scale.shape == padded_shape:
            return scale
        padded = torch.zeros(padded_shape, dtype=scale.dtype, device=scale.device)
        padded[: scale.shape[0], : scale.shape[1]].copy_(scale)
        return padded

    def sync_from_main(self, main_weight: DBuffer) -> None:
        """Quantize the local FP32 master shard into MXFP8 grouped planes."""
        for index in range(len(self.rowwise_data.layout.tensor_shapes)):
            tensor = self.get_local_tensor(index)
            rowwise_scale = self.rowwise_scale.get_local_tensor(index)
            columnwise_scale = self.columnwise_scale.get_local_tensor(index)
            with torch.no_grad():
                tensor.quantize_(main_weight.get_local_tensor(index))
                assert tensor._rowwise_scale_inv is not None
                assert tensor._columnwise_scale_inv is not None
                rowwise_scale.copy_(
                    tensor._rowwise_scale_inv[: rowwise_scale.shape[0], : rowwise_scale.shape[1]]
                )
                columnwise_scale.copy_(
                    tensor._columnwise_scale_inv[
                        : columnwise_scale.shape[0], : columnwise_scale.shape[1]
                    ]
                )

    @property
    def planes(self) -> tuple[DBuffer, DBuffer, DBuffer, DBuffer]:
        """Physical planes in TE's rowwise-data-first order."""
        return self.rowwise_data, self.columnwise_data, self.rowwise_scale, self.columnwise_scale

    @property
    def is_symmetric_memory(self) -> bool:
        """Whether every plane is backed by symmetric memory."""
        return all(plane.is_symmetric_memory for plane in self.planes)

    def _result(
        self, planes: tuple[DBuffer, DBuffer, DBuffer, DBuffer], out: "GroupedDBuffer | None"
    ) -> "GroupedDBuffer":
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

    def view(self, placements: Iterable[Placement]) -> "GroupedDBuffer":
        """Return a storage-sharing view of every physical plane."""
        placements = tuple(placements)
        if self.rowwise_data.placements == placements:
            return self
        return GroupedDBuffer._from_planes(
            self.rowwise_data.view(placements),
            self.columnwise_data.view(placements),
            self.rowwise_scale.view(placements),
            self.columnwise_scale.view(self._columnwise_scale_placements(placements)),
        )

    def redistribute(
        self, new_placements: Iterable[Placement], *, out: "GroupedDBuffer | None" = None
    ) -> "GroupedDBuffer":
        """Redistribute every plane with the same placement transition."""
        new_placements = tuple(new_placements)
        if out is not None:
            if out.mesh != self.mesh:
                raise ValueError(f"Expected out mesh {self.mesh!r}, got {out.mesh!r}.")
            if out.rowwise_data.placements != new_placements:
                raise ValueError(
                    "Expected out rowwise-data placements "
                    f"{new_placements!r}, got {out.rowwise_data.placements!r}."
                )
        result_planes = (
            self.rowwise_data.redistribute(
                new_placements, out=None if out is None else out.rowwise_data
            ),
            self.columnwise_data.redistribute(
                new_placements, out=None if out is None else out.columnwise_data
            ),
            self.rowwise_scale.redistribute(
                new_placements, out=None if out is None else out.rowwise_scale
            ),
            self.columnwise_scale.redistribute(
                self._columnwise_scale_placements(new_placements),
                out=None if out is None else out.columnwise_scale,
            ),
        )
        return self._result(result_planes, out)

    def allgather(self, mesh_axis: int, *, out: "GroupedDBuffer | None" = None) -> "GroupedDBuffer":
        """All-gather every materialized physical plane on ``mesh_axis``."""
        result_planes = tuple(
            plane.allgather(mesh_axis, out=None if out is None else out.planes[index])
            for index, plane in enumerate(self.planes)
        )
        return self._result(result_planes, out)
