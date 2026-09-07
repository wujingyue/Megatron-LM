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

"""Coordinated multi-plane distributed buffers.

``GroupedDBuffer`` composes ordinary :class:`DBuffer` instances.  It is the
storage primitive for formats whose logical tensors have more than one physical
plane, such as MXFP8's data and scale tensors.  It deliberately does not depend
on Transformer Engine: a format adapter owns its metadata and uses named planes
to construct the corresponding tensor wrappers.
"""

from collections.abc import Iterable, Mapping
from typing import Self

from torch.distributed import DeviceMesh
from torch.distributed.tensor.placement_types import Placement

from .dbuffer import DBuffer


class GroupedDBuffer:
    """A set of physical DBuffers that move together.

    Planes may have different dtypes, logical shapes, and layouts, but must
    live on the same mesh with the same placement state.  Format adapters are
    responsible for choosing coupled layouts such that corresponding data and
    metadata tiles have the same owner.

    The class intentionally exposes no ``get_dtensor`` equivalent: multi-plane
    storage is not one DTensor.  Adapters should retrieve a named plane and
    construct a format-specific view from it.
    """

    mesh: DeviceMesh
    placements: tuple[Placement, ...]
    planes: dict[str, DBuffer]

    def __init__(self, planes: Mapping[str, DBuffer]) -> None:
        if not planes:
            raise ValueError("GroupedDBuffer requires at least one physical plane.")
        if any(not name for name in planes):
            raise ValueError("GroupedDBuffer plane names must be non-empty.")

        self.planes = dict(planes)
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
