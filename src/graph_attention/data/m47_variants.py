"""M47 controlled relative-only HIT data variants; source samples are immutable.

A1 retains stored endpoint nodes and the native open graph.
A2 removes physically duplicated maximum-face nodes, retaining the open graph.
A3 uses A2's unique nodes and six-neighbour periodic attention topology.

No correlated source, projection, physical conditioning change or new data split.
"""
from __future__ import annotations

from dataclasses import replace
from math import isfinite

import torch

from .periodic_endpoints import reduce_cartesian_endpoints, reduce_sample

VARIANTS = ("a1", "a2", "a3")


def periodic_edges_in_native_order(coords: torch.Tensor, *, side: int) -> torch.Tensor:
    """Six directed periodic neighbours/node, preserving arbitrary exporter ordering.

    `coords` describe a complete unique Cartesian side^3 lattice, with max
    endpoints omitted. Reordering is derived from coordinates, never HDF5 IDs.
    Avoid sorting millions of 3-D edges: each +axis link is generated once
    in each direction and therefore duplicates cannot arise for side >= 3.
    """
    if side < 3 or coords.ndim != 2 or coords.shape != (side**3, 3):
        raise ValueError("expected complete unique Cartesian [side^3,3] coordinates")
    if not torch.isfinite(coords).all():
        raise ValueError("nonfinite Cartesian coordinates")
    x = coords.to(torch.float64)
    lo, hi = x.amin(dim=0), x.amax(dim=0)
    step = (hi - lo) / (side - 1)
    if not bool((step > 0).all()):
        raise ValueError("degenerate Cartesian lattice")
    scaled = (x - lo) / step
    ijk = scaled.round().to(torch.long)
    if (not bool(((scaled - ijk).abs() <= 1e-4).all())
            or not bool(((ijk >= 0) & (ijk < side)).all())):
        raise ValueError("coordinates are not a complete unique Cartesian lattice")
    linear = (ijk[:, 0] * side + ijk[:, 1]) * side + ijk[:, 2]
    if torch.unique(linear).numel() != side**3:
        raise ValueError("nonbijective Cartesian exporter node order")
    native_of_linear = torch.empty(side**3, dtype=torch.long, device=coords.device)
    native_of_linear[linear] = torch.arange(side**3, dtype=torch.long, device=coords.device)
    flat = torch.arange(side**3, dtype=torch.long, device=coords.device)
    target = native_of_linear[flat]
    links = []
    for stride, position in (
        (side * side, flat // (side * side)),
        (side, (flat // side) % side),
        (1, flat % side),
    ):
        next_flat = flat + torch.where(
            position == side - 1,
            torch.full_like(flat, -(side - 1) * stride),
            torch.full_like(flat, stride),
        )
        source = native_of_linear[next_flat]
        links.extend((torch.stack((source, target)), torch.stack((target, source))))
    return torch.cat(links, dim=1).contiguous()


class M47VariantDataset:
    """Lightweight adapter for the frozen M47 six-case dataset.

    The underlying reader still owns snapshot inventory, case definitions,
    references, split membership and mesh lifetime. Only a2/a3 alter nodes.
    Geometry is built once per underlying cached mesh, not per snapshot.
    """

    def __init__(self, source, *, variant: str):
        if variant not in VARIANTS:
            raise ValueError(f"M47 variant must be one of {VARIANTS}")
        self.source = source
        self.variant = variant
        self._cache = {}

    def __getattr__(self, name):
        return getattr(self.source, name)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        sample = self.source[index]
        if self.variant == "a1":
            return sample
        key = id(sample.mesh)
        if key not in self._cache:
            case = self.source.case_by_sample_id[sample.sample_id]
            side = self.source.resolution_by_sample_id[sample.sample_id]
            reduction = reduce_cartesian_endpoints(sample.mesh, side=side)
            mesh = reduction.mesh
            if self.variant == "a3":
                periods = torch.as_tensor(
                    mesh.metadata["periodic_box_lengths"], dtype=torch.float64
                )
                if not torch.allclose(periods, periods.mean().expand_as(periods),
                                      rtol=1e-5, atol=0):
                    raise ValueError(f"{case}: periodic lengths are not isotropic")
                length_ref = float(
                    self.source.case_definitions[case].reference_scales["L_ref"].value
                )
                if not isfinite(length_ref) or length_ref <= 0:
                    raise ValueError(f"{case}: invalid L_ref")
                nondimensional_length = float(periods.mean()) / length_ref
                if not isfinite(nondimensional_length) or nondimensional_length <= 0:
                    raise ValueError(f"{case}: invalid nondimensional periodic length")
                mesh = replace(
                    mesh,
                    edge_index=periodic_edges_in_native_order(mesh.coords, side=side - 1),
                    mesh_id=f"{mesh.mesh_id}:periodic",
                    metadata={**mesh.metadata,
                              "periodic_wrap_edges": True,
                              "periodic_length_nondimensional": nondimensional_length},
                )
            self._cache[key] = reduction, mesh
        reduction, mesh = self._cache[key]
        reduced = reduce_sample(sample, reduction)
        if self.variant == "a3":
            reduced = replace(reduced, mesh=mesh,
                              metadata={**reduced.metadata, "mesh_id": mesh.mesh_id})
        return reduced
