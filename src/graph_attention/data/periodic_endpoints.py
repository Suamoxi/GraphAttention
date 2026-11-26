"""Qualified Cartesian HIT endpoint identification and native mesh reduction.

Keep native retained node order. The historical drop_max policy keeps the
induced open mesh; periodic_quotient identifies every native hex vertex with
its physical representative, including cells crossing periodic faces.
Attention topology is constructed later by the geometry layer.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .contracts import Mesh, Sample


@dataclass(frozen=True)
class EndpointReduction:
    mesh: Mesh
    retained: torch.Tensor
    full_to_unique: torch.Tensor
    source_shape: tuple[int, int, int]


def reduce_cartesian_endpoints(
    mesh: Mesh, *, side: int, policy: str = "drop_max",
) -> EndpointReduction:
    """Qualify the stored lattice; return an open or periodic unique-node mesh."""
    if policy not in ("drop_max", "periodic_quotient"):
        raise ValueError("unknown endpoint reduction policy")
    if policy == "periodic_quotient" and side < 8:
        raise ValueError("periodic quotient needs at least seven unique points per axis")
    if side < 3 or mesh.coords.shape != (side**3, 3):
        raise ValueError("expected complete endpoint-inclusive cubic mesh")
    if mesh.num_edges or mesh.node_weights is not None:
        raise ValueError("endpoint reduction requires native hex cells, without explicit edges or weights")
    cells = mesh.cell_connectivity
    if cells is None or cells.shape != ((side-1)**3, 8):
        raise ValueError("unexpected endpoint-inclusive hex connectivity")
    coords = mesh.coords.to(torch.float64)
    lo, hi = coords.amin(0), coords.amax(0)
    spacing = (hi-lo)/(side-1)
    if torch.any(spacing <= 0):
        raise ValueError("degenerate Cartesian axis")
    scaled = (coords-lo)/spacing
    ijk = scaled.round().to(torch.long)
    if torch.any((scaled-ijk).abs() > 1e-4) or torch.any(ijk < 0) or torch.any(ijk >= side):
        raise ValueError("coordinates are not a uniform Cartesian lattice")
    linear = (ijk[:,0]*side+ijk[:,1])*side+ijk[:,2]
    if torch.unique(linear).numel() != side**3:
        raise ValueError("nonbijective Cartesian coordinate mapping")
    retained = torch.nonzero((ijk < side-1).all(dim=1), as_tuple=True)[0]
    old_to_new = torch.full((mesh.num_nodes,), -1, dtype=torch.long)
    old_to_new[retained] = torch.arange(retained.numel())
    if retained.numel() != (side-1)**3:
        raise ValueError("unexpected unique node count")
    keep_cells = (old_to_new[cells] >= 0).all(dim=1)
    reduced_cells = old_to_new[cells[keep_cells]]
    if reduced_cells.shape != ((side-2)**3,8) or torch.unique(reduced_cells).numel() != retained.numel():
        raise ValueError("unexpected retained native hex topology")
    # The inverse lookup is derived from coordinates, never exporter IDs.
    inverse = torch.empty_like(linear)
    inverse[linear] = torch.arange(mesh.num_nodes)
    wrapped = ijk.remainder(side-1)
    canonical = (wrapped[:,0]*side+wrapped[:,1])*side+wrapped[:,2]
    full_to_unique = old_to_new[inverse[canonical]]
    if torch.any(full_to_unique < 0) or not torch.equal(full_to_unique[retained], torch.arange(retained.numel())):
        raise ValueError("invalid canonical representative map")
    periodic = policy == "periodic_quotient"
    if periodic:
        # Each native cell must be an elementary Cartesian hex and each
        # anchor must occur exactly once.
        if not torch.allclose(spacing, spacing[0].expand_as(spacing), rtol=1e-6, atol=0.):
            raise ValueError("periodic quotient is qualified only for a cubic box")
        corners = ijk[cells]
        anchors = corners.amin(dim=1)
        offsets = corners - anchors[:, None, :]
        codes = (offsets[..., 0]*2+offsets[..., 1])*2+offsets[..., 2]
        if (torch.any(offsets > 1)
                or not torch.equal(codes.sort(dim=1).values, torch.arange(8).expand_as(codes))
                or torch.unique(anchors, dim=0).shape[0] != len(cells)):
            raise ValueError("native cells must be distinct elementary Cartesian hexahedra")
        reduced_cells = full_to_unique[cells]
        if torch.any(reduced_cells.sort(dim=1).values.diff(dim=1) == 0):
            raise ValueError("periodic identification collapsed a cell")
    metadata = dict(mesh.metadata, periodic_endpoint_policy=policy,
                    original_mesh_id=mesh.mesh_id, source_grid_shape=(side,)*3,
                    unique_grid_shape=(side-1,)*3,
                    periodic_box_lengths=tuple(float(x) for x in hi-lo),
                    periodic_wrap_edges=periodic)
    reduced = Mesh(coords=mesh.coords[retained], edge_index=mesh.edge_index,
                   mesh_id=f"{mesh.mesh_id}:{'unique_periodic' if periodic else 'unique_open'}",
                   cell_connectivity=reduced_cells, metadata=metadata)
    return EndpointReduction(reduced, retained, full_to_unique, (side,)*3)


def reduce_sample(sample: Sample, reduction: EndpointReduction) -> Sample:
    fields = {}
    for name, field in sample.fields.items():
        if field.shape[0] != reduction.full_to_unique.numel():
            raise ValueError(f"unexpected node support for {name}")
        unique = field[reduction.retained]
        # Fail closed: a declared duplicate must be the same physical value,
        # not an independent boundary value or a nonperiodic endpoint.
        if not torch.equal(field, unique[reduction.full_to_unique]):
            raise ValueError(f"{sample.sample_id}: nonidentical periodic endpoint values in {name}")
        fields[name] = unique
    return replace(sample, mesh=reduction.mesh, fields=fields,
                   metadata=dict(sample.metadata, mesh_id=reduction.mesh.mesh_id,
                                 periodic_endpoint_policy=reduction.mesh.metadata["periodic_endpoint_policy"]))


def lift_periodic_endpoints(values: torch.Tensor, reduction: EndpointReduction) -> torch.Tensor:
    """Recreate exporter duplicates exactly, for I/O and like-for-like metrics."""
    if values.shape[-2] != reduction.mesh.num_nodes:
        raise ValueError("unexpected unique-node dimension")
    return values.index_select(-2, reduction.full_to_unique.to(values.device))
