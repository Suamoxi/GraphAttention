"""Explicit endpoint-inclusive Cartesian HIT storage reduction (M49.2).

This is a data representation transform, not periodic attention topology.
Only the declared duplicate maximum planes are removed. Preserve the native
node ordering and the subset of original hexahedra entirely on retained nodes.
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


def reduce_cartesian_endpoints(mesh: Mesh, *, side: int) -> EndpointReduction:
    """Qualify a complete (side)^3 lattice; construct its induced open submesh."""
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
    metadata = dict(mesh.metadata, periodic_endpoint_policy="drop_max",
                    original_mesh_id=mesh.mesh_id, source_grid_shape=(side,)*3,
                    unique_grid_shape=(side-1,)*3,
                    periodic_box_lengths=tuple(float(x) for x in hi-lo),
                    periodic_wrap_edges=False)
    reduced = Mesh(coords=mesh.coords[retained], edge_index=mesh.edge_index,
                   mesh_id=f"{mesh.mesh_id}:unique_open",
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
                                 periodic_endpoint_policy="drop_max"))


def lift_periodic_endpoints(values: torch.Tensor, reduction: EndpointReduction) -> torch.Tensor:
    """Recreate exporter duplicates exactly, for I/O and like-for-like metrics."""
    if values.shape[-2] != reduction.mesh.num_nodes:
        raise ValueError("unexpected unique-node dimension")
    return values.index_select(-2, reduction.full_to_unique.to(values.device))
