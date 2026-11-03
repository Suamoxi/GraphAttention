"""Small, periodic Cartesian HIT geometry for two-scale graph-model qualification.

This constructs *geometry and attention supports*, not the physical DNS/LES Q or
right-inverse B. Full exact-two-hop graphs are intentionally limited to small
qualification problems; production runs must use M41 target-driven extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch

from .connectivity import exact_two_hop_edge_index


@dataclass(frozen=True)
class HITTwoScaleGeometry:
    coarse_coords: torch.Tensor
    fine_coords: torch.Tensor
    coarse_edges: torch.Tensor
    fine_edges: torch.Tensor
    coarse_dilated: torch.Tensor
    fine_dilated: torch.Tensor
    # [source, target] ordering: K/V source, query target.
    coarse_to_fine: torch.Tensor
    fine_to_coarse: torch.Tensor
    cross_weights: torch.Tensor
    refinement: int
    coarse_side: int
    periodic_length: float

    @property
    def coarse_spacing(self) -> float:
        return self.periodic_length / self.coarse_side

    @property
    def fine_side(self) -> int:
        return self.coarse_side * self.refinement


def _lattice(side: int, period: float, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    axes = torch.arange(side, dtype=torch.long)
    ijk = torch.stack(torch.meshgrid(axes, axes, axes, indexing="ij"), dim=-1).reshape(-1, 3)
    return ijk, ijk.to(dtype=dtype) * (period / side)


def periodic_cartesian_edges(side: int) -> torch.Tensor:
    """Six directed nearest neighbors per node, including periodic wrap."""
    if side < 3:
        raise ValueError("periodic Cartesian side must be >= 3")
    grid, _ = _lattice(side, 1.0, torch.float32)
    target = torch.arange(side**3, dtype=torch.long)
    parts = []
    for axis in range(3):
        for shift in (-1, 1):
            neighbor = grid.clone()
            neighbor[:, axis] = (neighbor[:, axis] + shift) % side
            source = (neighbor[:, 0] * side + neighbor[:, 1]) * side + neighbor[:, 2]
            parts.append(torch.stack((source, target), dim=0))
    edges = torch.cat(parts, dim=1)
    return torch.unique(edges.t().contiguous(), dim=0).t().contiguous()


def build_hit_two_scale_geometry(
    *,
    coarse_side: int,
    refinement: int,
    periodic_length: float = 1.0,
    dtype: torch.dtype = torch.float32,
    max_fine_nodes: int = 100_000,
) -> HITTwoScaleGeometry:
    """Construct a nested, periodic fine grid and sparse interpolation supports.

    Node coordinates use physical ordering (i,j,k), with periodic endpoint
    omitted. Fine-to-coarse attention uses the same weighted bipartite support
    as coarse-to-fine interpolation, reversed. It is *not* the physical Q.
    """
    if isinstance(coarse_side, bool) or not isinstance(coarse_side, int) or coarse_side < 3:
        raise ValueError("coarse_side must be an integer >= 3")
    if isinstance(refinement, bool) or not isinstance(refinement, int) or refinement < 2:
        raise ValueError("refinement must be an integer >= 2")
    if not 0 < periodic_length < float("inf"):
        raise ValueError("periodic_length must be finite and positive")
    fine_side = coarse_side * refinement
    if fine_side**3 > max_fine_nodes:
        raise ValueError("qualification geometry too large; use partitioned M41 extraction")

    _, coarse_coords = _lattice(coarse_side, periodic_length, dtype)
    fine_ijk, fine_coords = _lattice(fine_side, periodic_length, dtype)
    coarse_cell = fine_ijk // refinement
    alpha = (fine_ijk % refinement).to(dtype=dtype) / refinement
    corners = torch.tensor(list(product((0, 1), repeat=3)), dtype=torch.long)
    coarse_ijk = (coarse_cell[:, None, :] + corners[None, :, :]) % coarse_side
    columns = (coarse_ijk[..., 0] * coarse_side + coarse_ijk[..., 1]) * coarse_side + coarse_ijk[..., 2]
    weights = torch.where(corners[None, :, :] == 1, alpha[:, None, :], 1 - alpha[:, None, :]).prod(-1)
    valid = weights > 0
    fine_rows = torch.arange(fine_side**3, dtype=torch.long)[:, None].expand_as(columns)[valid]
    coarse_cols = columns[valid].contiguous()
    cross_weights = weights[valid].contiguous()
    coarse_to_fine = torch.stack((coarse_cols, fine_rows), dim=0)
    fine_to_coarse = torch.stack((fine_rows, coarse_cols), dim=0)
    counts = torch.bincount(fine_rows, minlength=fine_side**3)
    sums = torch.zeros_like(counts, dtype=dtype).index_add_(0, fine_rows, cross_weights)
    if not torch.all(counts > 0) or not torch.allclose(sums, torch.ones_like(sums), atol=1e-6):
        raise RuntimeError("invalid interpolation support")

    coarse_edges = periodic_cartesian_edges(coarse_side)
    fine_edges = periodic_cartesian_edges(fine_side)
    return HITTwoScaleGeometry(
        coarse_coords=coarse_coords,
        fine_coords=fine_coords,
        coarse_edges=coarse_edges,
        fine_edges=fine_edges,
        coarse_dilated=exact_two_hop_edge_index(coarse_edges, coarse_side**3),
        fine_dilated=exact_two_hop_edge_index(fine_edges, fine_side**3),
        coarse_to_fine=coarse_to_fine,
        fine_to_coarse=fine_to_coarse,
        cross_weights=cross_weights,
        refinement=refinement,
        coarse_side=coarse_side,
        periodic_length=periodic_length,
    )

