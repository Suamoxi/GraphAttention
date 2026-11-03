"""Exact input dependency closure for one static/dynamic two-scale DiNAT core.

All edges are global [source, target] graph indices. This small-graph planner
validates semantics; production must fetch incoming rows from disk-backed CSR.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TwoScalePatchPlan:
    owned_fine: torch.Tensor
    fine_post_queries: torch.Tensor
    coarse_up_keys: torch.Tensor
    coarse_dynamic_inputs: torch.Tensor
    fine_down_keys: torch.Tensor
    fine_read: torch.Tensor
    coarse_read: torch.Tensor


def _unique(values: torch.Tensor) -> torch.Tensor:
    return torch.unique(values.to(dtype=torch.long), sorted=True)


def _sources(targets: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    if targets.numel() == 0 or edges.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=targets.device)
    return _unique(edges[0, torch.isin(edges[1], targets)])


def _dependency_closure(
    targets: torch.Tensor, local: torch.Tensor, dilated: torch.Tensor, layers: int
) -> torch.Tensor:
    """Backward walk through local/exact-two-hop blocks, local first."""
    nodes = _unique(targets)
    for index in reversed(range(layers)):
        attention = local if index % 2 == 0 else dilated
        nodes = _unique(torch.cat((nodes, _sources(nodes, attention))))
    return nodes


def two_scale_patch_dependencies(
    owned_fine: torch.Tensor,
    *,
    fine_local: torch.Tensor,
    fine_dilated: torch.Tensor,
    coarse_local: torch.Tensor,
    coarse_dilated: torch.Tensor,
    coarse_to_fine: torch.Tensor,
    fine_to_coarse: torch.Tensor,
    fine_pre_layers: int,
    fine_post_layers: int,
    coarse_static_layers: int,
    coarse_dynamic_layers: int = 0,
) -> TwoScalePatchPlan:
    """Return exact dependencies for 1 up pass and optional 1 down pass.

    A coarse node can be read by arbitrarily many fine partitions. All input
    sets use global node IDs; losses are restricted to owned_fine.
    """
    for name, layers in (
        ("fine_pre_layers", fine_pre_layers),
        ("fine_post_layers", fine_post_layers),
        ("coarse_static_layers", coarse_static_layers),
        ("coarse_dynamic_layers", coarse_dynamic_layers),
    ):
        if not isinstance(layers, int) or layers < 0:
            raise ValueError(f"{name} must be nonnegative")
    owned = _unique(owned_fine)
    if owned.numel() == 0:
        raise ValueError("at least one owned fine node is required")
    fine_post = _dependency_closure(owned, fine_local, fine_dilated, fine_post_layers)
    coarse_up = _sources(fine_post, coarse_to_fine)
    if coarse_up.numel() == 0:
        raise ValueError("fine core has no coarse conditioning")
    if fine_to_coarse.numel() and coarse_dynamic_layers > 0:
        coarse_dyn = _dependency_closure(coarse_up, coarse_local, coarse_dilated, coarse_dynamic_layers)
        fine_down = _sources(coarse_dyn, fine_to_coarse)
        coarse_needed = coarse_dyn
    else:
        coarse_dyn = torch.empty(0, dtype=torch.long, device=owned.device)
        fine_down = torch.empty(0, dtype=torch.long, device=owned.device)
        coarse_needed = coarse_up
    coarse_read = _dependency_closure(
        coarse_needed, coarse_local, coarse_dilated, coarse_static_layers
    )
    fine_read = _dependency_closure(
        _unique(torch.cat((fine_post, fine_down))), fine_local, fine_dilated, fine_pre_layers
    )
    return TwoScalePatchPlan(
        owned, fine_post, coarse_up, coarse_dyn, fine_down, fine_read, coarse_read
    )


@dataclass(frozen=True)
class TwoScalePatchGeometry:
    """Induced local graphs with global-to-local IDs and stage ownership."""

    fine_global: torch.Tensor
    coarse_global: torch.Tensor
    owned_fine_local: torch.Tensor
    fine_coords: torch.Tensor
    coarse_coords: torch.Tensor
    fine_edges: torch.Tensor
    fine_dilated: torch.Tensor
    coarse_edges: torch.Tensor
    coarse_dilated: torch.Tensor
    coarse_to_fine: torch.Tensor
    fine_to_coarse: torch.Tensor
    cross_weights: torch.Tensor
    periodic_length: float
    coarse_spacing: float
    refinement: int


def _induced(edges: torch.Tensor, source_nodes: torch.Tensor, target_nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter global edges and remap their endpoints to sorted local node IDs."""
    keep = torch.isin(edges[0], source_nodes) & torch.isin(edges[1], target_nodes)
    selected = edges[:, keep]
    return torch.stack((
        torch.searchsorted(source_nodes, selected[0]),
        torch.searchsorted(target_nodes, selected[1]),
    )), keep


def extract_two_scale_patch_geometry(geometry, plan: TwoScalePatchPlan) -> TwoScalePatchGeometry:
    """Slice a *small* global qualification mesh without changing its topology.

    Unlike feature-interpolation CSR, cross-attention support is allowed to be
    incomplete for irrelevant halo queries. No local renormalization is done.
    """
    fine, coarse = plan.fine_read, plan.coarse_read
    fine_edges, _ = _induced(geometry.fine_edges, fine, fine)
    fine_dilated, _ = _induced(geometry.fine_dilated, fine, fine)
    coarse_edges, _ = _induced(geometry.coarse_edges, coarse, coarse)
    coarse_dilated, _ = _induced(geometry.coarse_dilated, coarse, coarse)
    up, mask = _induced(geometry.coarse_to_fine, coarse, fine)
    down, down_mask = _induced(geometry.fine_to_coarse, fine, coarse)
    if not torch.equal(mask, down_mask):
        raise RuntimeError("two bipartite supports were not transposes")
    return TwoScalePatchGeometry(
        fine_global=fine,
        coarse_global=coarse,
        owned_fine_local=torch.searchsorted(fine, plan.owned_fine),
        fine_coords=geometry.fine_coords[fine],
        coarse_coords=geometry.coarse_coords[coarse],
        fine_edges=fine_edges,
        fine_dilated=fine_dilated,
        coarse_edges=coarse_edges,
        coarse_dilated=coarse_dilated,
        coarse_to_fine=up,
        fine_to_coarse=down,
        cross_weights=geometry.cross_weights[mask],
        refinement=geometry.refinement,
        periodic_length=geometry.periodic_length,
        coarse_spacing=geometry.coarse_spacing,
    )

