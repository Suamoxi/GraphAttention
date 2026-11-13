"""Target-driven periodic two-scale attention geometry for M48 real HIT cores.

Construct *only* the graph dependencies of requested owned fine nodes,
without allocating all periodic fine/coarse edges. Every edge uses global
lattice IDs until final patch-local remapping. Cross-scale support is the
existing M45 trilinear attention support, NOT the MS_SR CEDAR observation.
No physical CEDAR projection is attempted on a patch.

The backward closure matches two_scale_patch_dependencies; comparison with
the full global graph is required before use in scientific training.
"""
from __future__ import annotations

from itertools import product

import torch

from .two_scale_partition import TwoScalePatchGeometry, TwoScalePatchPlan

_ONE = tuple(
    tuple(s if j == axis else 0 for j in range(3))
    for axis in range(3) for s in (-1, 1)
)
_TWO = (
    tuple(tuple(s * 2 if j == axis else 0 for j in range(3))
          for axis in range(3) for s in (-1, 1))
    + tuple(tuple(s0 if j == a else s1 if j == b else 0 for j in range(3))
            for a in range(3) for b in range(a + 1, 3)
            for s0 in (-1, 1) for s1 in (-1, 1))
)
_CORNERS = torch.tensor(list(product((0, 1), repeat=3)), dtype=torch.long)


def _ijk(ids: torch.Tensor, side: int) -> torch.Tensor:
    return torch.stack((
        torch.div(ids, side * side, rounding_mode="floor"),
        torch.div(ids, side, rounding_mode="floor") % side,
        ids % side,
    ), dim=-1)


def _ids(ijk: torch.Tensor, side: int) -> torch.Tensor:
    return (ijk[..., 0] * side + ijk[..., 1]) * side + ijk[..., 2]


def _unique(ids: torch.Tensor) -> torch.Tensor:
    return torch.unique(ids.to(dtype=torch.long), sorted=True)


def _incoming(targets: torch.Tensor, side: int, *, dilated: bool) -> torch.Tensor:
    """Global [source,target], ordered and deduplicated, for only these targets.

    Exactly-two-hop edges are walks of graph distance exactly 2; on a side-3
    periodic grid, some length-two walks become direct and must be removed.
    """
    if targets.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    coords = _ijk(targets, side)
    offsets = torch.as_tensor(_TWO if dilated else _ONE, dtype=torch.long)
    src = _ids((coords[:, None] + offsets[None]) % side, side).reshape(-1)
    trg = targets[:, None].expand(-1, offsets.shape[0]).reshape(-1)
    total = side**3
    keys = _unique(src * total + trg)
    if dilated:
        # Remove self and all actual first-hop neighbors, not just Manhattan
        # distance 1 in the unwrapped coordinates.
        one = _incoming(targets, side, dilated=False)
        direct = _unique(one[0] * total + one[1])
        pos = torch.searchsorted(direct, keys)
        is_direct = (pos < direct.numel()) & (direct[pos.clamp(max=direct.numel()-1)] == keys)
        keys = keys[(keys // total != keys % total) & ~is_direct]
    return torch.stack((keys // total, keys % total))


def _closure(targets: torch.Tensor, side: int, layers: int) -> torch.Tensor:
    nodes = _unique(targets)
    if layers < 0:
        raise ValueError("layers must be nonnegative")
    for index in reversed(range(layers)):
        incoming = _incoming(nodes, side, dilated=bool(index % 2))
        nodes = _unique(torch.cat((nodes, incoming[0])))
    return nodes


def _cross_up(fine: torch.Tensor, nc: int, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    """All trilinear (coarse source -> fine query) edges of specified fine nodes."""
    nf = nc * r
    fine_ijk = _ijk(fine, nf)
    cell = torch.div(fine_ijk, r, rounding_mode="floor")
    alpha = fine_ijk.remainder(r).to(torch.float64) / r
    corners = _CORNERS
    parents = (cell[:, None] + corners[None]) % nc
    weights = torch.where(corners[None] > 0, alpha[:, None], 1 - alpha[:, None]).prod(-1)
    source = _ids(parents, nc)
    valid = weights > 0
    edge = torch.stack((source[valid], fine[:, None].expand_as(source)[valid]))
    return edge, weights[valid]


def _cross_down_sources(coarse: torch.Tensor, nc: int, r: int) -> torch.Tensor:
    """All fine sources interpolating into a coarse query (no global graph)."""
    offsets = torch.tensor(
        list(product(range(1-r, r), repeat=3)), dtype=torch.long
    )
    fine_ijk = (_ijk(coarse, nc)[:, None] * r + offsets[None]) % (nc * r)
    return _unique(_ids(fine_ijk, nc * r).reshape(-1))


def _induced_edges(nodes: torch.Tensor, side: int, *, dilated: bool) -> torch.Tensor:
    # Query set is exactly the read set. Missing incoming keys of halo-only
    # queries are permissible because they cannot influence owned outputs.
    full = _incoming(nodes, side, dilated=dilated)
    pos = torch.searchsorted(nodes, full[0])
    valid = (pos < nodes.numel()) & (nodes[pos.clamp(max=nodes.numel()-1)] == full[0])
    return torch.stack((
        pos[valid],
        torch.searchsorted(nodes, full[1, valid]),
    ))


def build_targeted_two_scale_patch(
    *, coarse_side: int, refinement: int, owned_fine: torch.Tensor,
    fine_pre_layers: int, fine_post_layers: int, coarse_static_layers: int,
    coarse_dynamic_layers: int = 0, periodic_length: float = 1.0,
    dtype: torch.dtype = torch.float32, max_fine_read: int = 100_000,
    max_coarse_read: int = 50_000,
) -> tuple[TwoScalePatchPlan, TwoScalePatchGeometry]:
    """Return exact read sets and induced graphs without a global edge list.

    This builder is CPU-only and designed for periodic cubic Cartesian grids.
    Requires source graph periodicity, not artificial periodic patch faces.
    """
    if (coarse_side < 3 or refinement < 2 or periodic_length <= 0
            or max_fine_read < 1 or max_coarse_read < 1):
        raise ValueError("invalid periodic geometry settings")
    for n in (fine_pre_layers, fine_post_layers,
              coarse_static_layers, coarse_dynamic_layers):
        if not isinstance(n, int) or n < 0:
            raise ValueError("layer counts must be nonnegative integers")
    if (owned_fine.ndim != 1 or owned_fine.dtype != torch.long
            or owned_fine.device.type != "cpu" or owned_fine.numel() == 0):
        raise ValueError("owned_fine must be nonempty CPU int64 [N]")
    nc, r, nf = coarse_side, refinement, coarse_side * refinement
    owned = _unique(owned_fine)
    if int(owned[0]) < 0 or int(owned[-1]) >= nf**3:
        raise ValueError("owned fine IDs out of range")

    fine_post = _closure(owned, nf, fine_post_layers)
    up_edges, _ = _cross_up(fine_post, nc, r)
    coarse_up = _unique(up_edges[0])
    if coarse_dynamic_layers:
        coarse_dynamic = _closure(coarse_up, nc, coarse_dynamic_layers)
        fine_down = _cross_down_sources(coarse_dynamic, nc, r)
        coarse_needed = coarse_dynamic
    else:
        coarse_dynamic = torch.empty(0, dtype=torch.long)
        fine_down = torch.empty(0, dtype=torch.long)
        coarse_needed = coarse_up
    coarse_read = _closure(coarse_needed, nc, coarse_static_layers)
    fine_read = _closure(
        torch.cat((fine_post, fine_down)), nf, fine_pre_layers,
    )

    if fine_read.numel() > max_fine_read or coarse_read.numel() > max_coarse_read:
        raise ValueError(
            f"patch read set exceeds safety cap: fine={fine_read.numel()}/"
            f"{max_fine_read}, coarse={coarse_read.numel()}/{max_coarse_read}"
        )

    up_global, up_weight = _cross_up(fine_read, nc, r)
    coarse_pos = torch.searchsorted(coarse_read, up_global[0])
    keep = ((coarse_pos < coarse_read.numel())
            & (coarse_read[coarse_pos.clamp(max=coarse_read.numel()-1)] == up_global[0]))
    fine_pos = torch.searchsorted(fine_read, up_global[1, keep])
    up = torch.stack((coarse_pos[keep], fine_pos))
    # These are the same edges in reverse, exactly matching M45.
    down = up.flip(0).contiguous()
    weights = up_weight[keep].to(dtype=dtype)

    plan = TwoScalePatchPlan(
        owned_fine=owned, fine_post_queries=fine_post,
        coarse_up_keys=coarse_up, coarse_dynamic_inputs=coarse_dynamic,
        fine_down_keys=fine_down, fine_read=fine_read,
        coarse_read=coarse_read,
    )
    geometry = TwoScalePatchGeometry(
        fine_global=fine_read, coarse_global=coarse_read,
        owned_fine_local=torch.searchsorted(fine_read, owned),
        fine_coords=_ijk(fine_read, nf).to(dtype) * (periodic_length / nf),
        coarse_coords=_ijk(coarse_read, nc).to(dtype) * (periodic_length / nc),
        fine_edges=_induced_edges(fine_read, nf, dilated=False),
        fine_dilated=_induced_edges(fine_read, nf, dilated=True),
        coarse_edges=_induced_edges(coarse_read, nc, dilated=False),
        coarse_dilated=_induced_edges(coarse_read, nc, dilated=True),
        coarse_to_fine=up, fine_to_coarse=down,
        cross_weights=weights, periodic_length=periodic_length,
        coarse_spacing=periodic_length / nc, refinement=r,
    )
    # All owned fine queries must have their complete interpolation support.
    owned_mask = torch.isin(up[1], geometry.owned_fine_local)
    mass = torch.zeros(fine_read.numel(), dtype=dtype)
    mass.index_add_(0, up[1, owned_mask], weights[owned_mask])
    if not torch.allclose(
        mass[geometry.owned_fine_local],
        torch.ones(len(owned), dtype=dtype),
        atol=1e-6, rtol=1e-6,
    ):
        raise RuntimeError("missing coarse attention sources for an owned fine node")
    return plan, geometry
