"""M44 SR-0 geometry-only pilot: qualified Cartesian hexahedral DNS patches.

This is NOT a DNS-to-LES physical restriction. It constructs a deterministic
point-sampled coarse lattice and trilinear coarse-to-fine transfer for a
small, explicitly qualified Cartesian patch of an existing fine hexa mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_HEX_CORNERS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
_HEX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
              (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


@dataclass(frozen=True, slots=True)
class SR0CartesianPatch:
    """Geometry and sparse interpolation of one fine Cartesian patch."""

    fine_node_ids: torch.Tensor  # original DNS node ordering
    fine_coords: torch.Tensor
    fine_hexes: torch.Tensor  # canonical local vertex ordering
    coarse_local_ids: torch.Tensor  # coarse nodes indexed in fine patch
    coarse_coords: torch.Tensor
    transfer_rowptr: torch.Tensor
    transfer_columns: torch.Tensor
    transfer_weights: torch.Tensor
    fine_side: int
    coarse_side: int
    refinement: int
    origin: tuple[int, int, int]


def build_sr0_cartesian_patch(
    coords: torch.Tensor,
    hexes: torch.Tensor,
    *,
    side: int,
    patch_cells: int = 8,
    refinement: int = 2,
    origin: tuple[int, int, int] = (0, 0, 0),
    tolerance_fraction: float = 1.0e-3,
) -> SR0CartesianPatch:
    """Qualify a structured DNS patch and build point-sampling/trilinear transfer.

    The full native vertex numbering may be arbitrarily permuted. The routine
    checks the Cartesian lattice and the *native* hexahedral connectivity in
    the selected patch, including the M37 canonical 12-edge convention.
    No periodic edges, filtering, conservative restriction, or mesh-topology
    generation outside the selected physical patch are implied.
    """
    if isinstance(side, bool) or not isinstance(side, int) or side < 3:
        raise ValueError("side must be an integer >= 3")
    if (isinstance(patch_cells, bool) or not isinstance(patch_cells, int)
            or patch_cells < 2 or patch_cells > side - 1):
        raise ValueError("patch_cells must lie in [2, side-1]")
    if (isinstance(refinement, bool) or not isinstance(refinement, int)
            or refinement < 2 or patch_cells % refinement):
        raise ValueError("refinement must be >=2 and divide patch_cells")
    if (not isinstance(origin, tuple) or len(origin) != 3
            or any(isinstance(v, bool) or not isinstance(v, int) for v in origin)
            or any(v < 0 or v + patch_cells > side - 1 for v in origin)):
        raise ValueError("origin must select a complete interior patch")
    if not 0 < tolerance_fraction < 0.5:
        raise ValueError("tolerance_fraction must lie in (0, 0.5)")
    if (coords.ndim != 2 or coords.shape != (side**3, 3)
            or not coords.is_floating_point() or not bool(torch.isfinite(coords).all())):
        raise ValueError("coords must be finite [side**3, 3] Cartesian coordinates")
    if (hexes.ndim != 2 or hexes.shape != ((side - 1)**3, 8)
            or hexes.dtype != torch.long or hexes.device != coords.device):
        raise ValueError("hexes must have zero-based int64 [(side-1)**3, 8] shape")
    if int(hexes.min()) < 0 or int(hexes.max()) >= coords.shape[0]:
        raise ValueError("hexahedral connectivity contains invalid node indices")
    if coords.device.type != "cpu":
        raise ValueError("SR-0 pilot mesh qualification runs on CPU before GPU transfer")

    lower = coords.amin(dim=0).to(torch.float64)
    extent = coords.amax(dim=0).to(torch.float64) - lower
    if bool((extent <= 0).any()):
        raise ValueError("Cartesian grid must have positive extent on each axis")
    spacing = extent / (side - 1)
    grid_float = (coords.to(torch.float64) - lower) / spacing
    grid = grid_float.round().to(torch.long)
    if (bool(((grid_float - grid).abs() > tolerance_fraction).any())
            or bool((grid < 0).any()) or bool((grid >= side).any())):
        raise ValueError("coordinates do not match a regular Cartesian lattice")
    linear = (grid[:, 0] * side + grid[:, 1]) * side + grid[:, 2]
    if not torch.equal(torch.sort(linear).values, torch.arange(side**3)):
        raise ValueError("Cartesian node lattice must be a complete bijection")
    grid_to_native = torch.empty_like(linear)
    grid_to_native[linear] = torch.arange(linear.numel())

    start = torch.tensor(origin, dtype=torch.long)
    stop = start + patch_cells
    first_corner = grid[hexes[:, 0]]
    plausible = ((first_corner >= start) & (first_corner <= stop)).all(dim=1)
    candidates = grid[hexes[plausible]]
    min_corner = candidates.amin(dim=1)
    max_corner = candidates.amax(dim=1)
    inside = ((min_corner >= start) & (max_corner <= stop)).all(dim=1)
    selected = candidates[inside]
    selected_min = min_corner[inside]
    selected_max = max_corner[inside]
    if selected.shape[0] != patch_cells**3:
        raise ValueError("native hexahedral mesh does not cover the selected patch")
    if bool((selected_max - selected_min != 1).any()):
        raise ValueError("selected native cells are not Cartesian unit hexahedra")
    bits_xyz = selected - selected_min[:, None, :]
    bits = bits_xyz[:, :, 0] * 4 + bits_xyz[:, :, 1] * 2 + bits_xyz[:, :, 2]
    if not torch.equal(torch.sort(bits, dim=1).values,
                       torch.arange(8).expand(selected.shape[0], 8)):
        raise ValueError("native cell does not have eight distinct corners")
    pairs = torch.tensor(_HEX_EDGES, dtype=torch.long)
    if bool(((selected[:, pairs[:, 0]] - selected[:, pairs[:, 1]]).abs().sum(dim=2) != 1).any()):
        raise ValueError("native cell edges do not match M37 Cartesian hex topology")
    low = selected_min - start
    cell_keys = (low[:, 0] * patch_cells + low[:, 1]) * patch_cells + low[:, 2]
    if not torch.equal(torch.sort(cell_keys).values, torch.arange(patch_cells**3)):
        raise ValueError("native mesh has missing or repeated hexahedral cells")

    n = patch_cells + 1
    axes = [torch.arange(origin[k], origin[k] + n) for k in range(3)]
    patch_grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    fine_linear = (patch_grid[:, 0] * side + patch_grid[:, 1]) * side + patch_grid[:, 2]
    fine_node_ids = grid_to_native[fine_linear]
    fine_coords = coords[fine_node_ids]

    cell_origins = torch.stack(
        torch.meshgrid(*(torch.arange(patch_cells) for _ in range(3)), indexing="ij"),
        dim=-1,
    ).reshape(-1, 3)
    corners = torch.tensor(_HEX_CORNERS, dtype=torch.long)
    cell_xyz = cell_origins[:, None, :] + corners[None, :, :]
    fine_hexes = ((cell_xyz[..., 0] * n + cell_xyz[..., 1]) * n + cell_xyz[..., 2]).long()

    nc = patch_cells // refinement + 1
    coarse_axes = [torch.arange(0, n, refinement) for _ in range(3)]
    coarse_ijk = torch.stack(torch.meshgrid(*coarse_axes, indexing="ij"), dim=-1).reshape(-1, 3)
    coarse_local_ids = (coarse_ijk[:, 0] * n + coarse_ijk[:, 1]) * n + coarse_ijk[:, 2]
    coarse_coords = fine_coords[coarse_local_ids]

    fine_ijk = torch.stack(
        torch.meshgrid(*(torch.arange(n) for _ in range(3)), indexing="ij"),
        dim=-1,
    ).reshape(-1, 3)
    cell = torch.div(fine_ijk, refinement, rounding_mode="floor").clamp(max=nc - 2)
    alpha = (fine_ijk - refinement * cell).to(torch.float64) / refinement
    parent = cell[:, None, :] + corners[None, :, :]
    column = (parent[..., 0] * nc + parent[..., 1]) * nc + parent[..., 2]
    weight = torch.where(corners[None, :, :] == 1, alpha[:, None, :], 1 - alpha[:, None, :]).prod(2)
    keep = weight > 0
    counts = keep.sum(dim=1)
    rowptr = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    return SR0CartesianPatch(
        fine_node_ids=fine_node_ids,
        fine_coords=fine_coords,
        fine_hexes=fine_hexes,
        coarse_local_ids=coarse_local_ids,
        coarse_coords=coarse_coords,
        transfer_rowptr=rowptr,
        transfer_columns=column[keep].contiguous(),
        transfer_weights=weight[keep].to(dtype=coords.dtype).contiguous(),
        fine_side=n,
        coarse_side=nc,
        refinement=refinement,
        origin=origin,
    )
