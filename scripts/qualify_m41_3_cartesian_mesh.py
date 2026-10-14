"""Strict, bounded-RAM qualification for the M41.3 Cartesian-hex fast path.

The metadata preflight alone did NOT prove ordering or topology. This scanner
verifies ALL mesh coordinates map bijectively to a regular n^3 lattice, and
ALL hexahedra are unique canonical unit cubes with the exact M37 12-edge
local connectivity. Otherwise it fails closed; no shortcut graph is allowed.
The output permutations are reusable as memory-mapped .npy arrays.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

HEX_LOCAL_EDGES = np.array([
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
], dtype=np.intp)
AXES = ("Coordinates/x", "Coordinates/y", "Coordinates/z")


def _cell_chunks(dataset: h5py.Dataset, count: int, chunk_cells: int):
    for start in range(0, count, chunk_cells):
        stop = min(count, start + chunk_cells)
        if dataset.ndim == 1:
            block = dataset[start * 8 : stop * 8].reshape(-1, 8)
        elif dataset.shape[1] == 8:
            block = dataset[start:stop, :]
        elif dataset.shape[0] == 8:
            block = dataset[:, start:stop].T
        else:
            raise ValueError("unsupported hex connectivity layout")
        yield start, np.asarray(block, dtype=np.int64)


def qualify_cartesian_hex_mesh(
    mesh_file: Path,
    output_dir: Path,
    *,
    side: int = 257,
    tolerance_fraction: float = 1.0e-3,
    chunk_nodes: int = 262144,
    chunk_cells: int = 65536,
) -> dict:
    if side < 3 or not 0 < tolerance_fraction < 0.5 or chunk_nodes < 1 or chunk_cells < 1:
        raise ValueError("invalid side, tolerance or chunk size")
    mesh_file = Path(mesh_file).expanduser().resolve(strict=True)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    n = side**3
    count = (side - 1)**3
    if n >= 2**31:
        raise ValueError("int32 grid-to-global permutation only supports fewer than 2^31 nodes")
    forward_tmp = output_dir / "global_to_cartesian.building.npy"
    inverse_tmp = output_dir / "cartesian_to_global.building.npy"
    for path in (forward_tmp, inverse_tmp):
        if path.exists():
            raise FileExistsError(f"incomplete or competing map build already exists: {path}")
    completed = False
    try:
        with h5py.File(mesh_file, "r") as mesh:
            coords = [mesh[key] for key in AXES]
            if any(ds.ndim != 1 or ds.shape[0] != n for ds in coords):
                raise ValueError(f"coordinate arrays must all have length {n}")
            connectivity = mesh["Connectivity/hex->node"]
            if not np.issubdtype(connectivity.dtype, np.integer):
                raise TypeError("connectivity must be integer")
            if connectivity.size != count * 8:
                raise ValueError(f"expected {count} hex cells, got {connectivity.size // 8}")
            lo = np.full(3, np.inf)
            hi = np.full(3, -np.inf)
            for start in range(0, n, chunk_nodes):
                stop = min(n, start + chunk_nodes)
                slab = np.column_stack([ds[start:stop] for ds in coords])
                if not np.isfinite(slab).all():
                    raise ValueError("nonfinite coordinate")
                lo = np.minimum(lo, slab.min(axis=0))
                hi = np.maximum(hi, slab.max(axis=0))
            spacing = (hi - lo) / (side - 1)
            if np.any(spacing <= 0):
                raise ValueError("degenerate coordinate axis")
            forward = np.lib.format.open_memmap(forward_tmp, mode="w+", dtype=np.int32, shape=(n,))
            inverse = np.lib.format.open_memmap(inverse_tmp, mode="w+", dtype=np.int32, shape=(n,))
            inverse[:] = -1
            worst_residual = np.zeros(3, dtype=np.float64)
            for start in range(0, n, chunk_nodes):
                stop = min(n, start + chunk_nodes)
                slab = np.column_stack([ds[start:stop] for ds in coords])
                scaled = (slab - lo) / spacing
                grid = np.rint(scaled).astype(np.int32)
                residual = np.abs(scaled - grid)
                worst_residual = np.maximum(worst_residual, residual.max(axis=0))
                if np.any(residual > tolerance_fraction) or np.any(grid < 0) or np.any(grid >= side):
                    raise ValueError(
                        f"coordinate lattice mismatch in nodes [{start},{stop}), "
                        f"max fractional spacing residual {worst_residual.tolist()}"
                    )
                linear = ((grid[:, 0].astype(np.int64) * side + grid[:, 1]) * side + grid[:, 2])
                if len(np.unique(linear)) != len(linear) or np.any(inverse[linear] != -1):
                    raise ValueError(f"duplicate Cartesian coordinate detected among node IDs near {start}")
                forward[start:stop] = linear.astype(np.int32)
                inverse[linear] = np.arange(start, stop, dtype=np.int32)
            if np.any(inverse == -1):
                raise ValueError("Cartesian coordinate mapping has missing grid nodes")
            forward.flush()
            inverse.flush()

            visited = np.zeros(count, dtype=np.bool_)
            for start, cell in _cell_chunks(connectivity, count, chunk_cells):
                stop = start + len(cell)
                if np.any(cell < 1) or np.any(cell > n):
                    raise ValueError(f"connectivity is not one-based/in range in cells [{start},{stop})")
                linear = forward[cell - 1]
                xyz = np.stack((linear // (side*side), (linear // side) % side, linear % side), axis=-1)
                lower = xyz.min(axis=1)
                upper = xyz.max(axis=1)
                if np.any(upper - lower != 1):
                    raise ValueError(f"non-unit hexahedral cell in [{start},{stop})")
                bits = xyz - lower[:, None, :]
                bits = (bits[:, :, 0] * 4 + bits[:, :, 1] * 2 + bits[:, :, 2])
                if np.any(np.sort(bits, axis=1) != np.arange(8)[None, :]):
                    raise ValueError(f"hexahedron lacks exactly eight lattice corners in [{start},{stop})")
                # Crucial: the ORIGINAL M37 graph uses these exact local edges.
                # Even valid cube vertex sets may have a different vertex order.
                for a, b in HEX_LOCAL_EDGES:
                    distance = np.abs(xyz[:, a, :] - xyz[:, b, :]).sum(axis=1)
                    if np.any(distance != 1):
                        raise ValueError(f"M37 canonical local hex edge ({a},{b}) is not a Cartesian axis neighbor in [{start},{stop})")
                key = ((lower[:, 0] * (side - 1) + lower[:, 1]) * (side - 1) + lower[:, 2]).astype(np.int64)
                if len(np.unique(key)) != len(key) or np.any(visited[key]):
                    raise ValueError(f"duplicate physical hexahedral cells in [{start},{stop})")
                visited[key] = True
            if not np.all(visited):
                raise ValueError("missing physical hexahedral cells")
            del forward, inverse
        os.replace(forward_tmp, output_dir / "global_to_cartesian.npy")
        os.replace(inverse_tmp, output_dir / "cartesian_to_global.npy")
        result = {
            "status": "qualified_complete_nonperiodic_cartesian_hex_graph",
            "mesh_file": str(mesh_file),
            "num_nodes": n,
            "num_cells_verified": count,
            "coordinate_lower": lo.tolist(),
            "coordinate_upper": hi.tolist(),
            "spacing": spacing.tolist(),
            "max_coordinate_error_in_cell_widths": worst_residual.tolist(),
            "max_tolerance_fraction": tolerance_fraction,
            "connectivity_indexing": "one_based",
            "cell_vertex_permutation": "M37_hex_local_edges_verified",
            "global_to_cartesian": str(output_dir / "global_to_cartesian.npy"),
            "cartesian_to_global": str(output_dir / "cartesian_to_global.npy"),
            "periodicity_caveat": "No wraparound edges occur in the scanned hex->node topology; separate solver periodic maps are not inspected.",
            "fast_path_scope": "Cartesian grid graph equivalent to M37 edge extraction, not yet GPU inference",
        }
        (output_dir / "cartesian_qualification.json").write_text(json.dumps(result, indent=2) + "\n")
        completed = True
        return result
    finally:
        if not completed:
            for path in (forward_tmp, inverse_tmp):
                path.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--side", type=int, default=257)
    p.add_argument("--tolerance-fraction", type=float, default=1.0e-3)
    p.add_argument("--chunk-nodes", type=int, default=262144)
    p.add_argument("--chunk-cells", type=int, default=65536)
    args = p.parse_args()
    print(json.dumps(qualify_cartesian_hex_mesh(
        args.mesh_file, args.output_dir, side=args.side,
        tolerance_fraction=args.tolerance_fraction,
        chunk_nodes=args.chunk_nodes, chunk_cells=args.chunk_cells,
    ), indent=2))


if __name__ == "__main__":
    main()
