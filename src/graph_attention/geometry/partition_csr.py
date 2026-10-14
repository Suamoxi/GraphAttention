"""Mesh-independent owned-query/exact-two-hop partition extraction.

Only the one-hop incoming CSR is global. Exact-two-hop edges are constructed
for owned targets in bounded batches and persisted per partition. Neither
Cartesian node order nor geometric regularity is assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .external_csr import DiskCSR


def spatial_owners(
    coordinates: np.ndarray,
    *,
    blocks: tuple[int, ...],
    chunk_nodes: int = 262144,
) -> np.ndarray:
    """Same bounding-box bin semantics as coordinate_block_partition_ids.

    Returned IDs are compacted in ascending spatial-bin order. This works
    equally for irregular hexahedral, tetrahedral and Cartesian CFD meshes.
    """
    coords = np.asarray(coordinates)
    if coords.ndim != 2 or len(blocks) != coords.shape[1] or any(k <= 0 for k in blocks):
        raise ValueError("blocks must have one positive entry per coordinate dimension")
    if not np.isfinite(coords).all():
        raise ValueError("coordinates contain NaN or Inf")
    lo = np.min(coords, axis=0)
    span = np.max(coords, axis=0) - lo
    if np.any(span <= 0):
        raise ValueError("degenerate coordinate bounding box")
    n = len(coords)
    result = np.empty(n, dtype=np.int32)
    for start in range(0, n, chunk_nodes):
        stop = min(n, start + chunk_nodes)
        ratios = (coords[start:stop].astype(np.float64) - lo) / span
        labels = np.minimum(
            np.floor(ratios * np.asarray(blocks)).astype(np.int64),
            np.asarray(blocks) - 1,
        )
        ids = labels[:, 0].copy()
        for axis in range(1, len(blocks)):
            ids = ids * blocks[axis] + labels[:, axis]
        result[start:stop] = ids.astype(np.int32)
    occupied = np.unique(result)
    return np.searchsorted(occupied, result).astype(np.int32)


def _expand_incoming(csr: DiskCSR, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized source/target expansion of incoming rows for a target batch."""
    targets = np.asarray(targets, dtype=np.int64)
    starts = csr.indptr[targets]
    counts = (csr.indptr[targets+1] - starts).astype(np.int64)
    offsets = np.cumsum(counts, dtype=np.int64) - counts
    indices = np.repeat(starts, counts) + (
        np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(offsets, counts)
    )
    return (
        np.asarray(csr.indices[indices], dtype=np.int64),
        np.repeat(targets, counts),
    )


def _edge_keys(source: np.ndarray, target: np.ndarray, n: int) -> np.ndarray:
    return target.astype(np.uint64)*np.uint64(n) + source.astype(np.uint64)


def owned_query_edge_keys(
    csr: DiskCSR, owned: np.ndarray, *, chunk_targets: int = 8192,
    local_self: bool = False, dilated_self: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (one-hop, exact-two-hop) edge keys, both target-major sorted.

    For target j: incoming one-hop keys are k->j; incoming exact-two-hop
    keys are i->j for any i->k->j, with self and all direct edges removed.
    Multiple paths to one source are deduplicated, exactly as in M37.
    """
    n = csr.num_nodes
    if chunk_targets < 1:
        raise ValueError("chunk_targets must be positive")
    if owned.ndim != 1 or owned.dtype.kind not in "iu":
        raise ValueError("owned indices must be a 1D integer array")
    if np.any(owned < 0) or np.any(owned >= n):
        raise ValueError("owned target indices out of range")
    parts1, parts2 = [], []
    for start in range(0, len(owned), chunk_targets):
        targets = owned[start:start+chunk_targets].astype(np.int64)
        first_source, first_target = _expand_incoming(csr, targets)
        direct = np.unique(_edge_keys(first_source, first_target, n))
        walk_count = int(np.sum(csr.indptr[first_source+1] - csr.indptr[first_source], dtype=np.int64))
        if walk_count > 2000000:
            raise MemoryError(
                "more than two million length-two paths in one owned-query batch; "
                "reduce --chunk-targets to bound intermediate memory"
            )
        second_source, second_target = _expand_incoming(csr, first_source)
        # second_target is the intermediate key/node, not the final target.
        del second_target
        # Every first edge k->j has one target j; expand by incoming degree of k.
        incoming_degree = csr.indptr[first_source+1] - csr.indptr[first_source]
        path_target = np.repeat(first_target, incoming_degree)
        candidates = _edge_keys(second_source, path_target, n)
        nonself = second_source != path_target
        candidates = np.unique(candidates[nonself])
        exact = np.setdiff1d(candidates, direct, assume_unique=True)
        if local_self:
            direct = np.union1d(direct, _edge_keys(targets, targets, n))
        if dilated_self:
            exact = np.union1d(exact, _edge_keys(targets, targets, n))
        parts1.append(direct)
        parts2.append(exact)
    return (
        np.concatenate(parts1) if parts1 else np.empty(0, np.uint64),
        np.concatenate(parts2) if parts2 else np.empty(0, np.uint64),
    )


def build_partition_files(
    csr: DiskCSR, owners: np.ndarray, *,
    output_dir: Path,
    chunk_targets: int = 8192,
    max_owned_nodes: int = 400000,
    include_self_attention: bool = False,
    include_local_self_attention: bool = False,
) -> dict:
    """Write partition edges to disk, never materializing global A^2.

    The ownership array may contain any contiguous partition IDs. A caller
    supplies those IDs, so spatial partitioning is *not* a model constraint.
    """
    owners = np.asarray(owners)
    n = csr.num_nodes
    if owners.shape != (n,) or owners.dtype.kind not in "iu":
        raise ValueError("owners must be a length-N integer array")
    pids = np.unique(owners)
    if not np.array_equal(pids, np.arange(len(pids))):
        raise ValueError("partition IDs must be contiguous from zero")
    if max_owned_nodes < 1:
        raise ValueError("max_owned_nodes must be positive")
    directory = Path(output_dir).expanduser().resolve()
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite existing partition plan: {directory}")
    directory.mkdir(parents=True)
    partitions = []
    for pid in range(len(pids)):
        owned = np.flatnonzero(owners == pid).astype(np.int64)
        if len(owned) > max_owned_nodes:
            raise ValueError(
                f"partition {pid} owns {len(owned)} nodes; raise partition count "
                f"or --max-owned-nodes={len(owned)} explicitly"
            )
        direct, exact = owned_query_edge_keys(
            csr, owned, chunk_targets=chunk_targets,
            local_self=include_self_attention or include_local_self_attention,
            dilated_self=include_self_attention,
        )
        source1, target1 = (direct % np.uint64(n)), (direct // np.uint64(n))
        source2, target2 = (exact % np.uint64(n)), (exact // np.uint64(n))
        read = np.unique(np.concatenate((
            owned, source1.astype(np.int64), source2.astype(np.int64),
        )))
        if n >= 2**31:
            dtype = np.int64
        else:
            dtype = np.int32
        np.savez(
            directory / f"partition_{pid:05d}.npz",
            owned=owned.astype(dtype), read=read.astype(dtype),
            local_source=np.searchsorted(read, source1).astype(dtype),
            local_target=np.searchsorted(read, target1).astype(dtype),
            dilated_source=np.searchsorted(read, source2).astype(dtype),
            dilated_target=np.searchsorted(read, target2).astype(dtype),
        )
        partitions.append({
            "id": pid, "owned_nodes": int(len(owned)),
            "read_nodes": int(len(read)), "halo_nodes": int(len(read)-len(owned)),
            "local_edges": int(len(direct)), "dilated_edges": int(len(exact)),
        })
    manifest = {
        "status": "complete", "num_nodes": n,
        "csr_directory": str(csr.directory),
        "num_partitions": len(partitions),
        "include_self_attention": bool(include_self_attention),
        "include_local_self_attention": bool(include_local_self_attention),
        "parts": partitions,
        "notes": "exact M37 incoming local and exact-two-hop neighborhoods; no Cartesian assumptions",
    }
    (directory / "partitions.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest
