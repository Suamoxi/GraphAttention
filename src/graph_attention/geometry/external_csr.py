"""Out-of-core, mesh-independent one-hop CSR construction (M41.4).

Builds exactly the M37 physical element edges for hex meshes, and the
standard complete element edge set for tetrahedral meshes. It never builds
a complete global two-hop graph and does not assume a Cartesian node order.

CSR rows represent incoming source/key IDs indexed by query/target IDs.
External sorting is by target-node buckets; temporary disk usage is O(E).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from collections.abc import Iterable, Iterator

import h5py
import numpy as np


ELEMENT_EDGES = {
    "hex": ((0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)),
    "tet": ((0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (2, 3)),
}
ELEMENT_VERTICES = {"hex": 8, "tet": 4}


@dataclass(frozen=True)
class DiskCSR:
    """Read-only node-major CSR stored in mmap-compatible files."""
    num_nodes: int
    indptr: np.ndarray
    indices: np.ndarray
    directory: Path
    directed_edges: int

    @classmethod
    def open(cls, directory: Path) -> "DiskCSR":
        directory = Path(directory)
        metadata = json.loads((directory / "csr.json").read_text())
        if metadata.get("status") != "complete":
            raise ValueError("CSR build is not complete")
        n = int(metadata["num_nodes"])
        e = int(metadata["directed_edges"])
        indptr = np.load(directory / "indptr.npy", mmap_mode="r")
        index_dtype = np.dtype(metadata["index_dtype"])
        indices = (
            np.memmap(directory / "indices.bin", dtype=index_dtype,
                      mode="r", shape=(e,))
            if e else np.empty(0, dtype=index_dtype)
        )
        if indptr.shape != (n + 1,) or int(indptr[0]) != 0 or int(indptr[-1]) != e:
            raise ValueError("invalid CSR offsets")
        if np.any(indptr[1:] < indptr[:-1]):
            raise ValueError("CSR offsets are not monotone")
        return cls(n, indptr, indices, directory, e)


def iter_hdf5_directed_edges(
    mesh_file: Path,
    *,
    connectivity_path: str = "Connectivity/hex->node",
    element_type: str = "hex",
    one_based: bool = True,
    num_nodes: int,
    chunk_cells: int = 65536,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield incoming edge pairs (source,target), exactly M37 for hex.

    Supports flat, [cells,vertices] and [vertices,cells] HDF5 storage.
    Cells are arbitrary irregular/unstructured elements; vertex numbering
    is never inferred from Cartesian geometry.
    """
    if element_type not in ELEMENT_VERTICES:
        raise ValueError("supported element types: hex, tet")
    if chunk_cells < 1:
        raise ValueError("chunk_cells must be positive")
    count = ELEMENT_VERTICES[element_type]
    edges = np.asarray(ELEMENT_EDGES[element_type], dtype=np.intp)
    with h5py.File(mesh_file, "r") as h5:
        ds = h5[connectivity_path]
        if ds.ndim == 1 and ds.size % count == 0:
            cells = ds.size // count
            layout = "flat"
        elif ds.ndim == 2 and ds.shape[1] == count:
            cells, layout = ds.shape[0], "rows"
        elif ds.ndim == 2 and ds.shape[0] == count:
            cells, layout = ds.shape[1], "columns"
        else:
            raise ValueError(f"invalid {element_type} connectivity shape: {ds.shape}")
        if not np.issubdtype(ds.dtype, np.integer):
            raise TypeError("cell connectivity must be integer")
        for start in range(0, cells, chunk_cells):
            stop = min(cells, start + chunk_cells)
            if layout == "flat":
                vertices = np.asarray(ds[start*count:stop*count], dtype=np.int64).reshape(-1, count)
            elif layout == "rows":
                vertices = np.asarray(ds[start:stop, :], dtype=np.int64)
            else:
                vertices = np.asarray(ds[:, start:stop].T, dtype=np.int64)
            if one_based:
                if np.any(vertices < 1) or np.any(vertices > num_nodes):
                    raise ValueError(f"one-based connectivity out of range in cells {start}:{stop}")
                vertices -= 1
            elif np.any(vertices < 0) or np.any(vertices >= num_nodes):
                raise ValueError(f"zero-based connectivity out of range in cells {start}:{stop}")
            a = vertices[:, edges[:, 0]].reshape(-1)
            b = vertices[:, edges[:, 1]].reshape(-1)
            # Duplicate cell vertices are not meaningful physical graph edges.
            if np.any(a == b):
                raise ValueError(f"degenerate {element_type} edge in cells {start}:{stop}")
            yield np.concatenate((a, b)), np.concatenate((b, a))


def build_disk_csr(
    edge_blocks: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    num_nodes: int,
    output_dir: Path,
    bucket_nodes: int = 131072,
) -> dict:
    """Externally deduplicate directed edges into a disk-backed CSR.

    Never loads the whole graph: only one emitted edge block or one
    target-node bucket is sorted in memory. Caller should choose bucket_nodes
    small enough for its available RAM; no node ordering assumptions.
    """
    n = int(num_nodes)
    if n < 1 or n >= 2**32 or bucket_nodes < 1:
        raise ValueError("num_nodes must be in [1,2**32) and bucket_nodes positive")
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing CSR: {output_dir}")
    output_dir.mkdir(parents=True)
    buckets = output_dir / "_buckets"
    buckets.mkdir()
    bucket_count = (n + bucket_nodes - 1) // bucket_nodes
    widths = np.zeros(bucket_count, dtype=np.int64)
    raw_candidates = 0
    index_dtype = np.int32 if n < 2**31 else np.int64
    try:
        for sources, targets in edge_blocks:
            source = np.asarray(sources, dtype=np.int64).reshape(-1)
            target = np.asarray(targets, dtype=np.int64).reshape(-1)
            if source.shape != target.shape:
                raise ValueError("directed edge source/target lengths differ")
            if (np.any(source < 0) or np.any(target < 0)
                    or np.any(source >= n) or np.any(target >= n)):
                raise ValueError("out-of-range directed graph edge")
            nonself = source != target
            source, target = source[nonself], target[nonself]
            raw_candidates += len(source)
            if not len(source):
                continue
            # target-major 64-bit encoding: target*num_nodes + source
            keys = target.astype(np.uint64)*np.uint64(n) + source.astype(np.uint64)
            bins = target // bucket_nodes
            order = np.argsort(bins, kind="stable")
            sorted_bins, sorted_keys = bins[order], keys[order]
            boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_bins)) + 1, len(bins)]
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                which = int(sorted_bins[left])
                with (buckets / f"{which:06d}.bin").open("ab") as stream:
                    sorted_keys[left:right].tofile(stream)
                widths[which] += right - left

        ptr = np.lib.format.open_memmap(
            output_dir / "indptr.npy", mode="w+", dtype=np.int64, shape=(n+1,)
        )
        ptr[0] = 0
        total_edges = 0
        with (output_dir / "indices.bin").open("wb") as stream:
            for bucket in range(bucket_count):
                start = bucket*bucket_nodes
                stop = min(n, start+bucket_nodes)
                filename = buckets / f"{bucket:06d}.bin"
                if filename.exists():
                    keys = np.unique(np.fromfile(filename, dtype=np.uint64))
                    target = (keys // np.uint64(n)).astype(np.int64)
                    source = (keys % np.uint64(n)).astype(index_dtype)
                    if np.any(target < start) or np.any(target >= stop):
                        raise RuntimeError("misbucketed edge")
                    counts = np.bincount(target-start, minlength=stop-start)
                    source.tofile(stream)
                    total_edges += len(source)
                    filename.unlink()
                else:
                    counts = np.zeros(stop-start, dtype=np.int64)
                ptr[start+1:stop+1] = total_edges - int(counts.sum()) + np.cumsum(counts)
        ptr.flush()
        del ptr
        if (output_dir / "indices.bin").stat().st_size != total_edges*np.dtype(index_dtype).itemsize:
            raise RuntimeError("CSR binary size mismatch")
        metadata = {
            "status": "complete", "num_nodes": n,
            "directed_edges": total_edges, "candidate_directed_edges": raw_candidates,
            "index_dtype": np.dtype(index_dtype).name,
            "bucket_nodes": bucket_nodes,
            "ordering": "incoming_sources_sorted_per_target",
            "self_loops": "removed",
            "topology": "cell_edge_connectivity_not_cartesian",
        }
        (output_dir / "csr.json").write_text(json.dumps(metadata, indent=2)+"\n")
        shutil.rmtree(buckets)
        return metadata
    except BaseException:
        # No 'complete' metadata marker: any partially written CSR is invalid.
        raise


def build_hdf5_disk_csr(
    mesh_file: Path, *,
    num_nodes: int,
    output_dir: Path,
    connectivity_path: str = "Connectivity/hex->node",
    element_type: str = "hex",
    one_based: bool = True,
    bucket_nodes: int = 131072,
    chunk_cells: int = 65536,
) -> dict:
    return build_disk_csr(
        iter_hdf5_directed_edges(
            mesh_file, connectivity_path=connectivity_path,
            element_type=element_type, one_based=one_based,
            num_nodes=num_nodes, chunk_cells=chunk_cells,
        ),
        num_nodes=num_nodes, output_dir=output_dir,
        bucket_nodes=bucket_nodes,
    )
