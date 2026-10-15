"""M41.6 selective and memory-mapped *static* partition metadata providers.

Both providers preserve M41.4's exact owned-query incoming graph, adjacency
ordering, relative displacements, and synchronous layerwise hidden updates.
Only the topology matching the current layer is loaded. Initial/final
projections require owned IDs only. The optional compiled mmap cache stores
CSR arrays and displacement on disk, avoiding repeated decompression and
CPU geometric reconstruction without keeping all partitions in RAM.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from collections.abc import Iterator, Sequence

import numpy as np
import torch

from .disk_streamed_dinat import _attention
from .owned_query_dinat import OwnedQueryTopology
from .partitioned_dinat import PartitionAttention


def _detail(profiler, label: str):
    return profiler.metadata_detail(label) if profiler is not None else nullcontext()


@dataclass(frozen=True)
class StageOwnedNodes:
    owned_global: torch.Tensor
    read_global: torch.Tensor | None = None
    owned_local: torch.Tensor | None = None
    local: PartitionAttention | None = None
    dilated: PartitionAttention | None = None


@dataclass(frozen=True)
class StagePartition:
    nodes: StageOwnedNodes
    local: OwnedQueryTopology | None = None
    dilated: OwnedQueryTopology | None = None


def _valid_kind(kind: str) -> str:
    if kind not in ("owned", "local", "dilated"):
        raise ValueError("pass kind must be owned, local or dilated")
    return kind


def _stage_partition(
    *, owned: torch.Tensor, read: torch.Tensor | None = None,
    owned_local: torch.Tensor | None = None,
    attention: PartitionAttention | None = None,
    topology: OwnedQueryTopology | None = None,
    kind: str,
) -> StagePartition:
    if kind == "owned":
        return StagePartition(nodes=StageOwnedNodes(owned_global=owned))
    if read is None or owned_local is None or attention is None or topology is None:
        raise ValueError("missing active stage attention inputs")
    return StagePartition(
        nodes=StageOwnedNodes(
            owned_global=owned, read_global=read,
            owned_local=owned_local,
            local=attention if kind == "local" else None,
            dilated=attention if kind == "dilated" else None,
        ),
        local=topology if kind == "local" else None,
        dilated=topology if kind == "dilated" else None,
    )


@dataclass
class SelectiveDiskPartitionCollection(Sequence[StagePartition]):
    """Read only the NPZ members required for this forward stage."""

    directory: Path
    raw_coords: torch.Tensor
    count: int

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        # Sequence compatibility is an active-local stage; the streamed
        # executor always uses iter_for_pass() to select the correct stage.
        return self.load_for_pass(index, "local")

    @torch.inference_mode(False)
    def load_for_pass(self, index: int, kind: str, stage_profiler=None) -> StagePartition:
        _valid_kind(kind)
        if index < 0:
            index += self.count
        if index < 0 or index >= self.count:
            raise IndexError(index)
        with _detail(stage_profiler, "metadata_npz_decode"):
            with np.load(
                self.directory / f"partition_{index:05d}.npz", allow_pickle=False
            ) as data:
                raw_owned = data["owned"]
                if kind != "owned":
                    raw_read = data["read"]
                    raw_source = data[f"{kind}_source"]
                    raw_target = data[f"{kind}_target"]
        with _detail(stage_profiler, "metadata_index_conversion"):
            owned = torch.from_numpy(raw_owned.astype(np.int64, copy=True))
            if kind == "owned":
                return _stage_partition(owned=owned, kind=kind)
            read = torch.from_numpy(raw_read.astype(np.int64, copy=True))
            source = torch.from_numpy(raw_source.astype(np.int64, copy=True))
            target = torch.from_numpy(raw_target.astype(np.int64, copy=True))
            owned_local = torch.searchsorted(read, owned)
            if owned.numel() and not torch.equal(read[owned_local], owned):
                raise ValueError("owned nodes are missing from read_global")
        attention, topology = _attention(
            source, target, owned_local=owned_local,
            read_count=len(read), raw_coords=self.raw_coords, read_global=read,
            compact=True, stage_profiler=stage_profiler,
        )
        return _stage_partition(
            owned=owned, read=read, owned_local=owned_local,
            attention=attention, topology=topology, kind=kind,
        )

    def iter_for_pass(self, kind: str, *, stage_profiler=None) -> Iterator[StagePartition]:
        _valid_kind(kind)
        for index in range(self.count):
            yield self.load_for_pass(index, kind, stage_profiler)


def source_manifest_sha256(partition_directory: Path) -> str:
    return hashlib.sha256(
        (Path(partition_directory) / "partitions.json").read_bytes()
    ).hexdigest()


@dataclass
class MappedDiskPartitionCollection(Sequence[StagePartition]):
    """Lazily memory-map precomputed partition CSR/relative displacements."""

    directory: Path
    count: int

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        return self.load_for_pass(index, "local")

    @torch.inference_mode(False)
    def load_for_pass(self, index: int, kind: str, stage_profiler=None) -> StagePartition:
        _valid_kind(kind)
        if index < 0:
            index += self.count
        if index < 0 or index >= self.count:
            raise IndexError(index)
        prefix = self.directory / f"part_{index:05d}"

        def mapped(name: str) -> torch.Tensor:
            with _detail(stage_profiler, "metadata_mmap_array_open"):
                # mmap_mode='c' is writeable copy-on-write, so torch.from_numpy
                # creates a normal, versioned tensor without cloning a whole
                # graph's data or modifying the immutable source files.
                arr = np.load(prefix / f"{name}.npy", mmap_mode="c", allow_pickle=False)
                return torch.from_numpy(arr)

        owned = mapped("owned")
        if kind == "owned":
            return _stage_partition(owned=owned, kind=kind)
        read = mapped("read")
        owned_local = mapped("owned_local")
        col = mapped(f"{kind}_col")
        row = mapped(f"{kind}_row")
        crow = mapped(f"{kind}_crow")
        displacement = mapped(f"{kind}_displacement")

        with _detail(stage_profiler, "metadata_mmap_csr_wrapper"):
            if (owned.dtype != torch.int64 or read.dtype != torch.int64
                or owned_local.dtype != torch.int64 or col.dtype != torch.int64
                or row.dtype != torch.int64 or crow.dtype != torch.int64
                or displacement.dtype != torch.float32):
                raise TypeError("mmap cache format has unexpected dtypes")
            if crow.shape != (len(owned) + 1,) or displacement.shape != (len(col), 3):
                raise ValueError("mmap cache CSR/geometry dimensions differ")
            if int(crow[-1]) != len(col) or len(row) != len(col):
                raise ValueError("mmap cache CSR offsets do not match edge count")
            pattern = torch.sparse_csr_tensor(
                crow, col, torch.ones(len(col), dtype=torch.float32),
                size=(len(owned), len(read)), dtype=torch.float32, device="cpu",
            )
        attention = PartitionAttention(
            edge_index=torch.empty((2, 0), dtype=torch.long),
            displacement=displacement, sparse_topology=None,
        )
        topology = OwnedQueryTopology(
            pattern=pattern, row_indices=row,
            col_indices=col, edge_order=None,
        )
        return _stage_partition(
            owned=owned, read=read, owned_local=owned_local,
            attention=attention, topology=topology, kind=kind,
        )

    def iter_for_pass(self, kind: str, *, stage_profiler=None) -> Iterator[StagePartition]:
        _valid_kind(kind)
        for index in range(self.count):
            yield self.load_for_pass(index, kind, stage_profiler)


def validate_mmap_cache(
    directory: Path, *,
    partition_dir: Path,
    expected_nodes: int, expected_partitions: int,
) -> dict:
    directory = Path(directory).resolve(strict=True)
    meta = json.loads((directory / "cache.json").read_text())
    if meta.get("status") != "complete" or meta.get("format_version") != 1:
        raise ValueError("incomplete or unrecognized M41.6 memory map cache")
    if (meta.get("num_nodes") != expected_nodes
            or meta.get("num_partitions") != expected_partitions):
        raise ValueError("mmap cache does not match graph node or partition count")
    if meta.get("source_partition_sha256") != source_manifest_sha256(partition_dir):
        raise ValueError("mmap cache is stale relative to original partition manifest")
    if Path(meta["source_partition_directory"]).resolve(strict=True) != Path(partition_dir).resolve(strict=True):
        raise ValueError("mmap cache was compiled from a different partition directory")
    return meta
