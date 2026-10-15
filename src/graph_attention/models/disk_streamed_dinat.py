"""M41.4: disk-backed generic topology provider for unchanged M41.3 executor.

A one-hop CSR is built from arbitrary physical mesh element connectivity.
Only owned-query one-hop / exact-two-hop edges are materialized per partition.
The model and forward_streamed_dinat are unchanged; partition plans are
loaded lazily from disk or preloaded into host RAM when available.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
from collections.abc import Iterator, Sequence

import numpy as np
import torch

from .dinat_dit import AlternatingDilatedGeometricDiT
from .dit_common import _normalize_coordinates_by_graph
from .owned_query_dinat import OwnedQueryPlan, OwnedQueryPartition, OwnedQueryTopology
from .partitioned_dinat import DiNATPartitionPlan, OwnedHaloPartition, PartitionAttention
from .streamed_dinat import StreamedDiNATPlan


def _metadata_substage(profiler, name: str):
    return profiler.metadata_detail(name) if profiler is not None else nullcontext()


def _attention(
    source: torch.Tensor, target: torch.Tensor,
    *, owned_local: torch.Tensor, read_count: int,
    raw_coords: torch.Tensor, read_global: torch.Tensor,
    compact: bool = False,
    stage_profiler=None,
) -> tuple[PartitionAttention, OwnedQueryTopology]:
    """Build exact target-major rectangular CSR and aligned displacement.

    Existing M41.4 construction uses the original full edge_index and
    identity edge_order. The selective path elides those redundant
    arrays, but only for partition files already target/source sorted.
    """
    with _metadata_substage(stage_profiler, "metadata_csr_assembly"):
        row = torch.searchsorted(owned_local, target)
        if row.numel() and (
            bool((row >= owned_local.numel()).any())
            or not torch.equal(owned_local[row], target)
        ):
            raise ValueError("an attention target is not owned")
        if row.numel() > 1 and bool((row[1:] < row[:-1]).any()):
            raise ValueError("attention edge targets must be sorted")
        if row.numel() > 1:
            keys = row * read_count + source
            if bool((keys[1:] <= keys[:-1]).any()):
                raise ValueError("attention CSR edges must be strictly target/source sorted")
        counts = torch.bincount(row, minlength=int(owned_local.numel()))
        crow = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        values = torch.ones(len(source), dtype=torch.float32)
        with torch.inference_mode(False):
            pattern = torch.sparse_csr_tensor(
                crow, source, values,
                size=(int(owned_local.numel()), read_count),
                dtype=torch.float32, device="cpu",
            )
        if compact:
            # _owned_context reads CSR columns, not the physical edge_index.
            # All canonical partition files have identity edge ordering.
            edges = torch.empty((2, 0), dtype=torch.long)
            edge_order = None
        else:
            edges = torch.stack((source, target), dim=0)
            edge_order = torch.arange(len(source), dtype=torch.long)
    with _metadata_substage(stage_profiler, "metadata_relative_displacement"):
        raw_source = raw_coords.index_select(0, read_global[source])
        raw_target = raw_coords.index_select(0, read_global[target])
        displacement = raw_source - raw_target
    attn = PartitionAttention(
        edge_index=edges, displacement=displacement, sparse_topology=None,
    )
    topology = OwnedQueryTopology(
        pattern=pattern, row_indices=row, col_indices=source,
        edge_order=edge_order,
    )
    return attn, topology

@dataclass
class DiskPartitionCollection(Sequence[OwnedQueryPartition]):
    directory: Path
    raw_coords: torch.Tensor
    count: int

    def __len__(self) -> int:
        return self.count

    @torch.inference_mode(False)
    def __getitem__(self, index: int | slice):
        """Load normal (versioned) CPU CSR metadata, including both families."""
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(self.count)))
        return self._load_partition(index, None)

    @torch.inference_mode(False)
    def _load_partition(self, index: int, stage_profiler=None) -> OwnedQueryPartition:
        if index < 0:
            index += self.count
        if index < 0 or index >= self.count:
            raise IndexError(index)
        # This is the original full partition materialization, with optional
        # CPU-only substage timers. No files or numerical semantics change.
        with _metadata_substage(stage_profiler, "metadata_npz_decode"):
            with np.load(
                self.directory / f"partition_{index:05d}.npz", allow_pickle=False
            ) as data:
                raw_owned = data["owned"]
                raw_read = data["read"]
                raw_s1 = data["local_source"]
                raw_t1 = data["local_target"]
                raw_s2 = data["dilated_source"]
                raw_t2 = data["dilated_target"]
        with _metadata_substage(stage_profiler, "metadata_index_conversion"):
            owned = torch.from_numpy(raw_owned.astype(np.int64))
            read = torch.from_numpy(raw_read.astype(np.int64))
            owned_local = torch.searchsorted(read, owned)
            if owned_local.numel() and not torch.equal(read[owned_local], owned):
                raise ValueError("owned nodes absent from read set")
            s1 = torch.from_numpy(raw_s1.astype(np.int64))
            t1 = torch.from_numpy(raw_t1.astype(np.int64))
            s2 = torch.from_numpy(raw_s2.astype(np.int64))
            t2 = torch.from_numpy(raw_t2.astype(np.int64))
        local_attn, local_topo = _attention(
            s1, t1, owned_local=owned_local,
            read_count=len(read), raw_coords=self.raw_coords, read_global=read,
            stage_profiler=stage_profiler,
        )
        dilated_attn, dilated_topo = _attention(
            s2, t2, owned_local=owned_local,
            read_count=len(read), raw_coords=self.raw_coords, read_global=read,
            stage_profiler=stage_profiler,
        )
        nodes = OwnedHaloPartition(
            partition_id=index, owned_global=owned,
            read_global=read, owned_local=owned_local,
            local=local_attn, dilated=dilated_attn,
        )
        return OwnedQueryPartition(
            nodes=nodes, local=local_topo, dilated=dilated_topo,
        )

    def __iter__(self) -> Iterator[OwnedQueryPartition]:
        for i in range(self.count):
            yield self[i]

    def iter_for_pass(self, kind: str, *, stage_profiler=None) -> Iterator[OwnedQueryPartition]:
        """Legacy reference deliberately loads both topologies for every pass."""
        if kind not in ("owned", "local", "dilated"):
            raise ValueError("unexpected partition pass kind")
        for i in range(self.count):
            yield self._load_partition(i, stage_profiler)


@torch.inference_mode(False)
@torch.no_grad()
def build_disk_streamed_dinat_plan(
    model: AlternatingDilatedGeometricDiT,
    *,
    coords: torch.Tensor,
    partition_dir: Path,
    preload: bool = False,
    pin_memory: bool = False,
    metadata_mode: str = "baseline",
    metadata_cache_dir: Path | None = None,
) -> StreamedDiNATPlan:
    """Open generic CSR partitions without global 1/2-hop edge tensors.

    coords: [N,3] float32 coordinates in the ORIGINAL model-facing
    physical/nondimensional convention, before its centered_bbox transform.
    """
    if not isinstance(model, AlternatingDilatedGeometricDiT):
        raise TypeError("expected M37 AlternatingDilatedGeometricDiT")
    if model.sparse_attention_backend != "torch_sparse":
        raise ValueError("only torch_sparse backend is currently supported")
    if model.training:
        raise ValueError("inference-only: call model.eval()")
    if coords.ndim != 2 or coords.device.type != "cpu" or coords.dtype != torch.float32:
        raise TypeError("coords must be CPU float32 [N,spatial_dim]")
    if coords.shape[1] != model.spatial_dim or not bool(torch.isfinite(coords).all()):
        raise ValueError("coordinate dimensionality or finite-value check failed")
    if model.input_projection.weight.dtype != torch.float32:
        raise ValueError("requires float32 model parameters")
    if pin_memory and model.input_projection.weight.device.type != "cuda":
        raise ValueError("pin_memory requires CUDA model")
    if metadata_mode not in ("baseline", "selective", "mmap"):
        raise ValueError("metadata_mode must be baseline, selective, or mmap")
    if preload and metadata_mode != "baseline":
        raise ValueError("preload is only valid with the legacy baseline metadata loader")
    if metadata_mode == "mmap" and metadata_cache_dir is None:
        raise ValueError("mmap metadata mode requires a compiled --metadata-cache-dir")
    directory = Path(partition_dir).expanduser().resolve(strict=True)
    meta = json.loads((directory / "partitions.json").read_text())
    if meta["status"] != "complete" or meta["num_nodes"] != len(coords):
        raise ValueError("partition files do not match model input count")
    for attr in ("include_self_attention", "include_local_self_attention"):
        if bool(meta[attr]) != bool(getattr(model, attr)):
            raise ValueError(f"saved partition plan differs in {attr}")
    n_parts = int(meta["num_partitions"])
    if n_parts != len(meta["parts"]) or n_parts < 1:
        raise ValueError("incomplete partition manifest")
    model_coords = coords.detach().clone()
    if model.coordinate_normalization == "centered_bbox":
        model_coords = _normalize_coordinates_by_graph(
            coords, batch_index=None, eps=model.coordinate_normalization_eps,
        )
    elif model.coordinate_normalization != "none":
        raise ValueError("unsupported coordinate normalization")
    if metadata_mode == "baseline":
        partitions: Sequence[OwnedQueryPartition] = DiskPartitionCollection(
            directory, coords.detach().clone(), n_parts
        )
        if preload:
            # The original M41.4 baseline remains available unchanged.
            partitions = tuple(partitions)
    elif metadata_mode == "selective":
        from .selective_disk_dinat import SelectiveDiskPartitionCollection
        partitions = SelectiveDiskPartitionCollection(
            directory, coords.detach().clone(), n_parts,
        )
    else:
        from .selective_disk_dinat import (
            MappedDiskPartitionCollection, validate_mmap_cache,
        )
        cache_dir = Path(metadata_cache_dir).expanduser().resolve(strict=True)
        validate_mmap_cache(
            cache_dir, partition_dir=directory,
            expected_nodes=len(coords), expected_partitions=n_parts,
        )
        partitions = MappedDiskPartitionCollection(cache_dir, n_parts)
    base = DiNATPartitionPlan(
        num_nodes=len(coords), spatial_dim=coords.shape[1],
        device=torch.device("cpu"), model_coords=model_coords,
        backend="torch_sparse",
        include_self_attention=model.include_self_attention,
        include_local_self_attention=model.include_local_self_attention,
        coordinate_normalization=model.coordinate_normalization,
        coordinate_normalization_eps=model.coordinate_normalization_eps,
        partitions=(),
    )
    return StreamedDiNATPlan(
        cpu_plan=OwnedQueryPlan(base=base, partitions=partitions),
        model_device=model.input_projection.weight.device,
        pinned_host_buffers=pin_memory,
    )
