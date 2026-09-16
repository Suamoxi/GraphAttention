"""Dataset-agnostic packing and loader helpers for graph training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from graph_attention.data import PackedBatch, Sample
from graph_attention.geometry import (
    build_attention_edge_indices,
    cartesian_4_neighbor_edge_index,
    hex_connectivity_to_edge_index,
)
from graph_attention.tasks import NodeRegressionBatch, NodeRegressionTask


class GraphTaskCollator:
    """Build graph topology from each sample and pack a task-facing batch.

    Dataset readers own physical samples and native mesh information. This
    collator only resolves graph connectivity when a reader intentionally leaves
    ``mesh.edge_index`` empty, then asks the configured task to pack/prepare the
    samples. Additional attention topologies are built per physical graph, so
    variable node counts and different meshes are supported without a shared-grid
    assumption.
    """

    def __init__(
        self,
        task: NodeRegressionTask,
        catalog: Any,
        geometry_cfg: Mapping[str, Any] | Any,
    ) -> None:
        self.task = task
        self.catalog = catalog
        self.attention_specifications = dict(
            geometry_cfg.get("attention_edge_indices", {})
        )

    def __call__(self, samples: list[Sample]) -> NodeRegressionBatch:
        prepared_samples: list[Sample] = []
        per_sample_attention: list[dict[str, torch.Tensor]] = []

        for sample in samples:
            prepared = _sample_with_graph_connectivity(sample)
            prepared_samples.append(prepared)
            per_sample_attention.append(
                build_attention_edge_indices(
                    prepared.mesh.edge_index,
                    prepared.mesh.num_nodes,
                    self.attention_specifications,
                )
            )

        batch = self.task.pack_and_prepare(prepared_samples, self.catalog)
        if not self.attention_specifications:
            return batch

        packed_attention: dict[str, torch.Tensor] = {}
        for name in self.attention_specifications:
            parts = [
                topologies[name] + int(offset)
                for topologies, offset in zip(
                    per_sample_attention,
                    batch.ptr[:-1],
                    strict=True,
                )
            ]
            packed_attention[name] = (
                torch.cat(parts, dim=1)
                if parts
                else torch.empty((2, 0), dtype=torch.long)
            )
        return replace(batch, attention_edge_indices=packed_attention)


def make_loader(
    dataset: Dataset[Sample],
    indices: Sequence[int],
    *,
    batch_size: int,
    num_workers: int,
    collator: GraphTaskCollator,
    shuffle: bool,
    seed: int,
) -> DataLoader[NodeRegressionBatch]:
    """Create the same deterministic DataLoader policy used by prior runners."""

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        generator=generator,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def dataset_sample_ids(dataset: Dataset[Sample]) -> tuple[str, ...]:
    """Return explicit sample IDs without requiring a dataset-specific attribute."""

    raw = getattr(dataset, "sample_ids", None)
    if raw is not None:
        ids = tuple(raw)
    else:
        ids = tuple(dataset[index].sample_id for index in range(len(dataset)))
    if len(ids) != len(dataset):
        raise ValueError("dataset sample ID count does not match dataset length")
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("dataset sample IDs must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("dataset sample IDs must be unique")
    return ids


def dataset_group_ids(
    dataset: Dataset[Sample],
    sample_ids: Sequence[str],
    group_metadata_key: str | None,
) -> tuple[str, ...]:
    """Resolve split groups from dataset semantics rather than dataset type.

    A reader may expose ``group_id(index, key)`` for efficient metadata access.
    Otherwise the canonical ``Sample`` metadata/case/mesh fields are used. When
    no grouping key is configured, every sample forms its own group.
    """

    if group_metadata_key is None:
        return tuple(sample_ids)
    if not isinstance(group_metadata_key, str) or not group_metadata_key.strip():
        raise ValueError("group_metadata_key must be null or a non-empty string")

    group_fn = getattr(dataset, "group_id", None)
    if callable(group_fn):
        groups = tuple(group_fn(index, group_metadata_key) for index in range(len(dataset)))
    else:
        values: list[str] = []
        for index in range(len(dataset)):
            sample = dataset[index]
            if group_metadata_key == "case_id":
                value = sample.case_id
            elif group_metadata_key == "mesh_id":
                value = sample.mesh.mesh_id
            else:
                value = sample.metadata.get(group_metadata_key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"sample '{sample.sample_id}' does not define non-empty group key "
                    f"'{group_metadata_key}'"
                )
            values.append(value)
        groups = tuple(values)

    if len(groups) != len(sample_ids):
        raise ValueError("dataset group ID count does not match sample ID count")
    if any(not isinstance(value, str) or not value for value in groups):
        raise ValueError("dataset group IDs must be non-empty strings")
    return groups


def task_batch_to_device(
    batch: NodeRegressionBatch,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> NodeRegressionBatch:
    """Move model/task tensors while preserving immutable sample provenance."""

    source = _packed_structure_to_device(batch.source, device=device, dtype=dtype)
    attention_edge_indices = {
        name: edge_index.to(device=device)
        for name, edge_index in batch.attention_edge_indices.items()
    }
    return replace(
        batch,
        source=source,
        coords=batch.coords.to(device=device, dtype=dtype),
        inputs=batch.inputs.to(device=device, dtype=dtype),
        targets=batch.targets.to(device=device, dtype=dtype),
        conditioning=batch.conditioning.to(device=device, dtype=dtype),
        attention_edge_indices=attention_edge_indices,
    )


def _sample_with_graph_connectivity(sample: Sample) -> Sample:
    mesh = sample.mesh
    if mesh.edge_index.shape[1] > 0:
        return sample

    raw_grid_shape = mesh.metadata.get("grid_shape_2d")
    if raw_grid_shape is not None:
        edge_index = cartesian_4_neighbor_edge_index(raw_grid_shape)
    elif mesh.cell_connectivity is not None:
        if mesh.cell_connectivity.ndim != 2 or mesh.cell_connectivity.shape[1] != 8:
            raise ValueError(
                f"sample '{sample.sample_id}' has unsupported cell connectivity shape "
                f"{tuple(mesh.cell_connectivity.shape)}"
            )
        edge_index = hex_connectivity_to_edge_index(
            mesh.cell_connectivity,
            mesh.num_nodes,
        )
    else:
        raise ValueError(
            f"sample '{sample.sample_id}' has no graph edges and no supported mesh "
            "connectivity from which to construct them"
        )

    return replace(sample, mesh=replace(mesh, edge_index=edge_index))


def _packed_structure_to_device(
    packed: PackedBatch,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> PackedBatch:
    node_weights = packed.node_weights
    if node_weights is not None:
        node_weights = node_weights.to(device=device, dtype=dtype)
    return replace(
        packed,
        edge_index=packed.edge_index.to(device=device),
        batch_index=packed.batch_index.to(device=device),
        ptr=packed.ptr.to(device=device),
        node_weights=node_weights,
    )
