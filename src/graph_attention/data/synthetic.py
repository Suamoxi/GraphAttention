"""Deterministic non-physical variable-mesh data for infrastructure tests."""

from __future__ import annotations

from operator import index as operator_index

import torch
from torch.utils.data import Dataset

from .contracts import FieldCatalog, FieldRole, FieldSpec, FieldSupport, Mesh, Sample

_TOPOLOGIES = ("chain", "cycle", "star")
_COMPONENTS = {2: ("x", "y"), 3: ("x", "y", "z")}


class SyntheticMeshDataset(Dataset):
    """Generate deterministic variable node-based graphs with named fields.

    The generated values are deliberately non-physical. This dataset exists to
    exercise data contracts, configuration, and later batching code before real
    CFD readers are introduced.
    """

    def __init__(
        self,
        num_samples: int = 9,
        min_nodes: int = 4,
        max_nodes: int = 12,
        spatial_dim: int = 2,
        seed: int = 42,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if min_nodes < 3:
            raise ValueError("min_nodes must be at least 3")
        if max_nodes < min_nodes:
            raise ValueError("max_nodes must be greater than or equal to min_nodes")
        if spatial_dim not in _COMPONENTS:
            raise ValueError("spatial_dim must be 2 or 3")
        if seed < 0:
            raise ValueError("seed must be non-negative")

        self.num_samples = num_samples
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.spatial_dim = spatial_dim
        self.seed = seed
        self.field_catalog = FieldCatalog(
            (
                FieldSpec(
                    name="rho",
                    support=FieldSupport.NODE,
                    role=FieldRole.PRIMARY_STATE,
                    provenance="SyntheticMeshDataset non-physical test field",
                    stored=False,
                ),
                FieldSpec(
                    name="momentum",
                    support=FieldSupport.NODE,
                    role=FieldRole.PRIMARY_STATE,
                    components=_COMPONENTS[spatial_dim],
                    provenance="SyntheticMeshDataset non-physical test field",
                    stored=False,
                ),
            )
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Sample:
        sample_index = operator_index(index)
        if sample_index < 0:
            sample_index += self.num_samples
        if sample_index < 0 or sample_index >= self.num_samples:
            raise IndexError(sample_index)

        node_span = self.max_nodes - self.min_nodes + 1
        num_nodes = self.min_nodes + sample_index % node_span
        topology = _TOPOLOGIES[sample_index % len(_TOPOLOGIES)]

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + sample_index)
        coords = torch.rand((num_nodes, self.spatial_dim), generator=generator)
        edge_index = _make_edge_index(topology, num_nodes)
        node_weights = torch.full((num_nodes,), 1.0 / num_nodes, dtype=coords.dtype)

        rho = 1.0 + coords.square().sum(dim=1)
        momentum = rho.unsqueeze(1) * (coords - 0.5)

        mesh = Mesh(
            coords=coords,
            edge_index=edge_index,
            mesh_id=f"synthetic-mesh-{sample_index:06d}",
            node_weights=node_weights,
            metadata={"synthetic": True, "topology": topology},
        )
        return Sample(
            sample_id=f"synthetic/{sample_index:06d}",
            mesh=mesh,
            fields={"rho": rho, "momentum": momentum},
            metadata={"synthetic": True, "topology": topology},
        )


def _make_edge_index(topology: str, num_nodes: int) -> torch.Tensor:
    if topology == "chain":
        source = torch.arange(num_nodes - 1, dtype=torch.long)
        target = source + 1
    elif topology == "cycle":
        source = torch.arange(num_nodes, dtype=torch.long)
        target = (source + 1) % num_nodes
    elif topology == "star":
        source = torch.zeros(num_nodes - 1, dtype=torch.long)
        target = torch.arange(1, num_nodes, dtype=torch.long)
    else:
        raise ValueError(f"unsupported synthetic topology '{topology}'")

    return torch.stack(
        (
            torch.cat((source, target)),
            torch.cat((target, source)),
        )
    )
