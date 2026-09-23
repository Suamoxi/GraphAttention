"""DiT-conditioned DiNAT-style alternating geometric sparse attention."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from graph_attention.geometry import edge_relative_displacement

from .dit_common import (
    _BaseDiTGraphTransformer,
    _normalize_coordinates_by_graph,
    _validate_packed_batch_index,
)
from .geometric_transformer import (
    GeometricSparseMultiheadAttention,
    _validate_model_coords,
)
from .sparse_transformer import (
    _build_dgl_sparse_adjacency,
    _validate_edge_index,
    _validate_model_inputs,
)

_DILATED_TOPOLOGY_NAME = "dilated"


class DiNATDiTMultiheadAttention(GeometricSparseMultiheadAttention):
    """M12 geometric sparse attention with the DiT block call signature."""

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor | None,
        edge_displacement: torch.Tensor,
        sparse_adj: object | None = None,
    ) -> torch.Tensor:
        del batch_index
        return super().forward(
            inputs,
            edge_index,
            edge_displacement,
            sparse_adj=sparse_adj,
        )


class AlternatingDilatedGeometricDiT(_BaseDiTGraphTransformer):
    """DiT backbone with M12 local/exact-two-hop geometric attention.

    Even-numbered layers use the supplied local edge_index. Odd-numbered layers
    use attention_edge_indices["dilated"]. Both attention schemes reuse the
    same learned relative-displacement score bias as M12.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        spatial_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        conditioning_channels: int = 0,
        condition_embed_dim: int = 128,
        use_coord_mlp: bool = True,
        coordinate_normalization: str = "centered_bbox",
        coordinate_normalization_eps: float = 1.0e-8,
        use_sdpa: bool = True,
        qkv_bias: bool = True,
        out_proj_bias: bool = False,
        sparse_attention_backend: str = "scatter",
    ) -> None:
        # Sparse geometric attention does not use dense SDPA, but this argument
        # keeps the same public configuration contract as the other DiT models.
        del use_sdpa
        backend = str(sparse_attention_backend)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            spatial_dim=spatial_dim,
            attention_factory=lambda dim, heads: DiNATDiTMultiheadAttention(
                hidden_dim=dim,
                num_heads=heads,
                spatial_dim=spatial_dim,
                dropout=dropout,
                qkv_bias=qkv_bias,
                out_proj_bias=out_proj_bias,
                sparse_attention_backend=backend,
            ),
            uses_local_edges=False,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            conditioning_channels=conditioning_channels,
            condition_embed_dim=condition_embed_dim,
            use_coord_mlp=use_coord_mlp,
            coordinate_normalization=coordinate_normalization,
            coordinate_normalization_eps=coordinate_normalization_eps,
        )
        self.sparse_attention_backend = backend

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        coords: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        conditioning: torch.Tensor | None = None,
        attention_edge_indices: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        _validate_model_inputs(
            inputs,
            expected_channels=self.in_channels,
            parameter=self.input_projection.weight,
        )
        _validate_model_coords(
            coords,
            num_nodes=inputs.shape[0],
            spatial_dim=self.spatial_dim,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        if batch_index is not None:
            _validate_packed_batch_index(
                batch_index,
                num_nodes=inputs.shape[0],
                device=inputs.device,
            )

        _validate_edge_index(
            edge_index,
            num_nodes=inputs.shape[0],
            device=inputs.device,
        )
        if (
            attention_edge_indices is None
            or _DILATED_TOPOLOGY_NAME not in attention_edge_indices
        ):
            raise ValueError(
                "DiNAT-DiT requires attention_edge_indices['dilated']"
            )
        dilated_edge_index = attention_edge_indices[_DILATED_TOPOLOGY_NAME]
        _validate_edge_index(
            dilated_edge_index,
            num_nodes=inputs.shape[0],
            device=inputs.device,
        )

        model_coords = coords
        if self.coordinate_normalization == "centered_bbox":
            model_coords = _normalize_coordinates_by_graph(
                coords,
                batch_index=batch_index,
                eps=self.coordinate_normalization_eps,
            )

        hidden = self.input_projection(inputs) + self.coordinate_projection(model_coords)
        condition_embedding = self._condition_embedding(
            inputs,
            batch_index=batch_index,
            conditioning=conditioning,
        )

        local_displacement = edge_relative_displacement(coords, edge_index)
        dilated_displacement = edge_relative_displacement(
            coords,
            dilated_edge_index,
        )

        local_sparse_adj = None
        dilated_sparse_adj = None
        if self.sparse_attention_backend == "dgl":
            local_sparse_adj = _build_dgl_sparse_adjacency(
                edge_index,
                num_nodes=inputs.shape[0],
            )
            dilated_sparse_adj = _build_dgl_sparse_adjacency(
                dilated_edge_index,
                num_nodes=inputs.shape[0],
            )

        for layer_index, block in enumerate(self.blocks):
            if layer_index % 2 == 0:
                layer_edge_index = edge_index
                layer_displacement = local_displacement
                layer_sparse_adj = local_sparse_adj
            else:
                layer_edge_index = dilated_edge_index
                layer_displacement = dilated_displacement
                layer_sparse_adj = dilated_sparse_adj

            attention_kwargs: dict[str, object] = {
                "edge_displacement": layer_displacement,
            }
            if layer_sparse_adj is not None:
                attention_kwargs["sparse_adj"] = layer_sparse_adj

            hidden = block(
                hidden,
                condition_embedding,
                edge_index=layer_edge_index,
                batch_index=batch_index,
                attention_kwargs=attention_kwargs,
            )

        return self.final_layer(
            hidden,
            condition_embedding,
            batch_index=batch_index,
        )
