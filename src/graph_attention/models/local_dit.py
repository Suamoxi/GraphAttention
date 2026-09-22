"""Local-attention DiT model for packed CFD graphs."""

from __future__ import annotations

import torch

from .dit_common import _BaseDiTGraphTransformer
from .sparse_transformer import SparseMultiheadAttention


class LocalDiTMultiheadAttention(SparseMultiheadAttention):
    """One-hop mesh self-attention with DiT-matched qkv/out projections."""

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor | None,
    ) -> torch.Tensor:
        del batch_index
        return super().forward(inputs, edge_index)


class LocalDiTGraphTransformer(_BaseDiTGraphTransformer):
    """DiT whose attention is restricted to one-hop mesh neighbours plus self."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        spatial_dim: int,
        mlp_ratio: int = 4,
        conditioning_channels: int = 0,
        condition_embed_dim: int = 128,
        use_coord_mlp: bool = True,
        coordinate_normalization: str = "centered_bbox",
        coordinate_normalization_eps: float = 1.0e-8,
        use_sdpa: bool = True,
    ) -> None:
        # Kept in the public signature so Full/Local configs can remain identical.
        del use_sdpa
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            spatial_dim=spatial_dim,
            attention_factory=lambda dim, heads: LocalDiTMultiheadAttention(dim, heads),
            uses_local_edges=True,
            mlp_ratio=mlp_ratio,
            conditioning_channels=conditioning_channels,
            condition_embed_dim=condition_embed_dim,
            use_coord_mlp=use_coord_mlp,
            coordinate_normalization=coordinate_normalization,
            coordinate_normalization_eps=coordinate_normalization_eps,
        )
