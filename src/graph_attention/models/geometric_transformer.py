"""M9 sparse transformer with learned relative-displacement attention bias."""

from __future__ import annotations

import torch
from torch import nn

from graph_attention.geometry import edge_relative_displacement

from .sparse_transformer import (
    SparseMultiheadAttention,
    _append_conditioning,
    _nonnegative_count,
    _positive_count,
    _validate_edge_index,
    _validate_hidden_inputs,
    _validate_model_inputs,
)


class GeometricSparseMultiheadAttention(SparseMultiheadAttention):
    """Sparse attention with a learned per-head bias from relative displacement."""

    def __init__(self, hidden_dim: int, num_heads: int, spatial_dim: int) -> None:
        super().__init__(hidden_dim=hidden_dim, num_heads=num_heads)
        self.spatial_dim = _positive_count(spatial_dim, "spatial_dim")
        self.geometry_mlp = nn.Sequential(
            nn.Linear(self.spatial_dim, self.num_heads),
            nn.GELU(),
            nn.Linear(self.num_heads, self.num_heads, bias=False),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_displacement: torch.Tensor,
    ) -> torch.Tensor:
        _validate_hidden_inputs(
            inputs,
            expected_channels=self.hidden_dim,
            parameter=self.qkv.weight,
        )
        _validate_edge_index(edge_index, num_nodes=inputs.shape[0], device=inputs.device)
        _validate_edge_displacement(
            edge_displacement,
            num_edges=edge_index.shape[1],
            spatial_dim=self.spatial_dim,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        return self._forward_with_geometry_validated(inputs, edge_index, edge_displacement)

    def _forward_with_geometry_validated(
        self,
        inputs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_displacement: torch.Tensor,
    ) -> torch.Tensor:
        score_bias = self.geometry_mlp(edge_displacement)
        return self._forward_validated(inputs, edge_index, score_bias=score_bias)


class GeometricSparseGraphTransformerBlock(nn.Module):
    """Pre-norm sparse Transformer block with M9 geometric attention bias."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        spatial_dim: int,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        hidden = _positive_count(hidden_dim, "hidden_dim")
        ratio = _positive_count(mlp_ratio, "mlp_ratio")
        self.norm1 = nn.LayerNorm(hidden)
        self.attention = GeometricSparseMultiheadAttention(
            hidden_dim=hidden,
            num_heads=num_heads,
            spatial_dim=spatial_dim,
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * ratio),
            nn.GELU(),
            nn.Linear(hidden * ratio, hidden),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_displacement: torch.Tensor,
    ) -> torch.Tensor:
        hidden = inputs + self.attention._forward_with_geometry_validated(
            self.norm1(inputs),
            edge_index,
            edge_displacement,
        )
        return hidden + self.mlp(self.norm2(hidden))


class GeometricSparseGraphTransformer(nn.Module):
    """M9 node transformer using topology plus relative edge displacement."""

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
    ) -> None:
        super().__init__()
        self.in_channels = _positive_count(in_channels, "in_channels")
        self.out_channels = _positive_count(out_channels, "out_channels")
        self.hidden_dim = _positive_count(hidden_dim, "hidden_dim")
        self.num_heads = _positive_count(num_heads, "num_heads")
        self.num_layers = _positive_count(num_layers, "num_layers")
        self.spatial_dim = _positive_count(spatial_dim, "spatial_dim")
        self.mlp_ratio = _positive_count(mlp_ratio, "mlp_ratio")
        self.conditioning_channels = _nonnegative_count(
            conditioning_channels,
            "conditioning_channels",
        )
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.input_projection = nn.Linear(
            self.in_channels + self.conditioning_channels,
            self.hidden_dim,
        )
        self.blocks = nn.ModuleList(
            GeometricSparseGraphTransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                spatial_dim=self.spatial_dim,
                mlp_ratio=self.mlp_ratio,
            )
            for _ in range(self.num_layers)
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.out_channels)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        coords: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        conditioning: torch.Tensor | None = None,
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
        edge_displacement = edge_relative_displacement(coords, edge_index)

        features = _append_conditioning(
            inputs,
            batch_index=batch_index,
            conditioning=conditioning,
            expected_channels=self.conditioning_channels,
        )
        hidden = self.input_projection(features)
        for block in self.blocks:
            hidden = block(hidden, edge_index, edge_displacement)
        return self.output_projection(self.final_norm(hidden))


def _validate_model_coords(
    coords: torch.Tensor,
    *,
    num_nodes: int,
    spatial_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if coords.ndim != 2 or coords.shape != (num_nodes, spatial_dim):
        raise ValueError(f"coords must have shape [{num_nodes}, {spatial_dim}]")
    if not coords.is_floating_point():
        raise TypeError("coords must use a floating-point dtype")
    if coords.dtype != dtype:
        raise TypeError("coords and inputs must share one dtype")
    if coords.device != device:
        raise ValueError("coords and inputs must be on the same device")


def _validate_edge_displacement(
    edge_displacement: torch.Tensor,
    *,
    num_edges: int,
    spatial_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if edge_displacement.ndim != 2 or edge_displacement.shape != (num_edges, spatial_dim):
        raise ValueError(f"edge_displacement must have shape [{num_edges}, {spatial_dim}]")
    if not edge_displacement.is_floating_point():
        raise TypeError("edge_displacement must use a floating-point dtype")
    if edge_displacement.dtype != dtype:
        raise TypeError("edge_displacement and inputs must share one dtype")
    if edge_displacement.device != device:
        raise ValueError("edge_displacement and inputs must be on the same device")
    if not torch.isfinite(edge_displacement).all():
        raise ValueError("edge_displacement contains NaN or Inf")
