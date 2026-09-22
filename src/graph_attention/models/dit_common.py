"""Shared DiT building blocks for packed CFD graph models."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import nn

from .geometric_transformer import _validate_model_coords
from .sparse_transformer import (
    _positive_count,
    _validate_edge_index,
    _validate_model_inputs,
)


def _modulate(
    inputs: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return inputs * (1.0 + scale) + shift


class DiTBlock(nn.Module):
    """adaLN-Zero DiT block shared by full and local attention models."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        mlp_ratio: int,
        attention: nn.Module,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = _positive_count(hidden_dim, "hidden_dim")
        ratio = _positive_count(mlp_ratio, "mlp_ratio")
        dropout_probability = float(dropout)
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1.0e-6)
        self.attention = attention
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1.0e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * ratio),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout_probability),
            nn.Linear(hidden * ratio, hidden),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, 6 * hidden),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        condition_embedding: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor | None,
    ) -> torch.Tensor:
        graph_modulation = self.adaLN_modulation(condition_embedding)
        node_modulation = _broadcast_graph_features(
            graph_modulation,
            batch_index=batch_index,
            num_nodes=inputs.shape[0],
        )
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = node_modulation.chunk(6, dim=-1)

        attention_input = _modulate(self.norm1(inputs), shift_attn, scale_attn)
        hidden = inputs + gate_attn * self.attention(
            attention_input,
            edge_index=edge_index,
            batch_index=batch_index,
        )

        mlp_input = _modulate(self.norm2(hidden), shift_mlp, scale_mlp)
        return hidden + gate_mlp * self.mlp(mlp_input)


class DiTFinalLayer(nn.Module):
    """Conditioned output head shared exactly by full and local DiT."""

    def __init__(self, hidden_dim: int, out_channels: int) -> None:
        super().__init__()
        hidden = _positive_count(hidden_dim, "hidden_dim")
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1.0e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, 2 * hidden),
        )
        self.projection = nn.Linear(hidden, _positive_count(out_channels, "out_channels"))

    def forward(
        self,
        inputs: torch.Tensor,
        condition_embedding: torch.Tensor,
        *,
        batch_index: torch.Tensor | None,
    ) -> torch.Tensor:
        graph_modulation = self.adaLN_modulation(condition_embedding)
        node_modulation = _broadcast_graph_features(
            graph_modulation,
            batch_index=batch_index,
            num_nodes=inputs.shape[0],
        )
        shift, scale = node_modulation.chunk(2, dim=-1)
        hidden = _modulate(self.norm(inputs), shift, scale)
        return self.projection(hidden)


class _BaseDiTGraphTransformer(nn.Module):
    """Shared DiT model body; concrete modules provide only attention connectivity."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        spatial_dim: int,
        *,
        attention_factory: Callable[[int, int], nn.Module],
        uses_local_edges: bool,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        conditioning_channels: int = 0,
        condition_embed_dim: int = 128,
        use_coord_mlp: bool = True,
        coordinate_normalization: str = "centered_bbox",
        coordinate_normalization_eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_count(in_channels, "in_channels")
        self.out_channels = _positive_count(out_channels, "out_channels")
        self.hidden_dim = _positive_count(hidden_dim, "hidden_dim")
        self.num_heads = _positive_count(num_heads, "num_heads")
        self.num_layers = _positive_count(num_layers, "num_layers")
        self.spatial_dim = _positive_count(spatial_dim, "spatial_dim")
        self.mlp_ratio = _positive_count(mlp_ratio, "mlp_ratio")
        self.dropout = float(dropout)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.conditioning_channels = _nonnegative_count(
            conditioning_channels,
            "conditioning_channels",
        )
        self.condition_embed_dim = _positive_count(
            condition_embed_dim,
            "condition_embed_dim",
        )
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if coordinate_normalization not in {"none", "centered_bbox"}:
            raise ValueError(
                "coordinate_normalization must be 'none' or 'centered_bbox'"
            )
        if coordinate_normalization_eps <= 0.0:
            raise ValueError("coordinate_normalization_eps must be positive")

        self.coordinate_normalization = coordinate_normalization
        self.coordinate_normalization_eps = float(coordinate_normalization_eps)
        self._uses_local_edges = bool(uses_local_edges)

        self.input_projection = nn.Linear(self.in_channels, self.hidden_dim)
        if use_coord_mlp:
            self.coordinate_projection = nn.Sequential(
                nn.Linear(self.spatial_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.coordinate_projection = nn.Linear(self.spatial_dim, self.hidden_dim)

        if self.conditioning_channels > 0:
            self.condition_projection = nn.Sequential(
                nn.Linear(
                    self.conditioning_channels * self.condition_embed_dim,
                    self.hidden_dim,
                ),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.condition_projection = None

        self.blocks = nn.ModuleList(
            DiTBlock(
                self.hidden_dim,
                mlp_ratio=self.mlp_ratio,
                attention=attention_factory(self.hidden_dim, self.num_heads),
                dropout=self.dropout,
            )
            for _ in range(self.num_layers)
        )
        self.final_layer = DiTFinalLayer(self.hidden_dim, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def initialize_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize_linear)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.projection.weight)
        nn.init.zeros_(self.final_layer.projection.bias)

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
        del attention_edge_indices
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

        model_edge_index = edge_index
        if self._uses_local_edges:
            _validate_edge_index(
                edge_index,
                num_nodes=inputs.shape[0],
                device=inputs.device,
            )
            model_edge_index = _with_missing_self_loops(
                edge_index,
                num_nodes=inputs.shape[0],
            )

        for block in self.blocks:
            hidden = block(
                hidden,
                condition_embedding,
                edge_index=model_edge_index,
                batch_index=batch_index,
            )

        return self.final_layer(
            hidden,
            condition_embedding,
            batch_index=batch_index,
        )

    def _condition_embedding(
        self,
        inputs: torch.Tensor,
        *,
        batch_index: torch.Tensor | None,
        conditioning: torch.Tensor | None,
    ) -> torch.Tensor:
        num_graphs = _num_graphs(batch_index, inputs.shape[0])
        if self.conditioning_channels == 0:
            if conditioning is not None and (
                conditioning.ndim != 2 or conditioning.shape[1] != 0
            ):
                raise ValueError(
                    "conditioning must have zero columns when conditioning_channels=0"
                )
            return torch.zeros(
                (num_graphs, self.hidden_dim),
                dtype=inputs.dtype,
                device=inputs.device,
            )

        if conditioning is None:
            raise ValueError("conditioning is required when conditioning_channels > 0")
        if conditioning.ndim != 2 or conditioning.shape != (
            num_graphs,
            self.conditioning_channels,
        ):
            raise ValueError(
                f"conditioning must have shape [{num_graphs}, {self.conditioning_channels}]"
            )
        if conditioning.dtype != inputs.dtype or conditioning.device != inputs.device:
            raise TypeError("conditioning must share input dtype and device")

        fourier = _sinusoidal_condition_features(
            conditioning,
            embed_dim=self.condition_embed_dim,
        )
        assert self.condition_projection is not None
        return self.condition_projection(fourier)


def _sinusoidal_condition_features(
    conditioning: torch.Tensor,
    *,
    embed_dim: int,
) -> torch.Tensor:
    half_dim = embed_dim // 2
    dtype = torch.float32
    device = conditioning.device
    frequencies = torch.exp(
        -torch.log(torch.tensor(10000.0, device=device, dtype=dtype))
        * torch.arange(half_dim, device=device, dtype=dtype)
        / max(half_dim - 1, 1)
    )
    arguments = conditioning.float().unsqueeze(-1) * frequencies
    features = torch.cat((torch.sin(arguments), torch.cos(arguments)), dim=-1)
    if features.shape[-1] < embed_dim:
        padding = torch.zeros(
            (*features.shape[:-1], 1),
            device=device,
            dtype=dtype,
        )
        features = torch.cat((features, padding), dim=-1)
    return features.reshape(conditioning.shape[0], -1).to(dtype=conditioning.dtype)


def _normalize_coordinates_by_graph(
    coords: torch.Tensor,
    *,
    batch_index: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    if batch_index is None:
        minimum = coords.amin(dim=0, keepdim=True)
        maximum = coords.amax(dim=0, keepdim=True)
        center = 0.5 * (minimum + maximum)
        span = (maximum - minimum).amax(dim=1, keepdim=True).clamp_min(eps)
        return (coords - center) / span

    counts = _validate_packed_batch_index(
        batch_index,
        num_nodes=coords.shape[0],
        device=coords.device,
    )
    num_graphs = int(counts.numel())
    scatter_index = batch_index[:, None].expand(-1, coords.shape[1])
    minimum = torch.full(
        (num_graphs, coords.shape[1]),
        torch.inf,
        dtype=coords.dtype,
        device=coords.device,
    )
    maximum = torch.full(
        (num_graphs, coords.shape[1]),
        -torch.inf,
        dtype=coords.dtype,
        device=coords.device,
    )
    minimum.scatter_reduce_(
        0,
        scatter_index,
        coords,
        reduce="amin",
        include_self=True,
    )
    maximum.scatter_reduce_(
        0,
        scatter_index,
        coords,
        reduce="amax",
        include_self=True,
    )
    center = 0.5 * (minimum + maximum)
    span = (maximum - minimum).amax(dim=1, keepdim=True).clamp_min(eps)
    return (coords - center[batch_index]) / span[batch_index]


def _with_missing_self_loops(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
) -> torch.Tensor:
    nodes = torch.arange(num_nodes, device=edge_index.device, dtype=torch.long)
    has_self = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.bool)
    if edge_index.shape[1] > 0:
        source, target = edge_index
        self_nodes = source[source == target]
        has_self[self_nodes] = True
    missing = nodes[~has_self]
    if missing.numel() == 0:
        return edge_index
    self_edges = torch.stack((missing, missing))
    return torch.cat((edge_index, self_edges), dim=1)


def _broadcast_graph_features(
    features: torch.Tensor,
    *,
    batch_index: torch.Tensor | None,
    num_nodes: int,
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("graph features must have shape [B, C]")
    if batch_index is None:
        if features.shape[0] != 1:
            raise ValueError("single-graph inputs require exactly one graph feature row")
        return features.expand(num_nodes, -1)
    return features[batch_index]


def _num_graphs(batch_index: torch.Tensor | None, num_nodes: int) -> int:
    if batch_index is None:
        return 1
    counts = _validate_packed_batch_index(
        batch_index,
        num_nodes=num_nodes,
        device=batch_index.device,
    )
    return int(counts.numel())


def _validate_packed_batch_index(
    batch_index: torch.Tensor,
    *,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    if batch_index.ndim != 1 or batch_index.shape[0] != num_nodes:
        raise ValueError("batch_index must have shape [N]")
    if batch_index.dtype != torch.long:
        raise TypeError("batch_index must use torch.long indices")
    if batch_index.device != device:
        raise ValueError("batch_index and inputs must be on the same device")
    if num_nodes == 0:
        raise ValueError("DiT requires at least one node")
    if int(batch_index.min()) != 0:
        raise ValueError("packed batch_index must start at graph 0")

    num_graphs = int(batch_index.max()) + 1
    counts = torch.bincount(batch_index, minlength=num_graphs)
    if bool(torch.any(counts == 0)):
        raise ValueError("packed batch_index must not skip graph IDs")
    expected = torch.repeat_interleave(
        torch.arange(num_graphs, device=device, dtype=torch.long),
        counts,
    )
    if not torch.equal(batch_index, expected):
        raise ValueError("packed graph nodes must be contiguous by graph")
    return counts


def _nonnegative_count(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    count = int(value)
    if count < 0 or count != value:
        raise ValueError(f"{name} must be a non-negative integer")
    return count
