"""Sparse one-hop transformer blocks operating directly on packed graph edges."""

from __future__ import annotations

from math import sqrt
from operator import index as operator_index

import torch
from torch import nn


class SparseMultiheadAttention(nn.Module):
    """Scaled dot-product attention restricted to explicitly supplied directed edges."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_dim = _positive_count(hidden_dim, "hidden_dim")
        self.num_heads = _positive_count(num_heads, "num_heads")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = 1.0 / sqrt(self.head_dim)
        self.qkv = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

    def forward(self, inputs: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        _validate_hidden_inputs(
            inputs,
            expected_channels=self.hidden_dim,
            parameter=self.qkv.weight,
        )
        _validate_edge_index(edge_index, num_nodes=inputs.shape[0], device=inputs.device)
        return self._forward_validated(inputs, edge_index)

    def _forward_validated(
        self,
        inputs: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = inputs.shape[0]
        qkv = self.qkv(inputs).reshape(
            num_nodes,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=1)

        if edge_index.shape[1] == 0:
            zero_message = query.reshape(num_nodes, self.hidden_dim) * 0.0
            return self.out_proj(zero_message)

        source = edge_index[0]
        target = edge_index[1]

        query_edge = query[target]
        key_edge = key[source]
        if query_edge.dtype in (torch.float16, torch.bfloat16):
            scores = (query_edge.float() * key_edge.float()).sum(dim=-1) * self.scale
        else:
            scores = (query_edge * key_edge).sum(dim=-1) * self.scale

        target_by_head = target[:, None].expand(-1, self.num_heads)
        max_per_target = torch.full(
            (num_nodes, self.num_heads),
            -torch.inf,
            dtype=scores.dtype,
            device=scores.device,
        )
        max_per_target.scatter_reduce_(
            0,
            target_by_head,
            scores,
            reduce="amax",
            include_self=True,
        )

        exp_scores = torch.exp(scores - max_per_target[target])
        denominator = torch.zeros(
            (num_nodes, self.num_heads),
            dtype=scores.dtype,
            device=scores.device,
        )
        denominator.scatter_add_(0, target_by_head, exp_scores)
        weights = exp_scores / denominator[target]

        messages = weights.to(dtype=value.dtype).unsqueeze(-1) * value[source]
        aggregated = torch.zeros_like(query)
        aggregated.index_add_(0, target, messages)

        return self.out_proj(aggregated.reshape(num_nodes, self.hidden_dim))


class SparseGraphTransformerBlock(nn.Module):
    """Pre-norm transformer block with one-hop sparse graph attention."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        hidden = _positive_count(hidden_dim, "hidden_dim")
        ratio = _positive_count(mlp_ratio, "mlp_ratio")
        self.norm1 = nn.LayerNorm(hidden)
        self.attention = SparseMultiheadAttention(hidden, num_heads)
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * ratio),
            nn.GELU(),
            nn.Linear(hidden * ratio, hidden),
        )

    def forward(self, inputs: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = inputs + self.attention._forward_validated(self.norm1(inputs), edge_index)
        return hidden + self.mlp(self.norm2(hidden))


class SparseGraphTransformer(nn.Module):
    """Node-to-node transformer using only the supplied one-hop sparse topology."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: int = 4,
        conditioning_channels: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_count(in_channels, "in_channels")
        self.out_channels = _positive_count(out_channels, "out_channels")
        self.hidden_dim = _positive_count(hidden_dim, "hidden_dim")
        self.num_heads = _positive_count(num_heads, "num_heads")
        self.num_layers = _positive_count(num_layers, "num_layers")
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
            SparseGraphTransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
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
        batch_index: torch.Tensor | None = None,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_model_inputs(
            inputs,
            expected_channels=self.in_channels,
            parameter=self.input_projection.weight,
        )
        _validate_edge_index(edge_index, num_nodes=inputs.shape[0], device=inputs.device)

        features = _append_conditioning(
            inputs,
            batch_index=batch_index,
            conditioning=conditioning,
            expected_channels=self.conditioning_channels,
        )
        hidden = self.input_projection(features)
        for block in self.blocks:
            hidden = block(hidden, edge_index)
        return self.output_projection(self.final_norm(hidden))


def _append_conditioning(
    inputs: torch.Tensor,
    *,
    batch_index: torch.Tensor | None,
    conditioning: torch.Tensor | None,
    expected_channels: int,
) -> torch.Tensor:
    if expected_channels == 0:
        if conditioning is not None and (conditioning.ndim != 2 or conditioning.shape[1] != 0):
            raise ValueError("conditioning must have zero columns when conditioning_channels=0")
        return inputs

    if batch_index is None or conditioning is None:
        raise ValueError("batch_index and conditioning are required when conditioning_channels > 0")
    if batch_index.ndim != 1 or batch_index.shape[0] != inputs.shape[0]:
        raise ValueError("batch_index must have shape [N]")
    if batch_index.dtype != torch.long:
        raise TypeError("batch_index must use torch.long indices")
    if batch_index.device != inputs.device:
        raise ValueError("batch_index and inputs must be on the same device")
    if conditioning.ndim != 2 or conditioning.shape[1] != expected_channels:
        raise ValueError(f"conditioning must have shape [B, {expected_channels}]")
    if not conditioning.is_floating_point():
        raise TypeError("conditioning must use a floating-point dtype")
    if conditioning.dtype != inputs.dtype:
        raise TypeError("conditioning and inputs must share one dtype")
    if conditioning.device != inputs.device:
        raise ValueError("conditioning and inputs must be on the same device")

    if batch_index.numel() > 0:
        if int(batch_index.min()) < 0:
            raise ValueError("batch_index contains a negative graph index")
        if int(batch_index.max()) >= conditioning.shape[0]:
            raise ValueError("batch_index references a graph outside conditioning")
    return torch.cat((inputs, conditioning[batch_index]), dim=1)


def _validate_model_inputs(
    inputs: torch.Tensor,
    *,
    expected_channels: int,
    parameter: torch.Tensor,
) -> None:
    if inputs.ndim != 2:
        raise ValueError("inputs must have shape [N, C]")
    if not inputs.is_floating_point():
        raise TypeError("inputs must use a floating-point dtype")
    if inputs.shape[1] != expected_channels:
        raise ValueError(f"inputs have {inputs.shape[1]} channels, expected {expected_channels}")
    if inputs.dtype != parameter.dtype:
        raise TypeError(f"inputs use dtype {inputs.dtype}, expected model dtype {parameter.dtype}")
    if inputs.device != parameter.device:
        raise ValueError(f"inputs are on {inputs.device}, expected model device {parameter.device}")


def _validate_hidden_inputs(
    inputs: torch.Tensor,
    *,
    expected_channels: int,
    parameter: torch.Tensor,
) -> None:
    if inputs.ndim != 2 or inputs.shape[1] != expected_channels:
        raise ValueError(f"inputs must have shape [N, {expected_channels}]")
    if not inputs.is_floating_point():
        raise TypeError("inputs must use a floating-point dtype")
    if inputs.dtype != parameter.dtype:
        raise TypeError(f"inputs use dtype {inputs.dtype}, expected model dtype {parameter.dtype}")
    if inputs.device != parameter.device:
        raise ValueError(f"inputs are on {inputs.device}, expected model device {parameter.device}")


def _validate_edge_index(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    device: torch.device,
) -> None:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must use torch.long indices")
    if edge_index.device != device:
        raise ValueError("edge_index and inputs must be on the same device")
    if edge_index.numel() == 0:
        return
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
        raise ValueError("edge_index contains an out-of-range node index")


def _positive_count(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        count = operator_index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if count <= 0:
        raise ValueError(f"{name} must be positive")
    return count


def _nonnegative_count(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        count = operator_index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if count < 0:
        raise ValueError(f"{name} must be non-negative")
    return count
