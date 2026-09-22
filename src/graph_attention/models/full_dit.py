"""Full-attention DiT model for packed CFD graphs."""

from __future__ import annotations

from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn

from .dit_common import _BaseDiTGraphTransformer, _validate_packed_batch_index
from .sparse_transformer import _positive_count, _validate_hidden_inputs


class FullDiTMultiheadAttention(nn.Module):
    """Global self-attention within each physical graph."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = _positive_count(hidden_dim, "hidden_dim")
        self.num_heads = _positive_count(num_heads, "num_heads")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = 1.0 / sqrt(self.head_dim)
        self.use_sdpa = bool(use_sdpa)

        # Names/shapes intentionally match LocalDiTMultiheadAttention.
        self.qkv = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor | None,
    ) -> torch.Tensor:
        del edge_index
        _validate_hidden_inputs(
            inputs,
            expected_channels=self.hidden_dim,
            parameter=self.qkv.weight,
        )

        if batch_index is None:
            return self._dense_attention(inputs.unsqueeze(0)).squeeze(0)

        counts = _validate_packed_batch_index(
            batch_index,
            num_nodes=inputs.shape[0],
            device=inputs.device,
        )
        num_graphs = int(counts.numel())

        if bool(torch.all(counts == counts[0])):
            nodes_per_graph = int(counts[0])
            dense = inputs.reshape(num_graphs, nodes_per_graph, self.hidden_dim)
            return self._dense_attention(dense).reshape_as(inputs)

        outputs: list[torch.Tensor] = []
        start = 0
        for count in counts.detach().cpu().tolist():
            stop = start + int(count)
            outputs.append(
                self._dense_attention(inputs[start:stop].unsqueeze(0)).squeeze(0)
            )
            start = stop
        return torch.cat(outputs, dim=0)

    def _dense_attention(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, hidden_dim = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch_size,
            num_nodes,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if self.use_sdpa:
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
            weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(weights, value)

        output = output.transpose(1, 2).contiguous().reshape(
            batch_size,
            num_nodes,
            hidden_dim,
        )
        return self.out_proj(output)


class FullDiTGraphTransformer(_BaseDiTGraphTransformer):
    """DiT whose only attention topology is dense global self-attention."""

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
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            spatial_dim=spatial_dim,
            attention_factory=lambda dim, heads: FullDiTMultiheadAttention(
                dim,
                heads,
                use_sdpa=use_sdpa,
            ),
            uses_local_edges=False,
            mlp_ratio=mlp_ratio,
            conditioning_channels=conditioning_channels,
            condition_embed_dim=condition_embed_dim,
            use_coord_mlp=use_coord_mlp,
            coordinate_normalization=coordinate_normalization,
            coordinate_normalization_eps=coordinate_normalization_eps,
        )
