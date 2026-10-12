"""Standalone physics-token DiT: Transolver-style dynamic global-state attention.

Unlike M37's one-hop/exact-two-hop DiNAT, attention here does not use mesh
edges. Each head learns soft node-to-state assignments, pools a small number
of global tokens, performs token self-attention, and broadcasts back to every
node. The surrounding adaLN-Zero DiT and Flow Matching interfaces are reused
unchanged. This is an independent architecture, NOT an M37 checkpoint path.

Based on the irregular-mesh Physics-Attention mechanism in Wu et al.,
"Transolver: A Fast Transformer Solver for PDEs on General Geometries",
ICML 2024, arXiv:2402.02366.
"""

from __future__ import annotations

from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn

from .dit_common import _BaseDiTGraphTransformer, _validate_packed_batch_index
from .sparse_transformer import _positive_count, _validate_hidden_inputs


class PhysicsTokenMultiheadAttention(nn.Module):
    """Soft physics-state slicing -> global token attention -> deslicing.

    For each head, each node distributes weight over `num_tokens` learned
    states. State tokens are normalized weighted averages. Token self-
    attention exchanges information globally; deslicing restores one
    output per original node.

    Optional `token_chunk_size` avoids materializing full [B,H,N,M]
    assignment weights within a single forward. It recomputes weights
    during deslicing. Gradient history during training can still scale
    with total N; this is not a fully out-of-core training algorithm.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        num_tokens: int = 64,
        slice_temperature: float = 0.5,
        token_chunk_size: int | None = None,
        use_sdpa: bool = True,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_proj_bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = _positive_count(hidden_dim, "hidden_dim")
        self.num_heads = _positive_count(num_heads, "num_heads")
        self.num_tokens = _positive_count(num_tokens, "num_tokens")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if token_chunk_size is not None:
            token_chunk_size = _positive_count(token_chunk_size, "token_chunk_size")
        self.token_chunk_size = token_chunk_size
        self.head_dim = self.hidden_dim // self.num_heads
        self.use_sdpa = bool(use_sdpa)
        self.attention_dropout = float(dropout)
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if not 0.0 < float(slice_temperature):
            raise ValueError("slice_temperature must be positive")

        # Original irregular-mesh Transolver projects separate slicing and
        # pooling features and uses head-shared token Q/K/V projections.
        self.in_project_x = nn.Linear(self.hidden_dim, self.hidden_dim, bias=qkv_bias)
        self.in_project_fx = nn.Linear(self.hidden_dim, self.hidden_dim, bias=qkv_bias)
        self.in_project_slice = nn.Linear(self.head_dim, self.num_tokens)
        self.temperature = nn.Parameter(
            torch.full((1, self.num_heads, 1, 1), float(slice_temperature))
        )
        self.to_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=out_proj_bias)

    def _slice_weights(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = x.shape
        features = self.in_project_x(x).reshape(
            batch, nodes, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # Clamping prevents nonpositive slicing temperature during training.
        temperature = self.temperature.clamp(min=0.1, max=5.0)
        return F.softmax(
            self.in_project_slice(features) / temperature, dim=-1
        )

    def _features_and_weights(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, nodes, _ = x.shape
        features = self.in_project_fx(x).reshape(
            batch, nodes, self.num_heads, self.head_dim
        ).transpose(1, 2)
        return features, self._slice_weights(x)

    def _aggregate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # [B,H,M,D] token sums; each node has normalized soft assignments
        # across M states. Never normalize weights across nodes.
        size = self.token_chunk_size
        if size is None or x.shape[1] <= size:
            features, weights = self._features_and_weights(x)
            numerator = torch.matmul(weights.transpose(-1, -2), features)
            denominator = weights.sum(dim=-2).unsqueeze(-1)
            return numerator / (denominator + 1e-5), weights

        numerators = []
        denominators = []
        for start in range(0, x.shape[1], size):
            features, weights = self._features_and_weights(x[:, start : start + size])
            numerators.append(torch.matmul(weights.transpose(-1, -2), features))
            denominators.append(weights.sum(dim=-2).unsqueeze(-1))
        # Summing as a tree avoids an autograd graph with a long linear chain.
        numerator = torch.stack(numerators, dim=0).sum(dim=0)
        denominator = torch.stack(denominators, dim=0).sum(dim=0)
        return numerator / (denominator + 1e-5), None

    def _attend_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        q, k, v = self.to_q(tokens), self.to_k(tokens), self.to_v(tokens)
        if self.use_sdpa:
            return F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
            )
        scores = (q @ k.transpose(-1, -2)) / sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(
            weights, p=self.attention_dropout, training=self.training
        )
        return weights @ v

    def _single_or_regular_batch(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = x.shape
        tokens, weights = self._aggregate(x)
        attended = self._attend_tokens(tokens)
        if weights is not None:
            out = torch.matmul(weights, attended)
        else:
            size = self.token_chunk_size
            assert size is not None
            blocks = []
            for start in range(0, nodes, size):
                local_weights = self._slice_weights(
                    x[:, start : start + size]
                )
                blocks.append(torch.matmul(local_weights, attended))
            out = torch.cat(blocks, dim=-2)
        out = out.transpose(1, 2).contiguous().reshape(
            batch, nodes, self.hidden_dim
        )
        return F.dropout(
            self.out_proj(out), p=self.attention_dropout, training=self.training
        )

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor | None,
    ) -> torch.Tensor:
        # Connectivity is intentionally unused. Geometry remains available
        # through the common DiT input-coordinate projection.
        del edge_index
        _validate_hidden_inputs(
            inputs,
            expected_channels=self.hidden_dim,
            parameter=self.in_project_x.weight,
        )
        if batch_index is None:
            return self._single_or_regular_batch(inputs.unsqueeze(0)).squeeze(0)

        counts = _validate_packed_batch_index(
            batch_index, num_nodes=inputs.shape[0], device=inputs.device
        )
        # Fast batched path for equal-sized, packed (contiguous) graphs;
        # otherwise index by explicit graph membership so tokens never mix.
        if bool(torch.all(counts == counts[0])):
            ordered_index = torch.repeat_interleave(
                torch.arange(len(counts), device=inputs.device), counts
            )
            if torch.equal(batch_index, ordered_index):
                reshaped = inputs.reshape(
                    len(counts), int(counts[0]), self.hidden_dim
                )
                return self._single_or_regular_batch(reshaped).reshape_as(inputs)

        output = torch.empty_like(inputs)
        for graph_id in range(len(counts)):
            indices = torch.nonzero(batch_index == graph_id).flatten()
            graph_output = self._single_or_regular_batch(
                inputs.index_select(0, indices).unsqueeze(0)
            ).squeeze(0)
            output = output.index_copy(0, indices, graph_output)
        return output


class PhysicsTokenDiTGraphTransformer(_BaseDiTGraphTransformer):
    """Independent DiT family using learned global physics-state tokens.

    It shares the existing input/coordinate/time conditioning and adaLN-Zero
    architecture with Full DiT and DiNAT-DiT, enabling the unchanged
    FlowMatchingTask and evaluation scripts to operate on this model.
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
        num_tokens: int = 64,
        slice_temperature: float = 0.5,
        token_chunk_size: int | None = None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            spatial_dim=spatial_dim,
            attention_factory=lambda dim, heads: PhysicsTokenMultiheadAttention(
                dim,
                heads,
                num_tokens=num_tokens,
                slice_temperature=slice_temperature,
                token_chunk_size=token_chunk_size,
                use_sdpa=use_sdpa,
                dropout=dropout,
                qkv_bias=qkv_bias,
                out_proj_bias=out_proj_bias,
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
        self.num_tokens = num_tokens
        self.token_chunk_size = token_chunk_size

    def initialize_weights(self) -> None:
        # _BaseDiTGraphTransformer calls this after constructing all blocks.
        # Its generic Xavier initialization would otherwise overwrite the
        # official Transolver orthogonal initialization for slicing weights.
        super().initialize_weights()
        for block in self.blocks:
            nn.init.orthogonal_(block.attention.in_project_slice.weight)
