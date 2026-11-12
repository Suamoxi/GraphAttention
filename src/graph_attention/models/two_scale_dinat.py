"""Static and dynamic two-scale DiNAT vector fields for residual Flow Matching.

All physical Q/B operations and training statistics are external. Inputs are
already nondimensionalized/standardized; the network predicts *normalized*
fine residual velocity. Cross-scale edges are [key_source, query_target].
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .dinat_dit import AlternatingDilatedGeometricDiT


class SparseCrossScaleAttention(nn.Module):
    """Rectangular, geometry-biased cross-attention with sparse fixed support.

    Prior weights are positive interpolation-support weights, NOT the Q or B
    physical operator. Attention is normalized per target query node.
    Queries with no available keys return zero (safe for irrelevant halo nodes).
    """

    def __init__(self, query_dim: int, key_dim: int, output_dim: int, heads: int = 4) -> None:
        super().__init__()
        if min(query_dim, key_dim, output_dim, heads) <= 0 or output_dim % heads:
            raise ValueError("positive dimensions and output_dim divisible by heads required")
        self.heads = heads
        self.head_dim = output_dim // heads
        self.q = nn.Linear(query_dim, output_dim)
        self.k = nn.Linear(key_dim, output_dim)
        self.v = nn.Linear(key_dim, output_dim)
        self.geo = nn.Sequential(nn.Linear(3, output_dim), nn.SiLU(), nn.Linear(output_dim, heads))
        self.out = nn.Linear(output_dim, output_dim)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        *,
        query_coords: torch.Tensor,
        key_coords: torch.Tensor,
        edges: torch.Tensor,
        weights: torch.Tensor,
        coarse_spacing: float,
        periodic_length: float,
    ) -> torch.Tensor:
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long:
            raise ValueError("bipartite edges must be int64 [2, E]")
        if weights.shape != (edges.shape[1],) or not torch.isfinite(weights).all() or torch.any(weights <= 0):
            raise ValueError("each cross-scale edge needs one positive finite prior weight")
        if query_coords.shape != (queries.shape[0], 3) or key_coords.shape != (keys.shape[0], 3):
            raise ValueError("cross-scale coordinate shape mismatch")
        if coarse_spacing <= 0 or periodic_length <= 0:
            raise ValueError("coarse spacing and periodic length must be positive")
        if queries.dtype != keys.dtype or queries.dtype != weights.dtype:
            raise TypeError("query, key and weight tensors must share dtype")
        if not (queries.device == keys.device == weights.device == edges.device
                == query_coords.device == key_coords.device):
            raise ValueError("cross-scale arrays must share device")
        nquery = queries.shape[0]
        result = queries.new_zeros((nquery, self.heads, self.head_dim))
        if edges.shape[1] == 0:
            return result.reshape(nquery, -1)
        source, target = edges.unbind(0)
        if (torch.any(source < 0) or torch.any(source >= keys.shape[0])
                or torch.any(target < 0) or torch.any(target >= nquery)):
            raise ValueError("cross-scale edges reference nonexistent nodes")
        q = self.q(queries).reshape(nquery, self.heads, self.head_dim)
        k = self.k(keys).reshape(keys.shape[0], self.heads, self.head_dim)
        v = self.v(keys).reshape(keys.shape[0], self.heads, self.head_dim)
        displacement = key_coords[source] - query_coords[target]
        displacement = torch.remainder(displacement + periodic_length / 2, periodic_length) - periodic_length / 2
        relative = displacement / coarse_spacing
        scores = ((q[target] * k[source]).sum(-1) / math.sqrt(self.head_dim)
                  + self.geo(relative) + torch.log(weights)[:, None])
        target_heads = target[:, None].expand(-1, self.heads)
        maxima = torch.full((nquery, self.heads), -torch.inf, dtype=scores.dtype, device=scores.device)
        maxima.scatter_reduce_(0, target_heads, scores, reduce="amax", include_self=True)
        exponent = torch.exp(scores - maxima[target])
        denominators = torch.zeros_like(maxima)
        denominators.scatter_add_(0, target_heads, exponent)
        alpha = exponent / denominators[target]
        result.index_add_(0, target, alpha.unsqueeze(-1) * v[source])
        present = denominators[:, 0] > 0
        output = self.out(result.reshape(nquery, -1))
        return output * present[:, None].to(output.dtype)


def _feature_dinat(
    in_channels: int, out_channels: int, *, hidden: int, heads: int,
    layers: int, condition_channels: int,
) -> AlternatingDilatedGeometricDiT:
    model = AlternatingDilatedGeometricDiT(
        in_channels=in_channels, out_channels=out_channels, hidden_dim=hidden,
        num_heads=heads, num_layers=layers, spatial_dim=3,
        conditioning_channels=condition_channels, coordinate_normalization="none",
        use_absolute_coordinate_embedding=False,
    )
    # The zero-output initialization of DiT is appropriate for a velocity
    # predictor, but would otherwise make every initial latent feature zero.
    nn.init.xavier_uniform_(model.final_layer.projection.weight)
    nn.init.zeros_(model.final_layer.projection.bias)
    return model


class TwoScaleDiNAT(nn.Module):
    """Shared static/dynamic architecture with one fine residual FM process.

    Static coarse encoding is conditional on log2(r) and never on FM time.
    Fine blocks and the optional dynamic coarse block use [t, log2(r)].
    """

    def __init__(
        self, *, mode: str, coarse_width: int = 64, fine_width: int = 64,
        num_heads: int = 4, fine_pre_layers: int = 2,
        fine_post_layers: int = 2, coarse_static_layers: int = 2,
        coarse_dynamic_layers: int = 4, channels: int = 5,
    ) -> None:
        super().__init__()
        if mode not in {"static", "dynamic"}:
            raise ValueError("mode must be static or dynamic")
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.mode = mode
        self.fine_pre_layers = fine_pre_layers
        self.fine_post_layers = fine_post_layers
        self.coarse_static_layers = coarse_static_layers
        self.coarse_dynamic_layers = coarse_dynamic_layers if mode == "dynamic" else 0
        self.coarse_encoder = _feature_dinat(
            channels, coarse_width, hidden=coarse_width, heads=num_heads,
            layers=coarse_static_layers, condition_channels=1,
        )
        self.fine_encoder = _feature_dinat(
            2 * channels, fine_width, hidden=fine_width, heads=num_heads,
            layers=fine_pre_layers, condition_channels=2,
        )
        self.up = SparseCrossScaleAttention(fine_width, coarse_width, fine_width, num_heads)
        if mode == "dynamic":
            self.down = SparseCrossScaleAttention(coarse_width, fine_width, coarse_width, num_heads)
            self.coarse_dynamic = _feature_dinat(
                coarse_width, coarse_width, hidden=coarse_width, heads=num_heads,
                layers=coarse_dynamic_layers, condition_channels=2,
            )
        self.fine_decoder = AlternatingDilatedGeometricDiT(
            in_channels=fine_width, out_channels=channels, hidden_dim=fine_width,
            num_heads=num_heads, num_layers=fine_post_layers, spatial_dim=3,
            conditioning_channels=2, coordinate_normalization="none",
            use_absolute_coordinate_embedding=False,
        )

    @staticmethod
    def _scale(refinement: int | torch.Tensor, *, time: torch.Tensor) -> torch.Tensor:
        ratio = torch.as_tensor(refinement, dtype=time.dtype, device=time.device).flatten()
        if ratio.numel() == 1:
            ratio = ratio.expand(time.numel())
        if ratio.shape != time.shape or torch.any(ratio < 2) or torch.any(ratio != ratio.round()):
            raise ValueError("refinement must be integer >= 2 per physical graph")
        return torch.log2(ratio)[:, None]

    def encode_coarse(
        self, coarse_state: torch.Tensor, *, coarse_coords: torch.Tensor,
        coarse_edges: torch.Tensor, coarse_dilated: torch.Tensor,
        refinement: int | torch.Tensor, time: torch.Tensor,
        coarse_batch_index: torch.Tensor | None = None,
        periodic_length: float,
    ) -> torch.Tensor:
        s = self._scale(refinement, time=time)
        return self.coarse_encoder(
            coarse_state, edge_index=coarse_edges, coords=coarse_coords,
            batch_index=coarse_batch_index, conditioning=s,
            attention_edge_indices={"dilated": coarse_dilated},
            periodic_length=periodic_length,
        )

    def forward(
        self, state: torch.Tensor, *, baseline: torch.Tensor,
        coarse_state: torch.Tensor, fine_coords: torch.Tensor,
        coarse_coords: torch.Tensor, fine_edges: torch.Tensor,
        fine_dilated: torch.Tensor, coarse_edges: torch.Tensor,
        coarse_dilated: torch.Tensor, coarse_to_fine: torch.Tensor,
        fine_to_coarse: torch.Tensor, cross_weights: torch.Tensor,
        refinement: int | torch.Tensor, time: torch.Tensor,
        periodic_length: float, coarse_spacing: float,
        fine_batch_index: torch.Tensor | None = None,
        coarse_batch_index: torch.Tensor | None = None,
        cached_coarse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.channels or baseline.shape != state.shape:
            raise ValueError(f"fine FM state and baseline must both be [N_f, {self.channels}]")
        if coarse_state.ndim != 2 or coarse_state.shape[1] != self.channels:
            raise ValueError(f"coarse state must have shape [N_c, {self.channels}]")
        if time.ndim != 1 or time.numel() < 1:
            raise ValueError("time must be [number of physical graphs]")
        if state.dtype != coarse_state.dtype or state.dtype != time.dtype:
            raise TypeError("state, coarse state and time must share dtype")
        s = self._scale(refinement, time=time)
        conditions = torch.cat((time[:, None], s), dim=1)
        fine_kwargs = dict(
            edge_index=fine_edges, coords=fine_coords,
            batch_index=fine_batch_index, conditioning=conditions,
            attention_edge_indices={"dilated": fine_dilated},
            periodic_length=periodic_length,
        )
        coarse_kwargs = dict(
            edge_index=coarse_edges, coords=coarse_coords,
            batch_index=coarse_batch_index, conditioning=conditions,
            attention_edge_indices={"dilated": coarse_dilated},
            periodic_length=periodic_length,
        )
        hc = cached_coarse
        if hc is None:
            hc = self.encode_coarse(
                coarse_state, coarse_coords=coarse_coords,
                coarse_edges=coarse_edges, coarse_dilated=coarse_dilated,
                refinement=refinement, time=time,
                coarse_batch_index=coarse_batch_index,
                periodic_length=periodic_length,
            )
        hf = self.fine_encoder(torch.cat((state, baseline), dim=1), **fine_kwargs)
        if self.mode == "dynamic":
            down = self.down(
                hc, hf, query_coords=coarse_coords, key_coords=fine_coords,
                edges=fine_to_coarse, weights=cross_weights,
                coarse_spacing=coarse_spacing, periodic_length=periodic_length,
            )
            hc = self.coarse_dynamic(hc + down, **coarse_kwargs)
        up = self.up(
            hf, hc, query_coords=fine_coords, key_coords=coarse_coords,
            edges=coarse_to_fine, weights=cross_weights,
            coarse_spacing=coarse_spacing, periodic_length=periodic_length,
        )
        return self.fine_decoder(hf + up, **fine_kwargs)

