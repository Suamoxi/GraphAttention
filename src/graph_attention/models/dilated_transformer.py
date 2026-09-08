"""DiNAT-like alternating local and exact-two-hop sparse graph attention."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from graph_attention.geometry import edge_relative_displacement

from .geometric_transformer import (
    GeometricSparseGraphTransformer,
    _validate_model_coords,
)
from .sparse_transformer import (
    _append_conditioning,
    _validate_edge_index,
    _validate_model_inputs,
)

_DILATED_TOPOLOGY_NAME = "dilated"


class AlternatingDilatedGeometricSparseGraphTransformer(GeometricSparseGraphTransformer):
    """Alternate local and externally supplied dilated sparse attention by layer.

    Geometry/topology construction is intentionally external to the model. The
    base ``edge_index`` is used by even-numbered layers (0, 2, ...), while odd
    layers use ``attention_edge_indices["dilated"]``. The first layer is always
    local, matching the initial DiNAT-like experiment definition.
    """

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
        _validate_edge_index(edge_index, num_nodes=inputs.shape[0], device=inputs.device)

        if attention_edge_indices is None or _DILATED_TOPOLOGY_NAME not in attention_edge_indices:
            raise ValueError(
                "alternating dilated attention requires an externally supplied "
                "attention_edge_indices['dilated'] topology"
            )
        dilated_edge_index = attention_edge_indices[_DILATED_TOPOLOGY_NAME]
        _validate_edge_index(
            dilated_edge_index,
            num_nodes=inputs.shape[0],
            device=inputs.device,
        )

        local_displacement = edge_relative_displacement(coords, edge_index)
        dilated_displacement = edge_relative_displacement(coords, dilated_edge_index)

        features = _append_conditioning(
            inputs,
            batch_index=batch_index,
            conditioning=conditioning,
            expected_channels=self.conditioning_channels,
        )
        hidden = self.input_projection(features)
        for layer_index, block in enumerate(self.blocks):
            if layer_index % 2 == 0:
                hidden = block(hidden, edge_index, local_displacement)
            else:
                hidden = block(hidden, dilated_edge_index, dilated_displacement)
        return self.output_projection(self.final_norm(hidden))
