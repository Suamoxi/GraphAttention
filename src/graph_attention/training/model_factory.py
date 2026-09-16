"""Model construction policies shared by controlled training runners."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig

from graph_attention.models import (
    AlternatingDilatedGeometricSparseGraphTransformer,
    GeometricSparseGraphTransformer,
    SparseGraphTransformer,
)
from graph_attention.tasks import NodeRegressionBatch

_M8_TARGET = "graph_attention.models.SparseGraphTransformer"
_M9_TARGET = "graph_attention.models.GeometricSparseGraphTransformer"
_M12_DILATED_TARGET = "graph_attention.models.AlternatingDilatedGeometricSparseGraphTransformer"


def instantiate_controlled_model(
    model_cfg: DictConfig,
    probe: NodeRegressionBatch,
    *,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Instantiate the controlled M8/M9/M12 model family with matched initialization.

    This is the exact initialization policy previously embedded in
    ``scripts.train_slice_ablation``. Moving it here lets generative runners stay
    independent of a particular dataset script without changing parameter values.
    """

    target = str(model_cfg.get("_target_", ""))
    if target not in {_M8_TARGET, _M9_TARGET, _M12_DILATED_TARGET}:
        raise TypeError("unsupported controlled graph model class")

    common = {
        "in_channels": probe.inputs.shape[1],
        "out_channels": probe.targets.shape[1],
        "hidden_dim": int(model_cfg.hidden_dim),
        "num_heads": int(model_cfg.num_heads),
        "num_layers": int(model_cfg.num_layers),
        "mlp_ratio": int(model_cfg.mlp_ratio),
        "conditioning_channels": probe.conditioning.shape[1],
    }

    torch.manual_seed(seed)
    reference = SparseGraphTransformer(**common)
    if target == _M8_TARGET:
        return reference, {
            "policy": "shared_m8_reference_initialization",
            "shared_parameter_seed": seed,
            "geometry_parameter_seed": None,
        }

    geometry_seed = seed + 1
    torch.manual_seed(geometry_seed)
    geometric_reference = GeometricSparseGraphTransformer(
        **common,
        spatial_dim=probe.coords.shape[1],
    )
    incompatible = geometric_reference.load_state_dict(reference.state_dict(), strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected keys while matching M8/M9 initialization: {incompatible.unexpected_keys}"
        )
    expected_geometry_keys = sorted(
        name for name in geometric_reference.state_dict() if ".geometry_mlp." in name
    )
    if sorted(incompatible.missing_keys) != expected_geometry_keys:
        raise RuntimeError(
            "M8/M9 shared-parameter initialization mismatch: "
            f"missing={sorted(incompatible.missing_keys)}, expected={expected_geometry_keys}"
        )

    if target == _M9_TARGET:
        return geometric_reference, {
            "policy": "matched_m8_shared_parameters_plus_m9_geometry",
            "shared_parameter_seed": seed,
            "geometry_parameter_seed": geometry_seed,
            "geometry_parameter_names": expected_geometry_keys,
        }

    model = AlternatingDilatedGeometricSparseGraphTransformer(
        **common,
        spatial_dim=probe.coords.shape[1],
    )
    model.load_state_dict(geometric_reference.state_dict(), strict=True)
    return model, {
        "policy": "exact_m9_parameter_initialization_with_external_alternating_topology",
        "shared_parameter_seed": seed,
        "geometry_parameter_seed": geometry_seed,
        "geometry_parameter_names": expected_geometry_keys,
        "layer_topology_schedule": "local_exact2hop_alternating_local_first",
    }
