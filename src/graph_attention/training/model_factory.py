"""Model construction policies shared by controlled training runners."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig

from graph_attention.models import (
    AlternatingDilatedGeometricDiT,
    AlternatingDilatedGeometricSparseGraphTransformer,
    FullDiTGraphTransformer,
    GeometricSparseGraphTransformer,
    LocalDiTGraphTransformer,
    SparseGraphTransformer,
)
from graph_attention.tasks import NodeRegressionBatch

_M8_TARGET = "graph_attention.models.SparseGraphTransformer"
_M9_TARGET = "graph_attention.models.GeometricSparseGraphTransformer"
_M12_DILATED_TARGET = "graph_attention.models.AlternatingDilatedGeometricSparseGraphTransformer"
_FULL_DIT_TARGET = "graph_attention.models.full_dit.FullDiTGraphTransformer"
_LOCAL_DIT_TARGET = "graph_attention.models.local_dit.LocalDiTGraphTransformer"
_DINAT_DIT_TARGET = "graph_attention.models.dinat_dit.AlternatingDilatedGeometricDiT"


def instantiate_controlled_model(
    model_cfg: DictConfig,
    probe: NodeRegressionBatch,
    *,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Instantiate a controlled model family with reproducible matched initialization."""

    target = str(model_cfg.get("_target_", ""))
    if target in {_FULL_DIT_TARGET, _LOCAL_DIT_TARGET, _DINAT_DIT_TARGET}:
        return _instantiate_matched_dit(model_cfg, probe, seed=seed)
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


def _instantiate_matched_dit(
    model_cfg: DictConfig,
    probe: NodeRegressionBatch,
    *,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    target = str(model_cfg.get("_target_", ""))
    common = {
        "in_channels": probe.inputs.shape[1],
        "out_channels": probe.targets.shape[1],
        "hidden_dim": int(model_cfg.hidden_dim),
        "num_heads": int(model_cfg.num_heads),
        "num_layers": int(model_cfg.num_layers),
        "spatial_dim": probe.coords.shape[1],
        "mlp_ratio": int(model_cfg.mlp_ratio),
        "conditioning_channels": probe.conditioning.shape[1],
        "condition_embed_dim": int(
            model_cfg.get("condition_embed_dim", model_cfg.hidden_dim)
        ),
        "use_coord_mlp": bool(model_cfg.get("use_coord_mlp", True)),
        "coordinate_normalization": str(
            model_cfg.get("coordinate_normalization", "centered_bbox")
        ),
        "coordinate_normalization_eps": float(
            model_cfg.get("coordinate_normalization_eps", 1.0e-8)
        ),
        "use_sdpa": bool(model_cfg.get("use_sdpa", True)),
        "dropout": float(model_cfg.get("dropout", 0.0)),
        "qkv_bias": bool(model_cfg.get("qkv_bias", True)),
        "out_proj_bias": bool(model_cfg.get("out_proj_bias", False)),
    }

    torch.manual_seed(seed)
    full_reference = FullDiTGraphTransformer(**common)
    if target == _FULL_DIT_TARGET:
        attention_mode = "full"
    elif target == _LOCAL_DIT_TARGET:
        attention_mode = "local_one_hop_plus_self"
    else:
        attention_mode = "alternating_local_exact2hop_geometric"

    metadata = {
        "policy": (
            "matched_full_local_dit_initialization"
            if target in {_FULL_DIT_TARGET, _LOCAL_DIT_TARGET}
            else "matched_full_dit_shared_parameters_plus_dinat_geometry"
        ),
        "shared_parameter_seed": seed,
        "architecture_family": "dit_adaln_zero",
        "coordinate_conditioning": "absolute_centered_bbox",
        "attention_mode": attention_mode,
    }
    if target == _FULL_DIT_TARGET:
        return full_reference, metadata

    if target == _LOCAL_DIT_TARGET:
        local = LocalDiTGraphTransformer(**common)
        incompatible = local.load_state_dict(full_reference.state_dict(), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "full/local DiT parameter contracts diverged: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        return local, metadata

    geometry_seed = seed + 1
    sparse_attention_backend = str(
        model_cfg.get("sparse_attention_backend", "scatter")
    )
    torch.manual_seed(geometry_seed)
    dinat_dit = AlternatingDilatedGeometricDiT(
        **common,
        sparse_attention_backend=sparse_attention_backend,
    )
    incompatible = dinat_dit.load_state_dict(full_reference.state_dict(), strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "unexpected keys while matching Full DiT/DiNAT-DiT initialization: "
            f"{incompatible.unexpected_keys}"
        )
    expected_geometry_keys = sorted(
        name for name in dinat_dit.state_dict() if ".geometry_mlp." in name
    )
    if sorted(incompatible.missing_keys) != expected_geometry_keys:
        raise RuntimeError(
            "Full DiT/DiNAT-DiT shared-parameter initialization mismatch: "
            f"missing={sorted(incompatible.missing_keys)}, "
            f"expected={expected_geometry_keys}"
        )
    metadata.update(
        {
            "geometry_parameter_seed": geometry_seed,
            "geometry_parameter_names": expected_geometry_keys,
            "layer_topology_schedule": "local_exact2hop_alternating_local_first",
            "sparse_attention_backend": sparse_attention_backend,
        }
    )
    return dinat_dit, metadata
