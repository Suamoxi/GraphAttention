"""Diagnostics for the M44 point-sampled DNS proxy, not a physical LES operator.

These pure tensor functions deliberately do not change SR-0 Flow Matching,
training, or generated samples. A point-anchor projection is valid ONLY for
the pilot's Q_sample (coarse nodes copied from a fine Cartesian mesh).
"""

from __future__ import annotations

from typing import Any

import torch

STATE_FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE")


def point_sample_anchors(transfer: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fine-node anchors and corresponding coarse IDs of an exact injection.

    Rejects arbitrary interpolation, duplicated coarse anchors and all layouts
    that cannot be interpreted as Q_sample. Never use this for an LES filter.
    """
    rowptr = transfer.crow_indices
    columns = transfer.col_indices
    weights = transfer.weights
    if rowptr.ndim != 1 or columns.ndim != 1 or weights.ndim != 1:
        raise ValueError("transfer must be CSR with one-dimensional components")
    if rowptr.numel() < 2 or int(rowptr[0]) != 0 or int(rowptr[-1]) != columns.numel():
        raise ValueError("invalid CSR row pointers")
    if columns.numel() != weights.numel():
        raise ValueError("CSR columns and weights differ in length")
    anchor = torch.nonzero(rowptr[1:] - rowptr[:-1] == 1).flatten()
    coarse_id = columns[rowptr[anchor]]
    if anchor.numel() != transfer.num_coarse_nodes:
        raise ValueError("point-sample transfer must retain exactly one node per coarse ID")
    expected_ids = torch.arange(transfer.num_coarse_nodes, device=columns.device)
    if not torch.equal(torch.sort(coarse_id).values, expected_ids):
        raise ValueError("each coarse ID must have exactly one retained fine node")
    if not torch.allclose(weights[rowptr[anchor]], torch.ones_like(weights[rowptr[anchor]]),
                          atol=1e-6, rtol=0):
        raise ValueError("point-sample retained node must have weight exactly one")
    return anchor, coarse_id


def project_point_sample(
    generated: torch.Tensor,
    coarse_state: torch.Tensor,
    transfer: Any,
) -> torch.Tensor:
    """Exact Q_sample correction by node replacement (diagnostic ONLY).

    Does not impose cell-integrated conservation, real LES consistency,
    positivity, continuity of derivatives, or any pressure constraint.
    """
    anchor, coarse_id = point_sample_anchors(transfer)
    if generated.ndim != 2 or coarse_state.ndim != 2:
        raise ValueError("generated and coarse state must be 2D node/channel arrays")
    if generated.shape != (transfer.num_fine_nodes, coarse_state.shape[1]):
        raise ValueError("generated fine fields do not match transfer dimensions")
    if coarse_state.shape[0] != transfer.num_coarse_nodes:
        raise ValueError("coarse state node count does not match transfer")
    if generated.device != coarse_state.device or anchor.device != generated.device:
        raise ValueError("projection arrays must share device")
    if generated.dtype != coarse_state.dtype:
        raise TypeError("projection arrays must share dtype")
    corrected = generated.clone()
    corrected[anchor] = coarse_state[coarse_id]
    return corrected


def per_field_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    node_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unweighted per-field MSE, optionally on unobserved fine nodes only."""
    if prediction.ndim != 2 or prediction.shape != target.shape or prediction.shape[1] != 5:
        raise ValueError("prediction and target must have matching shape [N_f, 5]")
    squared = (prediction - target).square()
    if node_mask is None:
        return squared.mean(dim=0)
    if (
        node_mask.dtype != torch.bool or node_mask.ndim != 1
        or node_mask.shape[0] != squared.shape[0]
        or node_mask.device != squared.device
    ):
        raise ValueError("node_mask must be a Boolean tensor [N_f] on the prediction device")
    if not bool(node_mask.any()):
        raise ValueError("per-field MSE needs at least one unobserved node")
    return squared[node_mask].mean(dim=0)


def ensemble_point_proxy_metrics(
    samples: torch.Tensor,
    baseline: torch.Tensor,
    target: torch.Tensor,
    coarse_state: torch.Tensor,
    transfer: Any,
) -> dict[str, Any]:
    """Summarize repeated draws under one fixed coarse condition.

    A single DNS target per coarse condition cannot verify conditional PDFs;
    these are descriptive second-moment and point-anchor checks only.
    """
    if samples.ndim != 3 or samples.shape[2] != 5 or samples.shape[0] < 2:
        raise ValueError("samples must be [draws>=2, fine_nodes, 5]")
    if baseline.shape != samples.shape[1:] or target.shape != baseline.shape:
        raise ValueError("baseline and target must be [fine_nodes, 5]")
    anchor, coarse_id = point_sample_anchors(transfer)
    if coarse_state.shape != (transfer.num_coarse_nodes, 5):
        raise ValueError("coarse state must be [coarse_nodes, 5]")
    if any(t.device != samples.device for t in (baseline, target, coarse_state, anchor)):
        raise ValueError("all ensemble metrics must run on the same device")
    before = samples[:, anchor, :] - coarse_state[coarse_id][None, :, :]
    projected = torch.stack(
        [project_point_sample(draw, coarse_state, transfer) for draw in samples], dim=0
    )
    after = projected[:, anchor, :] - coarse_state[coarse_id][None, :, :]
    nonanchor = torch.ones(samples.shape[1], dtype=torch.bool, device=samples.device)
    nonanchor[anchor] = False
    unaltered = (
        torch.equal(projected[:, nonanchor, :], samples[:, nonanchor, :])
        if bool(nonanchor.any()) else True
    )
    if not unaltered:
        raise RuntimeError("point-sample projection unexpectedly changed an unobserved node")

    raw_residual = samples - baseline[None, :, :]
    projected_residual = projected - baseline[None, :, :]
    true_residual = target - baseline
    per_draw_std = samples.float().std(dim=0, unbiased=False)

    def field_dict(value: torch.Tensor) -> dict[str, float]:
        if value.shape != (5,):
            raise ValueError("field summary must contain exactly five channels")
        return {name: float(value[i].detach().cpu()) for i, name in enumerate(STATE_FIELDS)}

    return {
        "draws": int(samples.shape[0]),
        "coarse_node_count": int(anchor.numel()),
        "raw_point_max_abs_by_field": field_dict(before.abs().amax(dim=(0, 1))),
        "projected_point_max_abs_by_field": field_dict(after.abs().amax(dim=(0, 1))),
        "raw_residual_rms_by_field": field_dict(raw_residual.square().mean(dim=(0, 1)).sqrt()),
        "projected_residual_rms_by_field": field_dict(
            projected_residual.square().mean(dim=(0, 1)).sqrt()
        ),
        "reference_residual_rms_by_field": field_dict(true_residual.square().mean(dim=0).sqrt()),
        "sample_to_sample_std_rms_by_field": field_dict(per_draw_std.square().mean(dim=0).sqrt()),
        "nonpositive_density_fraction_raw": float((samples[..., 0] <= 0).float().mean().cpu()),
        "nonpositive_density_fraction_projected": float((projected[..., 0] <= 0).float().mean().cpu()),
        "projection_nonanchor_unchanged": unaltered,
        "interpretation": (
            "Point-sampled DNS proxy; no physical filtering, volume conservation, "
            "conditional-PDF validation or positivity guarantee"
        ),
    }


def ensemble_residual_moments_and_increments(
    samples: torch.Tensor,
    baseline: torch.Tensor,
    target: torch.Tensor,
    coords: torch.Tensor,
    edge_index: torch.Tensor,
) -> dict[str, Any]:
    """Descriptive conditional moments and minimum-distance graph increments.

    The only spatial diagnostic is a single graph-edge separation. It is
    NOT an energy spectrum, a physical SGS flux, or a conditional PDF test.
    Coordinates are already nondimensionalized by the CFD reference length.
    """
    if samples.ndim != 3 or samples.shape[0] < 2 or samples.shape[-1] != 5:
        raise ValueError("samples must have shape [draws>=2, fine_nodes, 5]")
    num_nodes = samples.shape[1]
    if baseline.shape != (num_nodes, 5) or target.shape != baseline.shape:
        raise ValueError("fine baseline and target must match generated nodes")
    if coords.shape != (num_nodes, 3) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("expected [N_f,3] coords and [2,E] graph edges")
    if edge_index.dtype != torch.long or edge_index.numel() == 0:
        raise ValueError("graph edges must be nonempty int64 indices")
    if torch.any(edge_index < 0) or torch.any(edge_index >= num_nodes):
        raise ValueError("graph edge index is out of bounds")
    if any(x.device != samples.device for x in (baseline, target, coords, edge_index)):
        raise ValueError("spatial diagnostic tensors must share device")
    if not all(bool(torch.isfinite(x).all()) for x in (samples, baseline, target, coords)):
        raise ValueError("nonfinite spatial diagnostic input")

    residual = samples - baseline.unsqueeze(0)
    reference = target - baseline
    ensemble_mean = residual.mean(dim=0)
    ensemble_var = residual.var(dim=0, unbiased=False)

    src, dst = edge_index.unbind(dim=0)
    distance = torch.linalg.vector_norm(coords[src] - coords[dst], dim=1)
    forward = (src < dst) & (distance > 0)
    if not bool(forward.any()):
        raise ValueError("no nonself graph edges for increment metric")
    minimum = distance[forward].amin()
    # Only nearest-neighbor edges on the qualified Cartesian graph.
    nearest = forward & (
        (distance - minimum).abs() <= (minimum * 1.0e-3 + 1.0e-7)
    )
    if not bool(nearest.any()):
        raise RuntimeError("nearest-neighbor edge selection is empty")
    unique_pairs = torch.unique(
        torch.stack((src[nearest], dst[nearest]), dim=1), dim=0
    )
    a, b = unique_pairs.unbind(dim=1)
    generated_increment = (
        residual[:, a, :] - residual[:, b, :]
    ).square().mean(dim=(0, 1)).sqrt()
    reference_increment = (
        reference[a, :] - reference[b, :]
    ).square().mean(dim=0).sqrt()

    def fields(values: torch.Tensor) -> dict[str, float]:
        return {name: float(values[i].detach().cpu()) for i, name in enumerate(STATE_FIELDS)}

    return {
        "ensemble_mean_residual_rms_by_field": fields(
            ensemble_mean.square().mean(dim=0).sqrt()
        ),
        "ensemble_mean_residual_bias_by_field": fields(
            ensemble_mean.mean(dim=0) - reference.mean(dim=0)
        ),
        "ensemble_pointwise_std_rms_by_field": fields(
            ensemble_var.mean(dim=0).sqrt()
        ),
        "reference_residual_spatial_mean_by_field": fields(reference.mean(dim=0)),
        "reference_residual_spatial_std_by_field": fields(
            reference.var(dim=0, unbiased=False).sqrt()
        ),
        "nearest_neighbor_increments": {
            "separation_over_L_ref": float(distance[nearest].mean().detach().cpu()),
            "graph_edge_count": int(unique_pairs.shape[0]),
            "generated_rms_by_field": fields(generated_increment),
            "reference_rms_by_field": fields(reference_increment),
            "metric": "RMS residual increments along unique minimum-distance fine-graph edges",
        },
        "interpretation": (
            "One DNS target per coarse condition; these are descriptive "
            "conditional ensemble moments and one-scale increments, not "
            "conditional-PDF calibration, energy spectra or physical LES."
        ),
    }
