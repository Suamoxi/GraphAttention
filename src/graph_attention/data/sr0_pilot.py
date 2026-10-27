"""One-snapshot M44 SR-0 numerical pilot from an existing AVBP DNS sample.

This is point-sampled DNS, NOT a physical LES coarse-graining operator. Only
use for testing paired data geometry, conditioning and gradient propagation.
"""

from __future__ import annotations

from graph_attention.data.avbp import AVBP_FIELD_CATALOG
from graph_attention.data.contracts import Mesh, Sample
from graph_attention.geometry.sr0_cartesian import SR0CartesianPatch, build_sr0_cartesian_patch
from graph_attention.tasks.flow_matching import FlowMatchingTask
from graph_attention.tasks.sr0 import CoarseToFineCSR, SR0PairBatch
from graph_attention.training.data_pipeline import GraphTaskCollator

import torch

_STATE = ("rho", "rhou", "rhov", "rhow", "rhoE")


def build_sr0_pilot_pair(
    dns: Sample,
    *,
    fine_side: int,
    patch_cells: int = 8,
    refinement: int = 2,
    origin: tuple[int, int, int] = (0, 0, 0),
    prepared_geometry: SR0CartesianPatch | None = None,
) -> SR0PairBatch:
    """Build one 3D fine patch paired with its point-sampled coarse lattice.

    ``dns`` comes from the existing AVBP reader with actual M3.3 case scales.
    The geometry helper rejects invalid Cartesian node/cell layouts; the normal
    GraphTaskCollator builds M37 one-hop/exact-two-hop fine attention edges.
    No new physical restriction, SGS operator or grid filter is introduced.
    """
    if dns.mesh.cell_connectivity is None:
        raise ValueError("SR-0 pilot requires native hexahedral connectivity")
    if any(name not in dns.fields for name in _STATE):
        raise ValueError("SR-0 pilot requires the five conservative DNS fields")
    if dns.case_id is None:
        raise ValueError("SR-0 pilot requires a named case with audited M3.3 references")
    if prepared_geometry is None:
        geometry = build_sr0_cartesian_patch(
            dns.mesh.coords,
            dns.mesh.cell_connectivity,
            side=fine_side,
            patch_cells=patch_cells,
            refinement=refinement,
            origin=origin,
        )
    else:
        geometry = prepared_geometry
        if (
            geometry.origin != origin
            or geometry.fine_side != patch_cells + 1
            or geometry.coarse_side != patch_cells // refinement + 1
            or geometry.refinement != refinement
            or dns.mesh.coords.shape[0] != fine_side**3
            or geometry.fine_node_ids.max().item() >= dns.mesh.coords.shape[0]
        ):
            raise ValueError("prepared SR-0 patch geometry does not match the requested mesh")
        if not torch.equal(geometry.fine_coords, dns.mesh.coords[geometry.fine_node_ids]):
            raise ValueError("prepared SR-0 patch geometry does not match source coordinates")
    mesh = Mesh(
        coords=geometry.fine_coords,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        mesh_id=f"{dns.mesh.mesh_id}/sr0-point-patch-{origin}-{patch_cells}",
        cell_connectivity=geometry.fine_hexes,
        metadata={
            "pilot": "SR-0 point-sampled DNS (not LES)",
            "origin": origin,
            "fine_side": geometry.fine_side,
            "coarse_side": geometry.coarse_side,
            "refinement": refinement,
        },
    )
    fields = {
        name: dns.fields[name][geometry.fine_node_ids].reshape(-1).to(dtype=mesh.coords.dtype)
        for name in _STATE
    }
    sampled = Sample(
        sample_id=dns.sample_id,
        mesh=mesh,
        fields=fields,
        reference_scales=dns.reference_scales,
        regime_parameters=dns.regime_parameters,
        metadata={**dns.metadata, "sr0_pilot": "point_sampled_dns_not_les"},
        case_id=dns.case_id,
    )
    fine_task = FlowMatchingTask(state_fields=_STATE, physical_nondimensionalization=True)
    collator = GraphTaskCollator(
        fine_task,
        AVBP_FIELD_CATALOG,
        {"attention_edge_indices": {"dilated": "exact_two_hop"}},
    )
    fine_batch = collator([sampled])
    coarse_state = fine_batch.inputs[geometry.coarse_local_ids]
    transfer = CoarseToFineCSR(
        geometry.transfer_rowptr,
        geometry.transfer_columns,
        geometry.transfer_weights.to(dtype=fine_batch.inputs.dtype),
        num_coarse_nodes=coarse_state.shape[0],
    )
    # L=T is deterministic interpolation, not a conservative physical lifting.
    baseline = transfer.apply(coarse_state)
    return SR0PairBatch(
        fine=fine_batch,
        coarse_state=coarse_state,
        fine_baseline=baseline,
        transfer=transfer,
        fine_target=fine_batch.inputs,
        coarse_ptr=torch.tensor([0, coarse_state.shape[0]], dtype=torch.long),
    )
