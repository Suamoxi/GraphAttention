"""M44: one real-like DNS sample into a point-sampled coarse/fine SR-0 batch."""

import pytest
import torch

from graph_attention.data.contracts import Mesh, ReferenceScale, ReferenceScales, Sample
from graph_attention.data.sr0_pilot import build_sr0_pilot_pair
from graph_attention.tasks.sr0 import SR0FlowMatching, fit_sr0_scales

CORNERS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
           (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))


def _dns_sample(side: int = 7):
    xyz = torch.stack(torch.meshgrid(
        *(torch.linspace(0, 1, side) for _ in range(3)), indexing="ij"
    ), -1).reshape(-1, 3)
    starts = torch.stack(torch.meshgrid(
        *(torch.arange(side - 1) for _ in range(3)), indexing="ij"
    ), -1).reshape(-1, 3)
    v = starts[:, None] + torch.tensor(CORNERS)[None]
    conn = (v[..., 0] * side + v[..., 1]) * side + v[..., 2]
    x, y, z = xyz.unbind(1)
    fine_wave = torch.sin(7 * x + 3 * y + 5 * z)
    fields = {
        "rho": 1.0 + 0.03 * fine_wave,
        "rhou": 1.0 + x + 0.1 * fine_wave,
        "rhov": 1.0 + y + 0.1 * fine_wave,
        "rhow": 1.0 + z + 0.1 * fine_wave,
        "rhoE": 10.0 + x + y + z + 0.2 * fine_wave,
    }

    def ref(name, val, units):
        return ReferenceScale(name=name, value=val, units=units,
                              definition="controlled synthetic unit test",
                              provenance="synthetic grid constructed in unit test",
                              inference_available=True)

    refs = ReferenceScales(scheme="convective", scales=(
        ref("rho_ref", 2.0, "kg/m^3"),
        ref("U_ref", 4.0, "m/s"),
        ref("L_ref", 5.0, "m"),
    ))
    return Sample(
        sample_id="dns-sample-a",
        mesh=Mesh(coords=xyz, edge_index=torch.empty((2, 0), dtype=torch.long),
                  cell_connectivity=conn, mesh_id="dns-mesh"),
        fields=fields, reference_scales=refs,
        case_id="synthetic-case",
    )


def test_matched_patch_from_dns_uses_existing_M33_and_attention() -> None:
    dns = _dns_sample()
    pair = build_sr0_pilot_pair(dns, fine_side=7, patch_cells=4, refinement=2,
                                origin=(1, 1, 1))
    assert pair.fine.coords.shape == (125, 3)
    assert pair.coarse_state.shape == (27, 5)
    assert pair.transfer.num_fine_nodes == 125
    assert pair.fine.attention_edge_indices["dilated"].shape[0] == 2
    assert pair.fine.edge_index.shape[1] > 0
    assert pair.fine_target is not None
    assert pair.fine_target.shape == (125, 5)
    # Coordinates were actually divided by the physical reference L_ref=5m.
    assert torch.all(pair.fine.coords >= 0)
    assert float(pair.fine.coords.max()) < 1.0 / 5.0
    counts = pair.transfer.crow_indices[1:] - pair.transfer.crow_indices[:-1]
    roots = torch.nonzero(counts == 1).flatten()
    source = pair.transfer.col_indices[pair.transfer.crow_indices[roots]]
    torch.testing.assert_close(pair.fine_target[roots], pair.coarse_state[source])
    torch.testing.assert_close(pair.fine_baseline, pair.transfer.apply(pair.coarse_state))
    assert float((pair.fine_target - pair.fine_baseline).abs().amax()) > 1.0e-5


def test_pilot_pair_enters_original_flow_matching_without_a_coarse_encoder() -> None:
    pair = build_sr0_pilot_pair(_dns_sample(), fine_side=7, patch_cells=4, refinement=2)
    scales = fit_sr0_scales([pair], train_sample_ids=("dns-sample-a",))
    model_task = SR0FlowMatching(scales, time_embedding_scale=1000)
    problem = model_task.make_training_problem(pair, generator=torch.Generator().manual_seed(21))
    assert model_task.model_probe(problem).inputs.shape == (125, 15)
    assert model_task.model_probe(problem).targets.shape == (125, 5)
    assert problem.flow.conditioning.shape[-1] == 1


def test_pilot_fail_closed_without_case_or_native_connectivity() -> None:
    from dataclasses import replace

    sample = _dns_sample()
    with pytest.raises(ValueError, match="named case"):
        build_sr0_pilot_pair(replace(sample, case_id=None), fine_side=7, patch_cells=4)
    with pytest.raises(ValueError, match="native hexahedral"):
        build_sr0_pilot_pair(replace(sample, mesh=replace(sample.mesh, cell_connectivity=None)),
                            fine_side=7, patch_cells=4)
