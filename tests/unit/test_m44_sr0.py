"""Small CPU gates for the M44 SR-0 conditional Flow Matching contract."""

from dataclasses import replace

import pytest
import torch

from graph_attention.data.collate import PackedBatch
from graph_attention.data.contracts import ReferenceScale, ReferenceScales
from graph_attention.models.dinat_dit import AlternatingDilatedGeometricDiT
from graph_attention.tasks.regression import NodeRegressionBatch
from graph_attention.tasks.sr0 import (
    CoarseToFineCSR,
    SR0FlowMatching,
    SR0PairBatch,
    fit_sr0_scales,
)
from graph_attention.training.model_factory import instantiate_controlled_model
from omegaconf import OmegaConf


def _references() -> ReferenceScales:
    def ref(name: str, units: str) -> ReferenceScale:
        return ReferenceScale(
            name=name,
            value=1.0,
            units=units,
            provenance="synthetic SR-0 contract test",
            definition="unit reference for synthetic sample",
            inference_available=True,
        )

    return ReferenceScales(
        scheme="synthetic_convective",
        scales=(ref("rho_ref", "kg/m^3"), ref("U_ref", "m/s"), ref("L_ref", "m")),
    )


def _pair(sample_id: str = "train-a", *, with_target: bool = True) -> SR0PairBatch:
    coords = torch.tensor([[0.0, 0, 0], [0.5, 0, 0], [1.0, 0, 0]])
    local = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    dilated = torch.tensor([[0, 2], [2, 0]], dtype=torch.long)
    source = PackedBatch(
        coords=coords,
        edge_index=local,
        batch_index=torch.zeros(3, dtype=torch.long),
        ptr=torch.tensor([0, 3], dtype=torch.long),
        fields={},
        node_weights=torch.tensor([1.0, 2.0, 1.0]),
        sample_ids=(sample_id,),
        mesh_ids=("fine-mesh",),
        case_ids=("case-a",),
        reference_scales=(_references(),),
        regime_parameters=(),
        sample_metadata=({},),
        mesh_metadata=({},),
    )
    fine = NodeRegressionBatch(
        source=source,
        coords=coords,
        inputs=torch.zeros(3, 5),
        targets=torch.zeros(3, 5),
        conditioning=torch.empty(1, 0),
        input_channels=("rho", "rhou", "rhov", "rhow", "rhoE"),
        target_channels=("rho", "rhou", "rhov", "rhow", "rhoE"),
        conditioning_names=(),
        attention_edge_indices={"dilated": dilated},
    )
    transfer = CoarseToFineCSR(
        crow_indices=torch.tensor([0, 1, 3, 4]),
        col_indices=torch.tensor([0, 0, 1, 1]),
        weights=torch.tensor([1.0, 0.5, 0.5, 1.0]),
        num_coarse_nodes=2,
    )
    coarse = torch.tensor([[1.0, 0.5, 0.3, 0.2, 4.0], [1.2, 0.6, 0.4, 0.3, 5.0]])
    baseline = transfer.apply(coarse)
    residual = torch.tensor(
        [[0.01, 0.02, 0.03, 0.04, 0.05],
         [0.06, -0.03, -0.02, 0.01, -0.04],
         [-0.01, -0.02, -0.03, -0.04, -0.05]]
    )
    target = baseline + residual if with_target else None
    return SR0PairBatch(
        fine=fine,
        coarse_state=coarse,
        fine_baseline=baseline,
        fine_target=target,
        transfer=transfer,
        coarse_ptr=torch.tensor([0, 2], dtype=torch.long),
    )


def test_sparse_parent_transfer_and_partition_of_unity() -> None:
    pair = _pair()
    torch.testing.assert_close(
        pair.transfer.apply(pair.coarse_state),
        torch.stack(
            [pair.coarse_state[0], pair.coarse_state.mean(dim=0), pair.coarse_state[1]]
        ),
    )
    torch.testing.assert_close(
        pair.transfer.apply(torch.ones(2, 5)), torch.ones(3, 5)
    )
    with pytest.raises(ValueError, match="preserve constants"):
        replace(pair.transfer, weights=torch.tensor([1.0, 0.2, 0.5, 1.0]))


def test_m44_requires_provenance_for_lref() -> None:
    pair = _pair()
    invalid_source = replace(pair.fine.source, reference_scales=(ReferenceScales(),))
    with pytest.raises(ValueError, match="reference scheme"):
        replace(pair, fine=replace(pair.fine, source=invalid_source))


def test_sr0_scales_are_fitted_only_from_explicit_training_ids() -> None:
    pair = _pair()
    with pytest.raises(ValueError, match="unapproved"):
        fit_sr0_scales([pair], train_sample_ids=("heldout",))
    with pytest.raises(ValueError, match="missing expected"):
        fit_sr0_scales([pair], train_sample_ids=("train-a", "train-b"))
    stats = fit_sr0_scales([pair], train_sample_ids=("train-a",))
    assert stats.train_sample_ids == ("train-a",)
    assert torch.all(stats.coarse_std > 0)
    assert torch.all(stats.residual_rms > 0)


def test_sr0_flow_path_is_original_m11_and_inputs_have_15_channels() -> None:
    pair = _pair()
    stats = fit_sr0_scales([pair], train_sample_ids=("train-a",))
    task = SR0FlowMatching(stats, time_embedding_scale=1000.0)
    gen = torch.Generator().manual_seed(123)
    problem = task.make_training_problem(pair, generator=gen)
    reference = torch.Generator().manual_seed(123)
    t = torch.rand((1,), generator=reference)
    eps = torch.randn((3, 5), generator=reference)
    scale = stats.residual_rms.to(
        device=pair.fine_baseline.device, dtype=pair.fine_baseline.dtype
    )
    z = (pair.fine_target - pair.fine_baseline) / scale
    torch.testing.assert_close(problem.flow.inputs, (1 - t) * eps + t * z)
    torch.testing.assert_close(problem.flow.targets, z - eps)
    torch.testing.assert_close(problem.flow.conditioning[:, -1], t * 1000)
    assert task.model_probe(problem).inputs.shape == (3, 15)
    expected_interp = pair.transfer.apply(
        ((pair.coarse_state - stats.coarse_mean) / stats.coarse_std).float()
    )
    torch.testing.assert_close(problem.spatial_condition[:, :5], expected_interp)


def test_sr0_trains_with_unchanged_dinat_backbone() -> None:
    pair = _pair()
    stats = fit_sr0_scales([pair], train_sample_ids=("train-a",))
    task = SR0FlowMatching(stats)
    problem = task.make_training_problem(pair, generator=torch.Generator().manual_seed(23))
    config = OmegaConf.create(
        {
            "_target_": "graph_attention.models.dinat_dit.AlternatingDilatedGeometricDiT",
            "hidden_dim": 8,
            "num_heads": 2,
            "num_layers": 2,
            "mlp_ratio": 2,
            "condition_embed_dim": 8,
            "sparse_attention_backend": "scatter",
        }
    )
    model, metadata = instantiate_controlled_model(config, task.model_probe(problem), seed=7)
    assert isinstance(model, AlternatingDilatedGeometricDiT)
    assert model.in_channels == 15 and model.out_channels == 5
    assert metadata["architecture_family"] == "dit_adaln_zero"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    before = model.final_layer.projection.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    objective = task.training_loss(model, problem).mean
    assert torch.isfinite(objective)
    objective.backward()
    optimizer.step()
    assert not torch.equal(before, model.final_layer.projection.weight)


class _TimeVelocity(torch.nn.Module):
    def forward(
        self,
        inputs: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        coords: torch.Tensor,
        batch_index: torch.Tensor,
        conditioning: torch.Tensor,
        attention_edge_indices: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        del edge_index, coords, attention_edge_indices
        assert inputs.shape[1] == 15
        return conditioning[batch_index, -1:].expand(-1, 5)


def test_generation_is_les_only_and_reuses_m11_heun() -> None:
    pair = _pair()
    scales = fit_sr0_scales([pair], train_sample_ids=("train-a",))
    task = SR0FlowMatching(scales)
    condition_only = replace(pair, fine_target=None)
    euler = task.generate(_TimeVelocity(), condition_only, steps=1, solver="euler")
    heun = task.generate(_TimeVelocity(), condition_only, steps=4, solver="heun")
    torch.testing.assert_close(heun, euler + 0.5 * scales.residual_rms.float())
    assert torch.isfinite(heun).all()
    different = task.generate(
        _TimeVelocity(), condition_only, steps=4, solver="heun", sampling_keys=("draw-2",)
    )
    assert not torch.equal(different, heun)
    torch.testing.assert_close(
        task.generate(_TimeVelocity(), condition_only, steps=4, solver="heun"), heun
    )
