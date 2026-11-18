"""M47.3 mathematical invariants; no archive or GPU needed."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from graph_attention.training.scaling import ChannelStandardizer, TaskStandardizers
from scripts.m47_fm_diagnostics import (
    aggregate_velocity, heun_snapshots, observation_steps, roundtrip_audit,
    velocity_metrics,
)
from graph_attention.tasks.flow_matching import FlowMatchingTask


def test_velocity_metrics_known_prediction_and_zero_baseline():
    p = torch.tensor([[0., 2.], [2., 0.]])
    y = torch.tensor([[1., 2.], [1., 0.]])
    result = velocity_metrics(p, y, ("rho", "rhou"))
    assert result["rho"]["mse"] == pytest.approx(1)
    assert result["rho"]["zero_predictor_mse"] == pytest.approx(1)
    assert result["rho"]["improvement_over_zero_predictor"] == pytest.approx(0)
    assert result["rhou"]["mse"] == pytest.approx(0)
    assert result["rhou"]["cosine_similarity"] == pytest.approx(1)
    with pytest.raises(ValueError):
        velocity_metrics(p, y, ("rho",))


def test_observation_steps_validate_and_snap_to_heun_grid():
    assert observation_steps(8, (0., .25, .5, .75, 1.)) == (0, 2, 4, 6, 8)
    assert observation_steps(3, (0., .25, .5, 1.)) == (0, 1, 2, 3)
    with pytest.raises(ValueError):
        observation_steps(8, (0., 1.1))
    with pytest.raises(ValueError):
        observation_steps(8, (0., 0.))


def test_roundtrip_audit_fm_scaler_and_raw_state():
    names = ("rho", "rhou")
    channel = ChannelStandardizer(names, torch.tensor([10., 20.]), torch.tensor([2., 4.]))
    scalers = TaskStandardizers(channel, channel, ("train",), True)
    original = torch.tensor([[11., 22.], [15., 14.]])
    raw = SimpleNamespace(input_channels=names, target_channels=names,
                          inputs=original, targets=original.clone())
    scaled_state = channel.transform(original, names)
    scaled = SimpleNamespace(input_channels=names, target_channels=names,
                             inputs=scaled_state, targets=scaled_state.clone())
    result = roundtrip_audit(raw, scaled, scalers)
    assert result["max_standardized_roundtrip"] == pytest.approx(0.)
    bad = SimpleNamespace(input_channels=names, target_channels=names,
                          inputs=original, targets=original + 1.)
    with pytest.raises(ValueError, match="physical inputs and targets"):
        roundtrip_audit(bad, scaled, scalers)


@dataclass
class _FakePacked:
    sample_ids: tuple[str, ...]
    ptr: torch.Tensor


class _SimpleBatch:
    def __init__(self):
        self.inputs = torch.zeros((4, 2))
        self.targets = self.inputs.clone()
        self.source = _FakePacked(("s",), torch.tensor([0, 4]))
        self.conditioning = torch.zeros((1, 0))
        self.conditioning_names = ()
        self.input_channels = ("rho", "rhou")
        self.target_channels = self.input_channels
        self.batch_index = torch.zeros(4, dtype=torch.long)
        self.edge_index = torch.empty((2, 0), dtype=torch.long)
        self.coords = torch.zeros((4, 3))
        self.attention_edge_indices = {}
        self.num_graphs = 1
        self.ptr = self.source.ptr


class _ToyVelocity(torch.nn.Module):
    def forward(self, state, *, coords, edge_index, batch_index, conditioning,
                attention_edge_indices=None):
        return -0.25 * state + 0.002 * conditioning[:, -1][batch_index, None]


def test_heun_snapshots_identical_to_production_sampler_final_state():
    batch = _SimpleBatch()
    model = _ToyVelocity()
    task = FlowMatchingTask(("rho", "rhou"), time_embedding_scale=1000.)
    result = heun_snapshots(
        model, task, batch, steps=4, seed=41, sampling_key="sample",
        observe_steps={0, 2, 4},
    )
    expected = task.sample_standardized(
        model, batch, steps=4, solver="heun", sampling_seed=41,
        sampling_keys=("sample",),
    )
    assert sorted(result) == [0, 2, 4]
    assert torch.equal(result[4], expected)
    assert not torch.equal(result[0], result[4])


def test_aggregate_velocity_equal_sample_weighting():
    channels = ("rho", "rhou", "rhov", "rhow", "rhoE")
    def entry(mse):
        return {"mse": mse, "zero_predictor_mse": 4.,
                "predicted_rms": 2., "target_rms": 2.,
                "cosine_similarity": 0.5}
    rows = [
        {"case": "A", "by_time": [{"channels": {k: entry(1.) for k in channels}}]},
        {"case": "A", "by_time": [{"channels": {k: entry(3.) for k in channels}}]},
    ]
    out = aggregate_velocity(rows, ("A",), (0.,))
    assert out["A"]["0.0"]["channels"]["rho"]["mse"] == pytest.approx(2.)
    assert out["A"]["0.0"]["channels"]["rho"]["improvement_over_zero_predictor"] == pytest.approx(0.5)
