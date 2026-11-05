"""M45.4: exact Gaussian positive control and matched training scopes."""

from pathlib import Path

import torch

from graph_attention.data.two_scale_synthetic import (
    SyntheticQB, fit_synthetic_scales, make_synthetic_pair,
)
from graph_attention.geometry.two_scale_hit import build_hit_two_scale_geometry
from scripts.m45_checkpoint_diagnostics import BlockGaussianOracle, _generate_oracle
from scripts.m45_synthetic_pilot import _train_one, _to_device


class _PointwiseVelocity(torch.nn.Module):
    """Small test-only backbone; the M37 graph parity is tested separately."""
    mode = "static"
    fine_pre_layers = 1
    fine_post_layers = 1
    coarse_static_layers = 1
    coarse_dynamic_layers = 0

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, state, *, baseline, coarse_state, **kwargs):
        return self.linear(torch.cat((state, baseline), dim=-1))

    def encode_coarse(self, coarse_state, **kwargs):
        return coarse_state


def test_oracle_heun_local_is_known_gaussian_transport():
    """For local block noise, the exact Gaussian flow rescales each block."""
    mesh = build_hit_two_scale_geometry(coarse_side=3, refinement=2)
    qb = SyntheticQB(mesh, "block_average")
    pairs = [make_synthetic_pair(qb, case="local", sample_id=i, seed=100 + i)
             for i in range(4)]
    scales = fit_synthetic_scales(pairs)
    oracle = BlockGaussianOracle(qb, pairs[0].coarse, scales,
                                 "local", torch.device("cpu"))
    seed = 27
    gen = torch.Generator(device="cpu").manual_seed(seed)
    xi = qb.nullspace(torch.randn((qb.nf**3, 5), generator=gen))
    baseline = qb.reconstruct(pairs[0].coarse)
    strength = 0.75 + 0.25 * torch.tanh(20 * (baseline[:, :1] - 1.0))
    channel_scale = torch.tensor((0.018, 0.023, 0.023, 0.023, 0.050))
    expected = strength * channel_scale * xi
    actual = _generate_oracle(oracle, qb, scales, device=torch.device("cpu"),
                              seed=seed, steps=64)
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=1e-5)
    torch.testing.assert_close(qb.restrict(actual),
                               torch.zeros_like(pairs[0].coarse), atol=1e-6, rtol=0)


def test_training_scope_owned_equivalence_and_full_field_difference(tmp_path: Path):
    """Only the supervised node set should change; noise paths are identical."""
    mesh = build_hit_two_scale_geometry(coarse_side=3, refinement=2)
    qb = SyntheticQB(mesh, "block_average")
    train = [make_synthetic_pair(qb, case="local", sample_id=i, seed=100 + i)
             for i in range(3)]
    val = [make_synthetic_pair(qb, case="local", sample_id=100000+i, seed=900+i)
           for i in range(2)]
    scales = fit_synthetic_scales(train)
    torch.manual_seed(43)
    initial = _PointwiseVelocity()
    weights = {k: v.detach().clone() for k, v in initial.state_dict().items()}
    state_dicts = {}
    reports = {}
    for scope in ("patch_owned", "full_owned", "full_all"):
        model = _PointwiseVelocity()
        model.load_state_dict(weights)
        reports[scope] = _train_one(
            model, mesh, mesh, qb, qb, train, val, scales,
            projected=True, device=torch.device("cpu"), seed=79,
            steps=2, lr=1e-3, core_side=3, heun_steps=2, sample_count=1,
            out_dir=tmp_path / scope, training_scope=scope,
        )
        state_dicts[scope] = model.state_dict()
    for key in weights:
        torch.testing.assert_close(state_dicts["patch_owned"][key],
                                   state_dicts["full_owned"][key], rtol=1e-5, atol=1e-6)
    assert reports["patch_owned"]["training_scope"] == "patch_owned"
    assert reports["full_all"]["mean_fine_read_to_owned"] == 1.0
    assert reports["full_owned"]["mean_fine_read_to_owned"] > 1.0
    assert reports["full_all"]["train_samples"] == 3
    # All-node supervision is intentionally a different objective, not a
    # supposed numerically equivalent implementation of the owned-node loss.
    assert any(not torch.equal(state_dicts["full_all"][k], state_dicts["full_owned"][k])
               for k in weights)


def test_checkpoint_diagnostic_cli_oracle_and_fixed_validation_split(tmp_path: Path, monkeypatch):
    """End-to-end CSV smoke, including distinct neural/oracle Heun step lists."""
    import csv
    import json
    import sys
    from scripts import m45_checkpoint_diagnostics as diagnostics

    mesh = build_hit_two_scale_geometry(coarse_side=3, refinement=2)
    qb = SyntheticQB(mesh, "block_average")
    train = [make_synthetic_pair(qb, case="local", sample_id=i, seed=10+i)
             for i in range(3)]
    scales = fit_synthetic_scales(train)
    checkpoint_root = tmp_path / "inputs"
    checkpoint_root.mkdir()
    settings = {
        "operator": "block_average", "source_subspace": "nullspace",
        "seed": 10, "val_seed": 9910, "val_samples": 2,
        "coarse_side": 3, "refinement": 2, "width": 16, "heads": 4,
    }
    (checkpoint_root / "summary.json").write_text(json.dumps({"settings": settings}))
    monkeypatch.setattr(diagnostics, "_load_model",
                        lambda path, config, device: (_PointwiseVelocity().to(device), scales))
    output = tmp_path / "diagnostics"
    monkeypatch.setattr(sys, "argv", [
        "m45_checkpoint_diagnostics", "--checkpoint-root", str(checkpoint_root),
        "--output-dir", str(output), "--device", "cpu",
        "--cases", "local", "--modes", "static", "--generated-samples", "4",
        "--heun-steps", "2", "--oracle-sampling", "--oracle-heun-steps", "2", "4",
        "--eval-times", "0.25", "--velocity-noise-draws", "1",
    ])
    diagnostics.main()
    with (output / "generation_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len([r for r in rows if r["mode"] == "oracle"]) == 2
    assert len([r for r in rows if r["mode"] == "static"]) == 1
    assert len([r for r in rows if r["mode"] == "reference"]) == 1
    assert json.loads((output / "summary.json").read_text())["source_settings"]["val_seed"] == 9910
    with (output / "time_summary.csv").open() as f:
        time_rows = list(csv.DictReader(f))
    assert len(time_rows) == 1 and int(time_rows[0]["validation_coarse_fields"]) == 2
