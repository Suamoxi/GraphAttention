"""M48.4b frozen normalization and matched static/dynamic training gates.

No external MS_SR dataset is needed for these synthetic/provenance tests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from graph_attention.data.ms_sr_adapter import FIELDS, MANIFEST_SHA256
from graph_attention.data.ms_sr_training import _digest, record_identity
from scripts import m48_4b_frozen_train as train


def _records() -> list[dict]:
    return [
        {
            "role": "teacher", "source_role": "teacher",
            "case": f"ReT{60 + 40 * (case // 11):03d}/Mt{case % 11:02d}_Xi0.0",
            "snapshot_id": snapshot,
            "source_sha256": f"source-{case}-{snapshot}",
            "operator_sha256": "operator-id",
            "filter_id": "w1p25_l1",
            "fine_factor": 4, "coarse_factor": 8,
        }
        for case in range(22) for snapshot in (3, 35)
    ]


def _frozen(records):
    return {
        "schema": "graph_attention.m48_teacher_normalization.v1",
        "role": "teacher",
        "weighting": "equal_independent_teacher_snapshot",
        "fields": list(FIELDS),
        "manifest_sha256": MANIFEST_SHA256,
        "selection_sha256": _digest([record_identity(r) for r in records]),
        "audit_sha256": "a" * 64,
        "source_count": 44, "case_count": 22, "case_prefix": "",
        "filter_id": "w1p25_l1", "fine_factor": 4, "coarse_factor": 8,
        "physical_nondimensionalization": "none; original MS_SR conservative fields",
        "residual_centering": "none; only per-channel RMS scaling",
        "coarse_mean": [1., 0., 0., 0., 1.],
        "coarse_std": [1., 2., 3., 4., 5.],
        "residual_rms": [1., 2., 3., 4., 5.],
    }


def test_production_frozen_file_has_expected_audit_seal():
    payload = json.loads(train.DEFAULT_NORMALIZATION.read_text())
    assert payload["source_count"] == 44
    assert payload["case_count"] == 22
    assert payload["manifest_sha256"] == MANIFEST_SHA256
    assert payload["selection_sha256"] == (
        "32119fababe014f19d3d1fcfa0e06e4e6527f1e8a66d24cf2f84650e00155ca8"
    )
    assert payload["audit_sha256"] == (
        "defcf3ea5ea19644ab2f9b0bd507575ade316f7dc5582fce5fb10eb1e6daf1f4"
    )
    assert payload["fields"] == list(FIELDS)
    assert payload["coarse_mean"] == pytest.approx([
        1.1566685830436794, -1.181017487538634,
        -3.8266518398995886, -2.111530247034476, 276040.0204871589,
    ])
    assert payload["residual_rms"] == pytest.approx([
        0.032872240222765275, 20.912682816067942, 21.08809496595842,
        21.205531005879678, 11345.626151062495,
    ])


def test_frozen_normalization_is_teacher_sealed_and_mean_free(tmp_path: Path):
    records = _records()
    source_sha = _digest([record_identity(r) for r in records])
    payload = _frozen(records)
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(payload))
    validated, scales, sha = train._read_frozen(path, records, source_sha)
    assert validated["sources"] == [record_identity(r) for r in records]
    assert sha == train._sha256(path.read_bytes())
    assert tuple(float(t[0]) for t in scales) == (1., 1., 1.)

    modified = dict(payload, residual_rms=[2., 3., 4., 5., 0.])
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="degenerate"):
        train._read_frozen(path, records, source_sha)

    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="provenance"):
        train._read_frozen(path, records[:-1], source_sha)
    with pytest.raises(ValueError, match="provenance"):
        train._read_frozen(path, records, "incorrect")
    bad = [dict(records[0], role="locked_in_regime"), *records[1:]]
    with pytest.raises(ValueError, match="provenance"):
        train._read_frozen(path, bad, source_sha)


def test_all_44_sources_seen_each_epoch_with_matched_draws():
    records = _records()
    for epoch in range(3):
        for p in (1, 3):
            sampled = []
            start = epoch * 44 * p
            for step in range(start, start + 44 * p):
                got_epoch, source, patch = train._schedule_step(records, step, p, 484)
                assert got_epoch == epoch
                assert 0 <= patch < p
                sampled.append((source["case"], source["snapshot_id"]))
            assert set(sampled) == {
                (r["case"], r["snapshot_id"]) for r in records
            }
            assert all(sampled.count((r["case"], r["snapshot_id"])) == p
                       for r in records)
    assert train._schedule_step(records, 0, 3, 484) == (
        train._schedule_step(records, 0, 3, 484)
    )
    assert train._step_draw(484, 11, 48) == train._step_draw(484, 11, 48)
    assert train._step_draw(484, 11, 48) != train._step_draw(484, 12, 48)


class _Model(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(0.5))
        self.mode = kwargs["mode"]


def _fake_step(model, optimizer, args, **kwargs):
    optimizer.zero_grad(set_to_none=True)
    loss = (model.parameter - (float(kwargs["fm_time"]) + 0.1)).square()
    loss.backward()
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "gradient_norm_preclip": 1.0,
        "fine_nodes_read": 4, "coarse_nodes_read": 2,
        "owned_fine_nodes": 8,
        "projected_source_abs_defect": 0.0,
    }


def _options(tmp_path: Path, mode: str, output_dir: Path):
    return argparse.Namespace(
        ms_sr_root=tmp_path,
        normalization_file=tmp_path / "frozen.json",
        output_dir=output_dir, mode=mode,
        epochs=2, patches_per_source=1, core_side=2,
        width=16, heads=4, depth=1, learning_rate=1e-3,
        grad_clip=1.0, max_fine_read=10000, max_coarse_read=1000,
        checkpoint_every=2, max_steps=3, seed=484,
        device="cpu", resume=False,
    )


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_training_resume_matches_uninterrupted_steps(monkeypatch, tmp_path, mode):
    records = _records()
    seal = _digest([record_identity(r) for r in records])
    frozen = _frozen(records)
    (tmp_path / "frozen.json").write_text(json.dumps(frozen))
    monkeypatch.setattr(
        train, "select_role_records",
        lambda *_args, **_kwargs: (records, seal),
    )
    monkeypatch.setattr(train, "TwoScaleDiNAT", _Model)
    monkeypatch.setattr(
        train, "_prepare_pair",
        lambda *_args, **_kwargs: (
            SimpleNamespace(fine=torch.zeros(1, 5, 8, 8, 8)), None,
            None, None, None,
        ),
    )
    monkeypatch.setattr(train, "_one_step", _fake_step)
    monkeypatch.setattr(
        train.subprocess, "check_output",
        lambda command, **_kwargs: "commit-for-test\n" if command[1] == "rev-parse" else "",
    )

    first = _options(tmp_path, mode, tmp_path / "resumed")
    train.run(first)
    first.resume = True
    first.max_steps = 2
    resumed = train.run(first)
    assert resumed["completed_steps"] == 5

    reference = _options(tmp_path, mode, tmp_path / "reference")
    reference.max_steps = 5
    uninterrupted = train.run(reference)
    assert uninterrupted["completed_steps"] == 5
    for key in ("case", "snapshot_id", "patch_number", "fm_time",
                "core_origin", "loss", "step"):
        assert [r[key] for r in resumed["history"]] == [
            r[key] for r in uninterrupted["history"]
        ]
    checkpoints = [
        torch.load(x / "checkpoint_latest.pt", weights_only=False)
        for x in (first.output_dir, reference.output_dir)
    ]
    torch.testing.assert_close(
        checkpoints[0]["model_state"]["parameter"],
        checkpoints[1]["model_state"]["parameter"],
    )
    assert checkpoints[0]["completed_steps"] == 5
    assert checkpoints[1]["completed_steps"] == 5
    with pytest.raises(ValueError, match="configuration"):
        first.width = 32
        train.run(first)
    assert not (tmp_path / "locked_in_regime").exists()


def test_atomic_json_summary_reports_teacher_only(tmp_path):
    records = _records()
    cfg = {"mode": "dynamic"}
    report = train._summary(
        cfg, [
            {"case": records[0]["case"], "loss": 3., "optimizer_step_seconds": 1.},
            {"case": records[0]["case"], "loss": 1., "optimizer_step_seconds": 2.},
        ], total=44,
    )
    assert report["loss_by_teacher_case"][records[0]["case"]]["mean"] == 2.
    assert not report["training_complete"]
    assert report["total_forward_backward_seconds"] == 3.
    train._write_json_atomic(tmp_path / "report.json", report)
    assert json.loads((tmp_path / "report.json").read_text()) == report
