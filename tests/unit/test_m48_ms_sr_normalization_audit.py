"""Synthetic M48 normalization audit properties, with no external CFD files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from graph_attention.data import ms_sr_normalization_audit as audit
from graph_attention.data import ms_sr_training as train
from scripts import m48_normalization_audit as audit_script
from scripts.m48_normalization_audit import _case_rows, _snapshot_rows


def _record(case: str, snapshot: int, role: str = "teacher") -> dict:
    return {
        "role": role, "source_role": role,
        "case": case, "snapshot_id": snapshot,
        "physical_time_s": snapshot * 1e-5,
        "source_sha256": f"hash-{case}-{snapshot}",
        "operator_sha256": "cedar-sha",
        "filter_id": "w1p25_l1", "fine_factor": 4, "coarse_factor": 8,
    }


def _fake_loader(_root: Path, **kwargs):
    case = kwargs["snapshot_case"]
    snapshot = int(kwargs["snapshot_id"])
    assert kwargs["role"] == "teacher"
    assert kwargs["snapshot_case"] == kwargs["case_prefix"]
    assert kwargs["filter_id"] == "w1p25_l1"
    if case == "ReT060/Mt0.23_Xi0.0":
        coarse_value = {5: 1., 20: 3.}[snapshot]
        residual_value = {5: 2., 20: 4.}[snapshot]
        size = 2 if snapshot == 5 else 3
    else:
        coarse_value, residual_value, size = 10., 6., 4
    pair = SimpleNamespace(
        coarse=torch.full((1, 5, size, size, size), coarse_value, dtype=torch.float64),
        fine=torch.zeros((1, 5, 2 * size, 2 * size, 2 * size), dtype=torch.float64),
        residual=torch.full(
            (1, 5, 2 * size, 2 * size, 2 * size), residual_value,
            dtype=torch.float64,
        ),
        metadata={"role": "teacher", "case": case, "snapshot_id": snapshot},
    )
    checks = {"C_residual": {"max_component_scaled_linf": 1e-15}}
    return pair, None, checks


def test_case_moments_and_pooled_normalization_match_training(monkeypatch, tmp_path):
    records = [
        _record("ReT060/Mt0.23_Xi0.0", 5),
        _record("ReT060/Mt0.23_Xi0.0", 20),
        _record("ReT100/Mt0.30_Xi0.0", 8),
    ]
    seal = train._digest([train.record_identity(r) for r in records])
    monkeypatch.setattr(audit, "load_ms_sr_pair", _fake_loader)
    monkeypatch.setattr(train, "load_ms_sr_pair", _fake_loader)
    report = audit.audit_teacher_normalization(tmp_path, records, seal)
    training = train.fit_teacher_normalization(tmp_path, records, seal)

    # Equal weight per snapshot: 1,3,10, even though their grids have
    # 8,27,64 nodes. No equal-per-case averaging is used for the pool.
    expected_mean = 14.0 / 3
    expected_variance = ((1-expected_mean)**2 +
                         (3-expected_mean)**2 +
                         (10-expected_mean)**2) / 3
    pooled = report["pooled"]
    for x in pooled["coarse_mean"]:
        assert x == pytest.approx(expected_mean)
    for x in pooled["coarse_std"]:
        assert x == pytest.approx(expected_variance**0.5)
    for x in pooled["residual_mean"]:
        assert x == pytest.approx(4.)
    for x in pooled["residual_std"]:
        assert x == pytest.approx((8/3)**0.5)
    for x in pooled["residual_rms"]:
        assert x == pytest.approx((56/3)**0.5)
    assert pooled["coarse_mean"] == pytest.approx(training["coarse_mean"])
    assert pooled["coarse_std"] == pytest.approx(training["coarse_std"])
    assert pooled["residual_rms"] == pytest.approx(training["residual_rms"])

    a = report["per_case"]["ReT060/Mt0.23_Xi0.0"]
    b = report["per_case"]["ReT100/Mt0.30_Xi0.0"]
    assert a["source_count"] == 2 and b["source_count"] == 1
    assert a["moments"]["coarse_mean"] == pytest.approx([2.] * 5)
    assert a["moments"]["coarse_std"] == pytest.approx([1.] * 5)
    assert b["moments"]["coarse_mean"] == pytest.approx([10.] * 5)
    assert b["moments"]["coarse_std"] == pytest.approx([0.] * 5)
    assert a["moments"]["residual_rms"] == pytest.approx([10**0.5] * 5)
    assert b["moments"]["residual_rms"] == pytest.approx([6.] * 5)
    assert report["case_count"] == 2 and report["reynolds_group_count"] == 2
    assert len(_snapshot_rows(report)) == 15
    rows = _case_rows(report)
    assert len(rows) == (1+2+2)*5
    pooled_row = next(x for x in rows if x["scope"] == "pooled"
                      and x["field"] == "rho")
    assert pooled_row["residual_mean_over_global_rms"] == pytest.approx(
        4.0 / (56/3)**0.5
    )


def test_residual_mean_diagnostic_is_not_centering():
    coarse = torch.ones((1, 5, 2, 2, 2), dtype=torch.float64)
    residual = torch.full((1, 5, 2, 2, 2), 2.5, dtype=torch.float64)
    result = audit._population([audit._source_moments(coarse, residual)])
    assert result["residual_mean"].tolist() == [2.5] * 5
    assert result["residual_rms"].tolist() == [2.5] * 5
    assert result["residual_std"].tolist() == [0.] * 5
    # This records the nonzero mean, but does not subtract it.


@pytest.mark.parametrize("role", ["validation", "locked_in_regime"])
def test_audit_rejects_non_teacher_roles_without_loading(monkeypatch, tmp_path, role):
    def reject_loader(*_args, **_kwargs):
        raise AssertionError("must not load non-teacher data")
    monkeypatch.setattr(audit, "load_ms_sr_pair", reject_loader)
    records = [_record("ReT060/Mt0.23_Xi0.0", 5, role=role)]
    seal = train._digest([train.record_identity(r) for r in records])
    with pytest.raises(ValueError, match="teacher-only"):
        audit.audit_teacher_normalization(tmp_path, records, seal)


def test_audit_rejects_invalid_seal_duplicate_sources_and_mixed_filters(tmp_path):
    source = _record("ReT060/Mt0.23_Xi0.0", 5)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit.audit_teacher_normalization(tmp_path, [source], "not-this-seal")
    two = [source, dict(source)]
    with pytest.raises(ValueError, match="duplicate"):
        audit.audit_teacher_normalization(
            tmp_path, two, train._digest([train.record_identity(r) for r in two]),
        )
    mixed = [source, {**_record("ReT060/Mt0.23_Xi0.0", 20), "filter_id": "w1p5_l1"}]
    with pytest.raises(ValueError, match="mix filters"):
        audit.audit_teacher_normalization(
            tmp_path, mixed, train._digest([train.record_identity(r) for r in mixed]),
        )


def test_pooled_normalization_requires_nonzero_scales():
    tensors = audit._population([audit._source_moments(
        torch.ones((1, 5, 2, 2, 2), dtype=torch.float64),
        torch.zeros((1, 5, 2, 2, 2), dtype=torch.float64),
    )])
    with pytest.raises(ValueError, match="degenerate"):
        audit._comparison(tensors, tensors)


def test_subtracting_a_nonzero_constant_can_break_a_nullspace_constraint():
    # Constant-preserving restriction does not necessarily imply spatial
    # mean conservation. This abstract example isolates the logical issue.
    C = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]], dtype=torch.float64)
    residual = torch.tensor([0., 0., 2., 2.], dtype=torch.float64)
    torch.testing.assert_close(C @ residual, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(C @ torch.ones(4, dtype=torch.float64),
                               torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(C @ (residual / 3),
                               torch.zeros(2, dtype=torch.float64))
    assert not torch.allclose(C @ (residual - residual.mean()),
                              torch.zeros(2, dtype=torch.float64))


def test_audit_entrypoint_writes_json_and_csv(monkeypatch, tmp_path):
    """Exercise the real report-writing path without CFD I/O or a GPU."""
    records = [
        _record("ReT060/Mt0.23_Xi0.0", 5),
        _record("ReT060/Mt0.23_Xi0.0", 20),
        _record("ReT100/Mt0.30_Xi0.0", 8),
    ]
    selection_sha = train._digest([train.record_identity(r) for r in records])
    monkeypatch.setattr(
        audit_script, "select_role_records",
        lambda *_args, **_kwargs: (records, selection_sha),
    )
    monkeypatch.setattr(audit, "load_ms_sr_pair", _fake_loader)
    monkeypatch.setattr(
        audit_script.subprocess, "check_output",
        lambda command, **_kwargs: (
            "test-commit\n" if command[1] == "rev-parse" else ""
        ),
    )
    ms_sr_root = tmp_path / "MS_SR"
    code = ms_sr_root / "code"
    code.mkdir(parents=True)
    (code / "campaign_data.py").write_text("# synthetic\n")
    (code / "models.py").write_text("# synthetic\n")
    output_dir = tmp_path / "results"
    args = argparse.Namespace(
        ms_sr_root=ms_sr_root,
        output_dir=output_dir,
        manifest_sha256="test-manifest",
        filter_id="w1p25_l1",
        case_prefix="",
        max_sources=0,
        expected_sources=3,
    )
    report = audit_script.run(args)
    assert report["source_count"] == 3
    assert report["provenance"]["torch_version"] == torch.__version__
    assert report["provenance"]["graph_attention_commit"] == "test-commit"
    saved = json.loads((output_dir / "normalization_audit.json").read_text())
    assert saved["pooled"] == report["pooled"]
    assert len((output_dir / "case_comparison.csv").read_text().splitlines()) == 26
    assert len((output_dir / "snapshot_moments.csv").read_text().splitlines()) == 16
    with pytest.raises(FileExistsError, match="existing"):
        audit_script.run(args)
