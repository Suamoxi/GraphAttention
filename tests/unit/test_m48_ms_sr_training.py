"""M48.4 provenance and population-statistics gates (no external MS_SR data)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from graph_attention.data import ms_sr_training as tr
from graph_attention.data.ms_sr_adapter import select_pilot_record
from scripts.m48_ms_sr_train import _owned_cube


def _record(case, sid, *, role="teacher"):
    return {
        "role": role, "source_role": role, "case": case,
        "snapshot_id": sid, "physical_time_s": float(sid),
        "source_sha256": f"source-{case}-{sid}",
        "operator_sha256": "operator-hash",
        "filter_id": "w1p25_l1",
        "fine_factor": 4, "coarse_factor": 8,
    }


def test_explicit_snapshot_identity_is_not_silently_replaced():
    publication = {
        "schema": "ms-sr.campaign.v1",
        "pairs": [_record("ReT060/a", 5), _record("ReT060/a", 20)],
    }
    selection = select_pilot_record(
        publication, snapshot_case="ReT060/a", snapshot_id="20"
    )
    assert selection["snapshot_id"] == 20
    with pytest.raises(ValueError, match="no eligible"):
        select_pilot_record(
            publication, snapshot_case="ReT060/a", snapshot_id=99
        )


def test_round_robin_source_selection_and_locked_rejection(tmp_path: Path):
    records = [
        _record("ReT060/a", 1), _record("ReT060/a", 2),
        _record("ReT060/b", 1), _record("ReT060/b", 2),
        _record("ReT060/a", 3, role="locked_in_regime"),
    ]
    manifest = {"schema": "ms-sr.campaign.v1", "pairs": records}
    path = tmp_path / "dataset" / "manifest.json"
    path.parent.mkdir()
    raw = json.dumps(manifest).encode()
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    chosen, seal = tr.select_role_records(
        tmp_path, role="teacher", max_sources=2, manifest_sha256=digest,
    )
    assert [r["case"] for r in chosen] == ["ReT060/a", "ReT060/b"]
    assert seal == tr._digest([tr.record_identity(r) for r in chosen])
    with pytest.raises(ValueError, match="locked"):
        tr.select_role_records(
            tmp_path, role="locked_in_regime", manifest_sha256=digest,
        )


def test_equal_snapshot_normalization_and_provenance(monkeypatch, tmp_path: Path):
    records = [_record("ReT060/a", 1), _record("ReT060/b", 2)]
    seal = tr._digest([tr.record_identity(r) for r in records])

    def fake_load(_root, **kwargs):
        sid = int(kwargs["snapshot_id"])
        assert kwargs["role"] == "teacher"
        # Uniform snapshots with different means exercise between-source
        # variance, not only within-snapshot variance.
        coarse = torch.full((1, 5, 2, 2, 2), float(2 * sid - 1), dtype=torch.float64)
        residual = torch.full((1, 5, 4, 4, 4), float(2 * sid), dtype=torch.float64)
        return SimpleNamespace(coarse=coarse, residual=residual), None, None

    monkeypatch.setattr(tr, "load_ms_sr_pair", fake_load)
    stats = tr.fit_teacher_normalization(tmp_path, records, seal)
    mean, std, rms = tr.checked_normalization(stats, records, seal)
    torch.testing.assert_close(mean, torch.full((5,), 2.0, dtype=torch.float64))
    torch.testing.assert_close(std, torch.ones(5, dtype=torch.float64))
    torch.testing.assert_close(
        rms, torch.full((5,), (10.0) ** 0.5, dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="provenance"):
        tr.checked_normalization(stats, records, "incorrect-selection")
    with pytest.raises(ValueError, match="teacher"):
        tr.fit_teacher_normalization(
            tmp_path, [_record("ReT060/a", 1, role="validation")],
            "invalid-seal",
        )


def test_owned_core_respects_global_periodicity():
    ids = _owned_cube(8, 2, (7, 1, 1))
    assert len(ids) == 8
    ijk_x = torch.div(ids, 64, rounding_mode="floor")
    assert sorted(set(ijk_x.tolist())) == [0, 7]
