"""M46 stationary selection never silently uses all snapshots."""
from __future__ import annotations

import copy

import pytest

from scripts.m46_stationarity_selection import select_snapshots

POLICY = {"version": "m46_stationary_eligibility_v1", "cases": {
    "Re54/Ma0.05": {"first_counter": 41, "last_counter": 100,
                     "transient_end_counter": 10, "require_full_window": True,
                     "evidence": "M43 reviewed exploratory stationary cutoff for Re54/Ma0.05"}}}


def _sample(case: str, counter: int | None, stem: str | None = None) -> dict:
    name = stem or (f"solut_hit_{counter:08d}" if counter is not None else "last_solution")
    return {"sample_id": f"{case}:{name}", "case_id": case,
            "snapshot": f"/archive/{case}/SOLUT/{name}.h5",
            "source": "legacy_archive" if case.startswith("Re54") else "reT_pilot",
            "counter": counter}


def _inventory() -> dict:
    old, new = "Re54/Ma0.05", "ReT060/Mt0.30_Xi0.0"
    return {"cases": [{"case_id": old}, {"case_id": new}],
            "samples": ([_sample(old, n) for n in range(1, 101)]
                        + [_sample(new, n) for n in range(1, 5)]
                        + [_sample(new, None)])}


def test_legacy_reviewed_window_and_modern_pending() -> None:
    qualified, rows, summary = select_snapshots(_inventory(), POLICY)
    assert qualified["stationarity_filter"] is True
    assert qualified["counts"] == {"cases": 1, "snapshots": 60}
    assert summary["pending_cases"] == ["ReT060/Mt0.30_Xi0.0"]
    assert summary["status_counts"] == {
        "excluded_transient": 10,
        "excluded_transition_uncertain": 30,
        "pending_case_qualification": 5,
        "selected_exploratory_stationary": 60,
    }
    assert [s["counter"] for s in qualified["samples"]] == list(range(41, 101))
    assert sum(r["included"] for r in rows) == 60


def test_incomplete_window_is_rejected() -> None:
    manifest = _inventory()
    manifest["samples"] = [s for s in manifest["samples"]
                           if not (s["case_id"] == "Re54/Ma0.05" and s["counter"] == 49)]
    with pytest.raises(ValueError, match=r"missing=\[49\]"):
        select_snapshots(manifest, POLICY)


def test_duplicate_counter_in_window_is_rejected() -> None:
    manifest = _inventory()
    manifest["samples"].append(_sample("Re54/Ma0.05", 42, "solut_hit_00000042"))
    with pytest.raises(ValueError, match="repeated="):
        select_snapshots(manifest, POLICY)


def test_crash_snapshot_remains_visible_but_unselected() -> None:
    manifest = _inventory()
    manifest["samples"].append(_sample("Re54/Ma0.05", 71, "solut_hit_crash_00000071"))
    qualified, rows, summary = select_snapshots(manifest, POLICY)
    assert len(qualified["samples"]) == 60
    assert summary["status_counts"]["pending_noncanonical_filename"] == 1
    assert not next(row for row in rows if "crash" in row["snapshot"])["included"]


def test_policy_requires_reviewed_provenance_and_valid_boundaries() -> None:
    policy = copy.deepcopy(POLICY)
    policy["cases"]["Re54/Ma0.05"]["evidence"] = ""
    with pytest.raises(ValueError, match="provenance"):
        select_snapshots(_inventory(), policy)
    policy = copy.deepcopy(POLICY)
    policy["cases"]["Re54/Ma0.05"]["first_counter"] = 101
    with pytest.raises(ValueError, match="boundaries"):
        select_snapshots(_inventory(), policy)