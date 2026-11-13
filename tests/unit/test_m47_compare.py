"""Self-contained comparison contract tests; synthetic summaries only."""
import json
from pathlib import Path

import pytest

from scripts.m47_compare import CASES, MODES, compare


def _write(p: Path, payload):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))


def _fixture(root: Path):
    cases = sorted(CASES)
    ids = {case: [f"{case}:solut_hit_{n:08d}" for n in range(41,101)] for case in cases}
    split = {"train_ids":[], "validation_ids":[], "test_ids":[], "excluded_ids":[]}
    for case in cases:
        seq = ids[case]
        for part, sl in (("train_ids",slice(0,40)),("validation_ids",slice(44,49)),
                         ("test_ids",slice(53,60)), ("excluded_ids",slice(40,44))):
            split[part].extend(seq[sl])
        split["excluded_ids"].extend(seq[49:53])
    for i,(mode,conditioning) in enumerate(MODES.items()):
        d = root / mode / "train_s42"
        d.mkdir(parents=True, exist_ok=True)
        (d / "best.pt").write_bytes(b"synthetic checkpoint marker")
        _write(d / "split.json", split)
        _write(d / "run_signature.json", {"content":{
            "mode":"train", "seed":42,"physical_nondimensionalization":True,
            "temporal_embargo":2, "accumulation":6,"samples":[1,2,3],
            "references":{"shared":"reference hash"},
            "m47_physical_conditions":{
                "qualified_manifest_sha256":"demo", "condition_mode":mode,
                "conditioning_parameters":conditioning,
            },
        }})
        val=0.6-0.1*i
        _write(d / "summary.json", {
            "status":"train_complete","mode":"train",
            "conditioning_mode":mode,"conditioning_parameters":conditioning,
            "qualified_stationarity_manifest":True,"num_physical_samples":360,
            "train_samples":240,"validation_samples":30,"test_samples":42,
            "excluded_samples":48,"history":[{},{}],"best_epoch":1,
            "m47_training_hyperparameters":{"seed":42,"learning_rate":1e-4,"planned_epochs":2},
            "best_validation_velocity_mse":val,
            "test_at_best_validation":{"mean":0.5-i*0.1,
                "by_case":{case:{"mse":0.5-i*0.1} for case in cases}},
            "per_case_validation_at_best":{case:{"mse":val} for case in cases},
        })
    return root


def test_matched_ablation_reports_validation_deltas(tmp_path):
    root=_fixture(tmp_path)
    result=compare(root)
    assert result["same_signature_except_condition"] is True
    assert [round(row["validation_mse_delta_vs_none"],3) for row in result["results"]] == [0.,-0.1,-0.2]
    assert result["same_number_of_epochs"] == 2


def test_mismatched_reference_signature_fails(tmp_path):
    root=_fixture(tmp_path)
    p=root / "mach_re/train_s42/run_signature.json"
    sig=json.loads(p.read_text())
    sig["content"]["references"]["shared"]="different"
    _write(p,sig)
    with pytest.raises(ValueError, match="signature or temporal split differs"):
        compare(root)


def test_mismatched_split_fails(tmp_path):
    root=_fixture(tmp_path)
    p=root / "mach/train_s42/split.json"
    split=json.loads(p.read_text())
    split["train_ids"][0],split["test_ids"][0] = split["test_ids"][0],split["train_ids"][0]
    _write(p,split)
    with pytest.raises(ValueError, match="signature or temporal split differs"):
        compare(root)


def test_missing_or_incomplete_arm_fails(tmp_path):
    root=_fixture(tmp_path)
    p=root / "mach_re/train_s42/summary.json"
    p.unlink()
    with pytest.raises(FileNotFoundError):
        compare(root)
    _write(p,{"status":"smoke_passed"})
    with pytest.raises(ValueError, match="incomplete"):
        compare(root)


def test_unequal_epochs_fails(tmp_path):
    root=_fixture(tmp_path)
    p=root / "mach/train_s42/summary.json"
    s=json.loads(p.read_text())
    s["history"].append({})
    _write(p,s)
    with pytest.raises(ValueError, match="unequal training epoch counts"):
        compare(root)


def test_transient_sample_fails(tmp_path):
    root=_fixture(tmp_path)
    p=root / "none/train_s42/split.json"
    split=json.loads(p.read_text())
    split["train_ids"][0]="Re54/Ma0.05:solut_hit_00000030"
    _write(p,split)
    with pytest.raises(ValueError, match="transient"):
        compare(root)


def test_unequal_learning_rate_fails(tmp_path):
    root=_fixture(tmp_path)
    p=root / "mach/train_s42/summary.json"
    s=json.loads(p.read_text())
    s["m47_training_hyperparameters"]["learning_rate"]=2e-4
    _write(p,s)
    with pytest.raises(ValueError, match="unequal learning rates"):
        compare(root)