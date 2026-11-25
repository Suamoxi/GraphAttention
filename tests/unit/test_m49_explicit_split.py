"""Fixed HIT membership and fail-closed explicit split loading."""
import hashlib
import json
from pathlib import Path

import pytest

from graph_attention.data.splits import load_explicit_split_manifest


MANIFEST = Path(__file__).resolve().parents[2] / "configs/splits/m49_hit_les_forced_blocked50.json"


def test_m49_inventory_and_temporal_buffers():
    indices = sorted([*range(760, 13341, 10), 4464, 13342])
    ids = tuple(f"solut_hit_{i:08d}" for i in indices)
    digest = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
    assert digest == "1d748cb02068286ce18c903856fcaf4d4ea9e698fd33d4c2b1fd0dda573b1818"
    split, excluded = load_explicit_split_manifest(MANIFEST, ids, ids)
    assert split.train_ids == ids[:800]
    assert split.validation_ids == ids[850:1030]
    assert split.test_ids == ids[1080:]
    assert excluded == ids[800:850] + ids[1030:1080]
    assert (len(split.train_ids), len(split.validation_ids), len(split.test_ids), len(excluded)) == (800, 180, 181, 100)
    # Discovery order must not affect membership.
    assert load_explicit_split_manifest(MANIFEST, ids[::-1], ids[::-1]) == (split, excluded)
    for changed in (ids[:-1], ids + ("new_snapshot",)):
        with pytest.raises(ValueError, match="inventory mismatch"):
            load_explicit_split_manifest(MANIFEST, changed, changed)


def test_explicit_split_rejects_overlap(tmp_path):
    p = tmp_path / "split.json"
    p.write_text(json.dumps(dict(train_ids=["a"], validation_ids=["b"],
                                 test_ids=["c"], excluded_ids=["a"])))
    with pytest.raises(ValueError, match="duplicate or overlapping"):
        load_explicit_split_manifest(p, ("a","b","c"), ("a","b","c"))


def test_explicit_split_rejects_cross_group_leakage(tmp_path):
    p = tmp_path / "split.json"
    p.write_text(json.dumps(dict(train_ids=["a"], validation_ids=["b"],
                                 test_ids=["c"], excluded_ids=["d"])))
    with pytest.raises(ValueError, match="group leakage"):
        load_explicit_split_manifest(p, ("a","b","c","d"), ("x","x","y","z"))
