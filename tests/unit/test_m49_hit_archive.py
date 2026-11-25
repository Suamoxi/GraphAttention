import h5py
import numpy as np

from scripts.audit_m49_hit_archive import FIELD_PATHS, discover, enrich, compare


def test_discovery_and_overlap_without_assuming_case_identity(tmp_path):
    archive = tmp_path / "archive"
    scratch = tmp_path / "scratch"
    for root, prefix, value in ((archive, "case_a", 1), (archive, "case_b", 2), (scratch, "", 1)):
        d = root / prefix / "RUN" / "SOLUT"
        d.mkdir(parents=True)
        with h5py.File(d / "solut_hit_00000010.h5", "w") as h:
            for name in FIELD_PATHS.values():
                h.create_dataset(name, data=np.full(4, value, dtype=np.float64))
            h.attrs["time"] = .5
    a, b = enrich(discover(archive)), enrich(discover(scratch))
    assert a["counts"]["streams"] == 2
    assert a["counts"]["solutions"] == 2
    result = compare(a, b)
    assert result["same_basename_candidates"] == 2
    assert result["compared_payload_pairs"] == 2
    assert result["identical_payload_pairs"] == 1
    assert result["same_inode_pairs"] == 0
    assert all(r["inspection"]["time_metadata"]["/@time"] == "0.5" for r in a["solutions"])
