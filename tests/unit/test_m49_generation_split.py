import pytest
from scripts.generate_generative import _manifest_reference_ids, _manifest_test_ids


def test_validation_selection_preserves_default_test():
    manifest = dict(train_ids=["a"], validation_ids=["b"], test_ids=["c"])
    assert _manifest_test_ids(manifest) == ("c",)
    assert _manifest_reference_ids(manifest, "validation") == ("b",)
    with pytest.raises(ValueError):
        _manifest_reference_ids(manifest, "train")
    with pytest.raises(ValueError):
        _manifest_reference_ids(dict(manifest, validation_ids=["a"]), "validation")
    with pytest.raises(ValueError):
        _manifest_reference_ids(dict(manifest, validation_ids=["c"]), "validation")
