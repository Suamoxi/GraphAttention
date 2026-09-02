from graph_attention.utils.provenance import collect_runtime_provenance


def test_runtime_provenance_has_m1_sections() -> None:
    provenance = collect_runtime_provenance()

    assert set(provenance) == {"git", "runtime", "hardware"}
    assert "python" in provenance["runtime"]
    assert "pytorch" in provenance["runtime"]
    assert "cuda_available" in provenance["runtime"]
    assert "gpu_count" in provenance["hardware"]
