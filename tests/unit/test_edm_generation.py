from scripts.generate_slice_edm import _generation_name


def test_edm_generation_name_records_sigma_schedule() -> None:
    assert (
        _generation_name(
            sampler="edm_karras_heun_steps18",
            sigma_min=0.004,
            sigma_max=160.0,
            rho=7.0,
            seed=5678,
            override=None,
        )
        == "edm_karras_heun_steps18_sigmin0p004_sigmax160p0_rho7p0_seed5678"
    )


def test_edm_generation_name_allows_explicit_override() -> None:
    assert (
        _generation_name(
            sampler="edm_karras_heun_steps18",
            sigma_min=0.004,
            sigma_max=160.0,
            rho=7.0,
            seed=5678,
            override="controlled_edm_run",
        )
        == "controlled_edm_run"
    )
