# Case definition files

Case-definition YAML documents declare physical reference values and optional dimensionless regime descriptors known from the simulation setup. They are authoritative inputs to M3.3 physical preprocessing and are not generated from instantaneous solution snapshots.

A minimal document is:

```yaml
case_id: case_a
reference_scheme: bulk_flow_reference

references:
  rho_ref:
    value: 1.2
    units: kg/m^3
    definition: prescribed_reference_density
    provenance: simulation_setup
    inference_available: true

  U_ref:
    value: 10.0
    units: m/s
    definition: prescribed_reference_velocity
    provenance: simulation_setup
    inference_available: true
    derivation: optional human-readable derivation

  L_ref:
    value: 0.5
    units: m
    definition: prescribed_characteristic_length
    provenance: simulation_setup
    inference_available: true

regime:
  Re:
    value: 100000.0
    definition: rho_ref_U_ref_L_ref_over_mu_ref
    provenance: simulation_setup
    inference_available: true
    derivation: rho_ref * U_ref * L_ref / mu_ref

  Ma:
    value: 0.2
    definition: U_ref_over_a_ref
    provenance: simulation_setup
    inference_available: true
```

Every `value` must be an explicit numeric literal. `derivation` is documentation only and is not evaluated. `scope` is optional and defaults to `case`.

Entries under `references` are dimensional scales and therefore require explicit units. Entries under `regime` are dimensionless descriptors and intentionally do not accept a `units` key. Their physical definition must still be explicit because `Re`, `Re_tau`, `Re_lambda`, and similar names are not interchangeable.

The `regime` mapping is optional. The data layer preserves declared regime parameters on `Sample.regime_parameters`; deciding which of them become model conditioning belongs to the task/model interface later.

Actual case files may live inside or outside the repository. A run associates them through the AVBP data configuration `case_files` mapping and records the resolved file path as provenance. For reproducible experiments, preserve the exact case-definition content or a stable version/hash with the run artifacts.
