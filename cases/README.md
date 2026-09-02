# Case definition files

Case-definition YAML documents declare physical reference values known from the simulation setup. They are authoritative inputs to M3.3 physical nondimensionalization and are not generated from instantaneous solution snapshots.

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
```

`value` must be an explicit numeric literal. `derivation` is documentation only and is not evaluated. `scope` is optional and defaults to `case`.

Actual case files may live inside or outside the repository. A run associates them through the AVBP data configuration `case_files` mapping and records the resolved file path as provenance. For reproducible experiments, preserve the exact case-definition content or a stable version/hash with the run artifacts.
