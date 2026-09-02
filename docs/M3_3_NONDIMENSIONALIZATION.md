# M3.3 Physical Nondimensionalization Directive

## Status

The M3.3 scientific definition is frozen. The baseline reference metadata and forward/inverse convective field transformations were validated on Calypso with the 47-test suite on 2026-09-02. M3.3 now also implements explicit YAML case-definition loading and attachment of declared case references to AVBP samples; that new integration still requires target-environment validation. Coordinate nondimensionalization, regime conditioning, and statistical scaling remain deferred.

## 1. Purpose

M3.3 defines how dimensional CFD quantities become physically dimensionless before any machine-learning statistical scaling.

The required sequence is:

$$
\boxed{
q_{\mathrm{dim}}
\rightarrow
q^*
\rightarrow
\hat q
}
$$

where `q*` is physically nondimensional and `q_hat` may later be statistically scaled using training-set statistics.

Physical nondimensionalization and statistical standardization are separate operations with separate provenance.

## 2. Generality across CFD problems

GraphAttention does not define one universal problem-specific meaning for `U_ref` or `L_ref` across every CFD flow family.

Instead, each physical case declares a reference scheme appropriate to the problem. The numerical values and concrete physical definitions may differ between flow families, but the definitions must be explicit, reproducible, and stable within the chosen scheme.

Examples of valid scheme semantics include:

- freestream density, freestream velocity, and body length for an external flow;
- bulk density, bulk velocity, and hydraulic diameter for an internal flow;
- reference density, bulk velocity, and channel half-height for channel flow;
- reference density, prescribed turbulent velocity scale, and prescribed forcing/integral length scale for HIT.

These examples are not automatic inference rules and do not create required software subclasses. The configured case metadata must state the actual definitions used.

## 3. Baseline reference basis

For compressible Navier-Stokes-like cases, the baseline physical basis is:

$$
\boxed{
\rho_{\mathrm{ref}},\quad
U_{\mathrm{ref}},\quad
L_{\mathrm{ref}},\quad
T_{\mathrm{ref}}
}
$$

when temperature is part of the physical problem.

From this basis the framework uses the coherent convective scales:

$$
t_{\mathrm{ref}}=\frac{L_{\mathrm{ref}}}{U_{\mathrm{ref}}},
$$

$$
(\rho u)_{\mathrm{ref}}=\rho_{\mathrm{ref}}U_{\mathrm{ref}},
$$

$$
p_{\mathrm{scale}}=\rho_{\mathrm{ref}}U_{\mathrm{ref}}^2,
$$

$$
(\rho E)_{\mathrm{scale}}=\rho_{\mathrm{ref}}U_{\mathrm{ref}}^2,
$$

$$
\mu_{\mathrm{scale}}=\rho_{\mathrm{ref}}U_{\mathrm{ref}}L_{\mathrm{ref}}.
$$

A quantity that is irrelevant to the governing problem must not be invented merely to satisfy an implementation interface. For example, an incompressible problem need not fabricate `T_ref` or `Ma`.

## 4. Baseline field transformations

For the AVBP conservative state:

$$
\boxed{
\rho^*=\frac{\rho}{\rho_{\mathrm{ref}}}
}
$$

and for each momentum-density component:

$$
\boxed{
(\rho u_i)^*=\frac{\rho u_i}{\rho_{\mathrm{ref}}U_{\mathrm{ref}}}
}
$$

with the same scale applied consistently to `rhou`, `rhov`, and `rhow`.

Total-energy density uses:

$$
\boxed{
(\rho E)^*=\frac{\rho E}{\rho_{\mathrm{ref}}U_{\mathrm{ref}}^2}
}
$$

Auxiliary fields use the same coherent basis when their exact physical semantics are known:

$$
p^*=\frac{p}{\rho_{\mathrm{ref}}U_{\mathrm{ref}}^2},
$$

$$
T^*=\frac{T}{T_{\mathrm{ref}}},
$$

$$
\mu^*=\frac{\mu}{\rho_{\mathrm{ref}}U_{\mathrm{ref}}L_{\mathrm{ref}}}.
$$

A pressure fluctuation or offset transformation such as `(p - p_ref) / (rho_ref U_ref^2)` is allowed only when that offset has an explicit physical/task definition. It is not the default substitute for the absolute pressure scaling above.

The generic viscosity equation applies to dynamic viscosity `mu`. The current AVBP fields `vis_lam` and `vis_turb` are not mapped to that equation until their exact stored physical convention is confirmed. M3.3 does not infer dynamic-versus-kinematic viscosity semantics from their names.

The baseline coordinate nondimensionalization is:

$$
\boxed{
\mathbf x^*=\frac{\mathbf x}{L_{\mathrm{ref}}}
}
$$

An additional translation such as `(x - x_origin) / L_ref` is a separate explicit coordinate-origin convention, not a requirement for nondimensionalization. In particular, relative geometry cancels any common origin shift. Numerical per-mesh bounding-box centering remains a distinct geometry normalization operation.

## 5. Reynolds, Mach, and other regime descriptors

Dimensionless regime numbers characterize the physical problem; they are not the primary dimensional normalization scales.

For example:

$$
Re_{\mathrm{ref}}=
\frac{\rho_{\mathrm{ref}}U_{\mathrm{ref}}L_{\mathrm{ref}}}
{\mu_{\mathrm{ref}}},
$$

and:

$$
Ma_{\mathrm{ref}}=
\frac{U_{\mathrm{ref}}}{a_{\mathrm{ref}}}.
$$

The definition of `a_ref` follows the thermodynamic model of the case. An ideal-gas expression such as `sqrt(gamma R T_ref)` must not be assumed for a case whose equation of state does not justify it.

Other dimensionless controls such as `Pr`, `Re_tau`, `Re_lambda`, forcing parameters, or chemistry parameters may be retained when relevant and consistently defined.

These values may later be model-visible global conditioning. They do not replace the reference-state system because the same `Re` or `Ma` can arise from many different dimensional reference states.

## 6. Preserve the physical regime

Nondimensionalization removes units; it must not silently remove the physical regime.

For example, using:

$$
\mu^*=\frac{\mu}{\rho_{\mathrm{ref}}U_{\mathrm{ref}}L_{\mathrm{ref}}}
$$

preserves Reynolds-number information when `mu` is the relevant dynamic viscosity. Independently dividing every constant-viscosity case by its own viscosity would instead force `mu*=1` and erase that distinction.

The same principle applies to other quantities: a scale is chosen from a coherent physical reference scheme, not independently per field merely to make magnitudes close to one.

## 7. Low-Mach energy edge case

With convective scaling, compressible total/internal energy may be numerically large at low Mach number because thermodynamic energy scales approximately with `Ma^-2` relative to `U_ref^2`.

This is not by itself evidence that the physical nondimensionalization is wrong.

M3.3 must not change the energy reference scale merely to force values near unity. Numerical convenience belongs to the later statistical-scaling stage, whose statistics are fitted only on training data.

## 8. Reference scope and anti-leakage rule

Reference quantities are case- or operating-condition-level by default.

They must be available from information legitimately known for the task at inference time. M3.3 therefore rejects silent use of quantities such as:

- target-snapshot maxima or minima;
- target-snapshot mean or standard deviation chosen only for convenience;
- instantaneous target RMS velocity;
- future-state information;
- validation/test statistics;
- hidden DNS quantities unavailable to the deployed model.

Snapshot-dependent physical references are allowed only if snapshot-wise invariance is itself an explicit scientific objective and the required quantity is legitimately available at inference.

The baseline runtime implementation is intentionally stricter: `ConvectiveNondimensionalizer` rejects snapshot-scoped references. A future task that truly requires snapshot-level reference semantics must introduce and test that exception explicitly.

## 9. Reference metadata contract

Every physical reference used by preprocessing must carry enough information to reconstruct its meaning. Runtime `ReferenceScale` supports:

```text
name
value
units
physical definition
provenance
scope
inference availability
optional derivation rule
```

`ReferenceScales.scheme` names the declared reference scheme for the collection. Physical preprocessing requires a non-empty scheme and requires units, provenance, and `inference_available=true` for every reference it actually uses.

The low-level data contract permits partially populated references so source readers can represent information before a complete preprocessing specification is available. Such references cannot be used by the baseline physical transform until the required semantics are present.

A missing required reference is an error. The implementation does not silently substitute `1.0`, infer a semantic definition from units, or switch reference definitions between cases.

## 10. Authoritative case-definition documents

The baseline project convention is now:

$$
\boxed{\text{reference quantities are declared, not inferred}}
$$

The simulation author provides one YAML case-definition document containing the reference values known from the simulation setup. The framework does not derive those values from an instantaneous solution snapshot.

The current schema is:

```yaml
case_id: HIT_LES_FORCED
reference_scheme: hit_forcing_reference

references:
  rho_ref:
    value: 1.2
    units: kg/m^3
    definition: prescribed_reference_density
    provenance: simulation_setup
    inference_available: true

  U_ref:
    value: 69.44
    units: m/s
    definition: prescribed_turbulent_velocity_scale
    provenance: simulation_setup
    inference_available: true
    scope: case
    derivation: Ma_ref * sqrt(gamma * R * T_ref)
```

`value` is authoritative and must be an explicit numeric literal. `derivation` is provenance/documentation only: the loader does not evaluate it and does not implement a physical-expression engine. This is the selected option for derived reference quantities. If `U_ref` was obtained from a prescribed Mach number and thermodynamic state, the simulation author stores the resulting numerical value and records the equation used in `derivation`.

Each reference requires `value`, `units`, `definition`, `provenance`, and explicit `inference_available`. `scope` is optional and defaults to `case`; `derivation` is optional. Unknown keys fail so misspelled scientific metadata cannot silently disappear.

`CaseDefinition` loads and validates this file with OmegaConf without resolving reference-value interpolation. `CaseDefinition.source_path` records which document supplied the reference state.

Dimensionless regime descriptors such as `Re`, `Ma`, and `Pr` remain conceptually separate from `ReferenceScale`. Their persisted runtime contract and model-conditioning interface are still deferred rather than mixing them into reference scales prematurely.

## 11. AVBP case attachment

`AVBPHDF5Dataset` accepts a `case_files` mapping from `case_id` to the authoritative case-definition file. Individual `AVBPSampleSpec` entries may carry `case_id` independently of `mesh_id`.

This supports, for example, one mesh reused across several physical operating conditions:

```text
mesh_shared + case_A
mesh_shared + case_B
mesh_shared + case_C
```

Case definitions are loaded once at dataset construction. Samples sharing a `case_id` reuse the same immutable `ReferenceScales` object. The returned `Sample.case_id` is explicit and `Sample.reference_scales` receives the declared case references. The case-definition path is retained as sample provenance.

A sample without `case_id` remains valid for raw data inspection and carries empty reference scales. A sample that declares `case_id` but has no configured case file fails at dataset construction.

## 12. Invertibility

Physical nondimensionalization must be invertible up to documented floating-point tolerance.

For example:

$$
\rho=\rho^*\rho_{\mathrm{ref}},
$$

$$
\rho u_i=(\rho u_i)^*\rho_{\mathrm{ref}}U_{\mathrm{ref}},
$$

$$
\rho E=(\rho E)^*\rho_{\mathrm{ref}}U_{\mathrm{ref}}^2.
$$

Round-trip tests are mandatory for every implemented transformation.

## 13. Runtime field implementation

`graph_attention.data.ConvectiveNondimensionalizer` implements the multiplicative baseline transformations for the currently supported canonical physical fields:

```text
rho
rhou
rhov
rhow
rhoE
pressure
temperature
```

The transform consumes an explicit `ReferenceScales` object and a mapping of named tensors. It returns a new mapping and does not mutate the supplied tensors or infer a transform for unknown names.

References are required only when a requested field actually depends on them. For example, transforming `rho` requires `rho_ref` but does not require an irrelevant `T_ref`.

The inverse `dimensionalize` operation uses the same field scales. Numerical tests cover exact known scales and floating-point round trips.

Coordinate nondimensionalization is not yet implemented in runtime code even though its baseline scientific definition is now `coords / L_ref`.

## 14. Statistical scaling remains separate

After physical nondimensionalization, an optional training-only scaler will use statistics such as:

$$
\hat q=\frac{q^*-\mu_{q^*,\mathrm{train}}}{\sigma_{q^*,\mathrm{train}}}.
$$

Those statistics must be learned from the training split only and frozen for validation, testing, and inference.

Statistical scaling is intentionally not implemented in this M3.3 step because the repository does not yet have the task/split interfaces that define which named fields/components belong to the learned representation and which samples constitute the training split. Implementing a fitted scaler before those contracts would either fit the wrong population or invent premature interfaces. It must be implemented before meaningful M6 training, after the M5 task and split semantics exist.

M3.3 does not blur reference-state provenance with learned statistical-scaler provenance.

## 15. Assumptions, edge cases, and deferred decisions

### Frozen assumptions

- The initial M3.3 equations target compressible Navier-Stokes-like state variables while allowing irrelevant references to be absent for other governing systems.
- Reference scales are case- or operating-condition-level in the baseline runtime transform.
- Reference values used by the framework are authoritative values supplied by the simulation author rather than inferred from target snapshots.
- The baseline uses a coherent convective basis rather than independent per-field magnitude normalization.
- `Re`, `Ma`, and similar dimensionless numbers are physical-regime descriptors rather than replacements for dimensional reference scales.
- Concrete definitions of `U_ref` and `L_ref` may differ between explicitly declared flow-family reference schemes.
- Reference values and field values are assumed to use a mutually coherent unit system; M3.3 records unit strings but does not perform unit conversion.

### Explicit failure behavior

- Missing required references fail.
- A missing reference scheme fails.
- A declared `case_id` without a configured case file fails.
- Case-file identity mismatches and unknown schema keys fail.
- Reference values that are not explicit numeric literals fail.
- Non-positive references used as multiplicative physical scales fail.
- Used references without units or provenance fail.
- References not explicitly marked available at inference fail.
- Snapshot-scoped references fail in the baseline transform.
- Unsupported field names fail rather than silently passing through unchanged.
- Non-floating physical tensors fail rather than being implicitly cast.
- A transformation is not inferred from a field's tensor shape, numerical range, or units alone.

### Deferred

- Runtime coordinate nondimensionalization using `coords / L_ref`.
- Optional coordinate-origin transforms, which remain separate from physical nondimensionalization.
- Persisted global regime descriptors and model-conditioning interfaces for `Re`, `Ma`, `Pr`, and related quantities.
- Statistical scaling implementation after task/split semantics exist.
- Specialized wall-unit representations such as `u_tau`, `y+`, or `Re_tau`-based quantities.
- Pressure-offset transforms; the baseline implements absolute pressure scaling only.
- Mapping of AVBP `vis_lam` and `vis_turb` until dynamic/kinematic and stored-unit semantics are confirmed.
- Unit conversion or dimensional-analysis enforcement between fields and references.
- Interpretation of AVBP `VertexData/volume` as a physical quadrature weight.
- Automatic import from AVBP setup/configuration files; this is no longer the authoritative path and would only be a future convenience importer.

## 16. Implementation gate

The baseline runtime field transform includes:

- explicit named reference semantics, scope, inference availability, and provenance hooks;
- field-specific forward transformations;
- inverse transformations;
- round-trip numerical reference tests;
- missing-reference and anti-leakage failure tests;
- no dependence on sample node count, node numbering, or mesh topology.

The case-definition integration adds:

- authoritative explicit YAML reference values;
- literal-value validation with non-evaluated derivation metadata;
- explicit case identity distinct from mesh identity;
- one-time case-file loading and reuse across snapshots;
- attachment to `Sample.reference_scales`;
- failure tests for missing or mismatched case definitions.

Target-environment `pytest`, Ruff, and format validation remain required for the new case-definition integration before that portion is marked validated.
