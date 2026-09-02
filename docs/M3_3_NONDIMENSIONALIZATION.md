# M3.3 Physical Nondimensionalization Directive

## Status

The M3.3 scientific definition is frozen. The baseline runtime reference metadata and forward/inverse field transformations are implemented, pending target-environment validation. Automatic case-reference extraction, coordinate nondimensionalization, regime conditioning, and statistical scaling remain deferred.

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

Auxiliary fields use the same coherent basis when they are included:

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

Coordinates may use:

$$
\mathbf x^*=\frac{\mathbf x-\mathbf x_{\mathrm{origin}}}{L_{\mathrm{ref}}},
$$

where the origin convention is explicit. This physical operation is distinct from numerical per-mesh bounding-box centering or normalization.

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

preserves Reynolds-number information. Independently dividing every constant-viscosity case by its own viscosity would instead force `mu*=1` and erase that distinction.

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

`ReferenceScales.scheme` names the declared reference scheme for the collection. Physical preprocessing requires a non-empty scheme and requires units and provenance for every reference it actually uses.

The low-level data contract permits partially populated references so source readers can represent information before a complete preprocessing specification is available. Such references cannot be used by the baseline physical transform until the required semantics are present.

A missing required reference is an error. The implementation does not silently substitute `1.0`, infer a semantic definition from units, or switch reference definitions between cases.

## 10. Invertibility

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

## 11. Runtime implementation

`graph_attention.data.ConvectiveNondimensionalizer` implements the multiplicative baseline transformations for the currently supported canonical physical fields:

```text
rho
rhou
rhov
rhow
rhoE
pressure
temperature
vis_lam
vis_turb
```

The transform consumes an explicit `ReferenceScales` object and a mapping of named tensors. It returns a new mapping and does not mutate the supplied tensors or infer a transform for unknown names.

References are required only when a requested field actually depends on them. For example, transforming `rho` requires `rho_ref` but does not require an irrelevant `T_ref`.

The inverse `dimensionalize` operation uses the same field scales. Numerical tests cover exact known scales and floating-point round trips.

The runtime implementation deliberately does not nondimensionalize coordinates yet. Coordinate scaling requires an explicit `x_origin` convention in addition to `L_ref`, and that convention must not be guessed from a mesh bounding box.

## 12. Statistical scaling remains separate

After physical nondimensionalization, an optional training-only scaler may later use statistics such as:

$$
\hat q=\frac{q^*-\mu_{q^*,\mathrm{train}}}{\sigma_{q^*,\mathrm{train}}}.
$$

Those statistics are learned from the training split only and frozen for validation, testing, and inference.

M3.3 does not blur reference-state provenance with learned statistical-scaler provenance.

## 13. Assumptions, edge cases, and deferred decisions

### Frozen assumptions

- The initial M3.3 equations target compressible Navier-Stokes-like state variables while allowing irrelevant references to be absent for other governing systems.
- Reference scales are case- or operating-condition-level in the baseline runtime transform.
- The baseline uses a coherent convective basis rather than independent per-field magnitude normalization.
- `Re`, `Ma`, and similar dimensionless numbers are physical-regime descriptors rather than replacements for dimensional reference scales.
- Concrete definitions of `U_ref` and `L_ref` may differ between explicitly declared flow-family reference schemes.
- Reference values and field values are assumed to use a mutually coherent unit system; M3.3 records unit strings but does not perform unit conversion.

### Explicit failure behavior

- Missing required references fail.
- A missing reference scheme fails.
- Non-positive references used as multiplicative physical scales fail.
- Used references without units or provenance fail.
- References marked unavailable at inference fail.
- Snapshot-scoped references fail in the baseline implementation.
- Unsupported field names fail rather than silently passing through unchanged.
- Non-floating physical tensors fail rather than being implicitly cast.
- A transformation is not inferred from a field's tensor shape, numerical range, or units alone.

### Deferred

- Automatic extraction of case references from AVBP configuration files.
- Association/injection of reference schemes into `AVBPHDF5Dataset` samples.
- Physical coordinate nondimensionalization and its explicit origin convention.
- Global model-conditioning interfaces for `Re`, `Ma`, `Pr`, and related quantities.
- Statistical scaling implementation.
- Specialized wall-unit representations such as `u_tau`, `y+`, or `Re_tau`-based quantities.
- Pressure-offset transforms; the baseline implements absolute pressure scaling only.
- Unit conversion or dimensional-analysis enforcement between fields and references.
- Interpretation of AVBP `VertexData/volume` as a physical quadrature weight.

## 14. Implementation gate

The baseline runtime implementation includes:

- explicit named reference semantics, scope, inference availability, and provenance hooks;
- field-specific forward transformations;
- inverse transformations;
- round-trip numerical reference tests;
- missing-reference and anti-leakage failure tests;
- no dependence on sample node count, node numbering, or mesh topology.

Target-environment `pytest`, Ruff, and format validation remain required before the runtime portion is marked validated.
