# M3.3 Physical Nondimensionalization Directive

## Status

The scientific definition for M3.3 is frozen by this document. Runtime preprocessing code and numerical reference tests remain to be implemented.

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

## 9. Reference metadata contract

Every physical reference used by preprocessing must carry enough information to reconstruct its meaning. Conceptually this includes:

```text
name
value
units
physical definition
provenance
scope
optional derivation rule
```

The existing M2 `ReferenceScale` runtime contract records only part of this information. M3.3 implementation must extend or accompany that contract minimally rather than hiding missing semantics in generic metadata.

A missing required reference is an error. The implementation must not silently substitute `1.0`, infer a semantic definition from units, or switch reference definitions between cases.

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

## 11. Statistical scaling remains separate

After physical nondimensionalization, an optional training-only scaler may later use statistics such as:

$$
\hat q=\frac{q^*-\mu_{q^*,\mathrm{train}}}{\sigma_{q^*,\mathrm{train}}}.
$$

Those statistics are learned from the training split only and frozen for validation, testing, and inference.

M3.3 must not blur reference-state provenance with learned statistical-scaler provenance.

## 12. Assumptions, edge cases, and deferred decisions

### Frozen assumptions

- The initial M3.3 equations target compressible Navier-Stokes-like state variables while allowing irrelevant references to be absent for other governing systems.
- Reference scales are case-level by default.
- The baseline uses a coherent convective basis rather than independent per-field magnitude normalization.
- `Re`, `Ma`, and similar dimensionless numbers are physical-regime descriptors rather than replacements for dimensional reference scales.
- Concrete definitions of `U_ref` and `L_ref` may differ between explicitly declared flow-family reference schemes.

### Explicit failure behavior

- Missing required references fail.
- Incompatible or semantically undefined reference schemes fail.
- A transformation is not inferred from a field's tensor shape or units alone.
- Target-dependent normalization unavailable at inference is rejected.

### Deferred

- The exact runtime representation/API for reference schemes.
- Automatic extraction of case references from AVBP configuration files.
- Global model-conditioning interfaces for `Re`, `Ma`, `Pr`, and related quantities.
- Statistical scaling implementation.
- Specialized wall-unit representations such as `u_tau`, `y+`, or `Re_tau`-based quantities.
- Interpretation of AVBP `VertexData/volume` as a physical quadrature weight.

## 13. Implementation gate

Before M3.3 implementation is considered complete, it must include at least:

- explicit named reference semantics and provenance;
- field-specific forward transformations;
- inverse transformations;
- round-trip numerical reference tests;
- missing-reference and anti-leakage failure tests;
- no dependence on sample node count, node numbering, or mesh topology;
- documentation and traceability updates for any deviation from this directive.
