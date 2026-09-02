# Numerical Conventions

## 1. Purpose

This document defines numerical conventions that affect scientific meaning, reproducibility, stability, batching, and loss construction.

Numerically meaningful changes must update this document when they alter one of the conventions below.

## 2. Tensor conventions

Unless a task explicitly specifies otherwise:

- node physical state for one graph: `[N, C]`;
- coordinates: `[N, D]`;
- sparse connectivity: `[2, E]` integer edge indices;
- node weights: `[N]` or `[N, 1]` with explicit convention;
- packed node state: `[N_total, C]`;
- packed connectivity: `[2, E_total]`;
- packed sample membership: `batch_index [N_total]`;
- packed graph boundaries: `ptr [B+1]`.

Channel order must be explicit through named field metadata.

## 3. Packed graph convention

Graphs in one microbatch are represented as a disconnected union.

For graph `g`, local node indices are offset when concatenated into the packed graph. No cross-sample edges are created.

Processing disconnected packed graphs should be numerically equivalent, within documented tolerance, to processing the same graphs independently when the architecture contains no intentional batch-coupling operation.

## 4. Computational budgets

Variable-mesh training uses explicit per-device microbatch budgets.

At minimum support hard constraints of the form:

$$
N_{\mathrm{total}}\le N_{\max},
\qquad
E_{\mathrm{total}}\le E_{\max}.
$$

The concrete values are hardware/model configuration and must be recorded with benchmarks and runs.

A fixed graph count is not an adequate proxy for computational load when mesh sizes vary significantly.

If one graph alone exceeds a configured hard budget, fail explicitly rather than relying on a GPU OOM as control flow.

## 5. Statistical optimizer batch

Computational microbatching must not define statistical weighting implicitly.

A target optimizer batch may be defined by a number or weight of independent physical samples. Multiple microbatches may be accumulated before `optimizer.step()`.

Let each physical sample `g` have scalar sample loss `L_g` and statistical weight `s_g`. The intended optimizer-level objective is conceptually:

$$
L_{\mathrm{opt}}=
\frac{\sum_g s_g L_g}{\sum_g s_g}.
$$

The implementation must preserve this objective regardless of how those samples are partitioned across microbatches or DDP ranks.

## 6. Per-sample spatial loss reduction

A fine mesh must not receive greater statistical weight solely because it has more nodes.

### 6.1 Unweighted fallback

If no scientifically justified quadrature weights are available:

$$
L_g=\frac{1}{N_g}\sum_{i=1}^{N_g}\ell_{gi}.
$$

### 6.2 Physically weighted reduction

Where control-volume, quadrature, area, or other physical integration weights `w_i` are available and appropriate:

$$
L_g=
\frac{\sum_i w_i\ell_{gi}}{\sum_i w_i}.
$$

For wall quantities, weights may represent surface area rather than volume.

The meaning, units, source, and normalization of `w_i` must be documented.

## 7. DDP loss weighting

Do not naïvely average rank-local mean losses when DDP ranks contain different sample counts or sample weights.

The effective global objective must correspond to:

$$
L=
\frac{\sum_r\sum_{g\in r}s_gL_g}
{\sum_r\sum_{g\in r}s_g}.
$$

Implementation details may use scaled local losses or explicit distributed reductions, but the resulting gradient must represent the intended global statistical objective within numerical tolerance.

## 8. Physical nondimensionalization

Nondimensionalization occurs before statistical scaling.

A preprocessing specification contains:

1. reference-scale definitions;
2. numerical reference values or derivation rules per case;
3. named field transformations;
4. derived dimensionless regime variables;
5. training-set statistical scalers.

Reference definitions are semantic. Examples include `bulk_velocity`, `outer_velocity`, `channel_half_height`, `energy_injection_length`, etc.

Two references with the same unit are not interchangeable when their physical definitions differ.

## 9. Reference examples

Potential case-level scales include:

$$
U_{\mathrm{ref}},
L_{\mathrm{ref}},
\rho_{\mathrm{ref}},
T_{\mathrm{ref}},
p_{\mathrm{ref}}.
$$

Common transformations may include:

$$
\mathbf u^*=\frac{\mathbf u}{U_{\mathrm{ref}}},
\quad
\mathbf x^*=\frac{\mathbf x}{L_{\mathrm{ref}}},
\quad
\rho^*=\frac{\rho}{\rho_{\mathrm{ref}}},
$$

$$
(\rho\mathbf u)^*=\frac{\rho\mathbf u}{\rho_{\mathrm{ref}}U_{\mathrm{ref}}},
$$

$$
(\rho E)^*=\frac{\rho E}{\rho_{\mathrm{ref}}U_{\mathrm{ref}}^2},
$$

$$
p'^*=\frac{p-p_{\mathrm{ref}}}{\rho_{\mathrm{ref}}U_{\mathrm{ref}}^2}.
$$

These equations are examples, not permission to infer transformations automatically. Each supported field must have an explicit transformation contract.

## 10. Statistical scaling

After physical nondimensionalization, an optional training-set scaler may be applied:

$$
\tilde q=
\frac{q^*-\mu_{q^*,\mathrm{train}}}
{\sigma_{q^*,\mathrm{train}}}.
$$

Requirements:

- statistics are computed only from training data;
- statistics remain fixed for validation/test/inference;
- statistics are stored by named field/component;
- the associated nondimensional transformation is recorded;
- near-zero variance handling uses an explicit documented epsilon/policy;
- no per-snapshot recomputation unless scientifically specified.

## 11. Reference leakage prohibition

A reference or conditioning variable must be available at inference for the task.

Examples of prohibited behavior include:

- using target DNS wall stress to normalize a wall-stress predictor when wall stress is unavailable at deployment;
- computing a scale from a future state for a forecasting task;
- using test-set statistics to fit a scaler.

## 12. Resolution descriptors

Resolution is distinct from physical regime.

When required, define nondimensional resolution descriptors such as:

$$
\Delta_i^*=\frac{\Delta_i}{L_{\mathrm{ref}}}.
$$

`Delta_i` may be defined from:

- local edge lengths;
- directional mesh spacing;
- control-volume scale such as `V_i^(1/3)` in 3D;
- mesh metric tensors;
- another explicitly defined quantity.

No universal definition is assumed in M0.

## 13. Geometry normalization

Coordinate handling must be scientifically explicit.

Physical nondimensionalization such as `x/L_ref` is distinct from purely numerical coordinate centering/scaling.

Any additional coordinate transformation must state whether it preserves or removes:

- translation information;
- absolute location information;
- aspect ratio;
- physical scale.

Per-mesh coordinate transformations must not be conflated with per-mesh physical-field normalization.

## 14. Precision

Default numerical precision policies are not frozen by M0 beyond these requirements:

- precision mode must be explicit in resolved configuration;
- AMP/BF16/FP16 behavior must be covered by numerical tests for sensitive operations;
- mathematically sensitive sparse reductions may use higher-precision accumulation when required;
- changes in precision that materially change scientific results are numerically meaningful changes;
- benchmark results must include dtype/precision mode.

## 15. Numerical tolerances

Tests involving floating-point equivalence must use tolerances justified by:

- dtype;
- operation order;
- sparse reduction behavior;
- device/backend.

Do not use exact equality for floating-point graph aggregation unless exact behavior is guaranteed.

Permutation/edge-order tests may show small differences from reduction ordering; tolerance must be strict enough to catch real ordering dependence without rejecting expected floating-point noise.

## 16. Randomness

Random graph augmentation, sampling, masking, diffusion noise, and other stochastic mechanisms must have explicit seed handling.

Static topology augmentation intended to remain fixed for a mesh must be reproducible from recorded seeds/metadata and must not depend on arbitrary raw node numbering in a way that breaks node-renumbering equivariance.

## 17. Missing and invalid values

NaN/Inf handling must be explicit.

Do not silently replace invalid physical values with zero unless that behavior is scientifically justified, documented, and tested.

Dataset validation should fail early on unexpected invalid values in required fields.