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

## 18. M6 sample-balanced scaling and optimizer reduction

For M6, statistical scaling across variable-size meshes uses equal total weight per physical training sample. For channel `c`:

$$
\mu_c=\frac{1}{G}\sum_g\frac{1}{N_g}\sum_i q^*_{gic},
$$

$$
\sigma_c^2=\frac{1}{G}\sum_g\frac{1}{N_g}\sum_i(q^*_{gic}-\mu_c)^2.
$$

A global node-wise estimator is not the default because it would give larger meshes more influence solely through node count. Fitting is train-only, uses float64 online moment accumulation, and fails when `sigma_c <= 1e-12` rather than silently clamping a constant channel.

The M6 deterministic-regression pointwise loss is channel-mean squared error. Spatial reduction occurs independently within every physical sample, using `Mesh.node_weights` when supplied and an unweighted node mean otherwise. Physical samples then receive equal optimizer-level weight.

For a global optimizer step containing `G_global` samples on `R` DDP ranks, each rank backpropagates its local **sum** of sample losses scaled by:

$$
\alpha=\frac{R}{G_{\mathrm{global}}}.
$$

This compensates for PyTorch DDP's gradient averaging and yields the gradient of the global equal-sample mean even when ranks contain different sample counts. Rank-local computational microbatches use `DDP.no_sync()` except for the last local backward so unequal microbatch counts do not redefine the objective.

Under autocast, prediction and target dtypes are explicitly promoted for loss arithmetic. CUDA FP16 requires an explicit `GradScaler` in the M6 reference step. The default repository precision remains `32-true`; CUDA/NCCL AMP behavior is not yet a target-validated performance claim.

## 19. M8 sparse-attention numerical reductions

M8 interprets `edge_index[0]` as source nodes and `edge_index[1]` as target nodes. Attention normalization is performed independently for every target node and head over the supplied incoming edges.

The sparse softmax uses the standard stabilized form:

$$
\alpha_e=
\frac{\exp(s_e-m_{t(e)})}
{\sum_{e':t(e')=t(e)}\exp(s_{e'}-m_{t(e)})},
$$

where

$$
m_i=\max_{e:t(e)=i}s_e.
$$

For FP32/FP64 execution, score and normalization reductions use the active model dtype. For projected FP16 or BF16 query/key tensors, M8 promotes the edge score multiplication, score summation, exponentiation, max reduction, and denominator accumulation to FP32. Normalized weights are then cast to the value dtype before message multiplication and target accumulation.

This policy reduces overflow/underflow risk in the softmax normalization but creates FP32 temporary score/normalizer storage under low-precision execution. CUDA BF16/FP16 accuracy, throughput, and memory remain unvalidated until target measurements are collected.

Sparse target accumulation uses `scatter_add_` / `index_add_`. Mathematical output is independent of edge-list ordering, but floating-point summation order may differ across reordered edge lists or GPU atomic scheduling. Edge-order and node-renumbering tests therefore use strict numerical tolerances rather than requiring bitwise equality.

Nodes with no incoming edges have a zero attention message before the residual branch. A globally empty edge list returns zero from the sparse attention sublayer without evaluating a softmax over an empty set.

## 20. M9 relative-displacement geometry and score bias

M9 computes one relative displacement row per directed edge using the already task-prepared coordinates:

$$
\Delta r_{ij}=r_j-r_i
$$

for source `j` and target/query `i`. The displacement is computed once per model forward and reused across all geometric attention layers.

When physical nondimensionalization is enabled, the coordinate tensor has already been scaled by the case length reference, so no additional geometric normalization is applied:

$$
\Delta r^*_{ij}=\frac{r_j-r_i}{L_{\mathrm{ref}}}.
$$

M9 does not statistically standardize coordinates or relative displacement. It also does not append an explicitly computed Euclidean norm. Any later explicit distance feature is a separate numerical/scientific convention and must be benchmarked/ablated independently.

Each geometric attention layer maps displacement through a small MLP to one score bias per head. The geometric bias is added before the existing stabilized M8 target-wise softmax. Under FP16/BF16 projection/autocast, the geometric bias is converted to the same FP32 score dtype used by the M8 softmax before addition.

The exact geometry computation is translation invariant in real arithmetic. Floating-point subtraction after a large common translation may exhibit roundoff/cancellation; translation-property tests therefore use strict numerical tolerances rather than bitwise equality.

Coordinate tensors must be finite floating-point values with the configured spatial dimension and must share dtype/device with model inputs. Invalid geometry fails explicitly rather than being cast, centered, clipped, or repaired.

The per-layer geometry MLP hidden width equals `num_heads`, so geometry-specific edge activations scale as `O(E * num_heads)`. The existing M8 value/message path remains `O(E * hidden_dim)` and is still expected to dominate sparse activation memory for typical `hidden_dim >> num_heads` configurations.

## 21. M11 straight-path flow matching

M11 constructs its Gaussian source only after physical nondimensionalization and train-only statistical standardization. The data endpoint `x_1` therefore has the frozen training-set standardized channel semantics and the source is sampled in that same coordinate system:

$$
x_0\sim\mathcal N(0,I).
$$

One floating-point time is sampled per physical graph,

$$
t_g\sim\mathcal U(0,1),
$$

then broadcast with `batch_index` to all nodes of graph `g`. The numerical path and target are

$$
x_t=(1-t)x_0+t x_1,
\qquad
v^*=x_1-x_0.
$$

The velocity target is not statistically transformed again after path construction. The existing sample-reduced equal-channel MSE is applied directly in the standardized state coordinate system.

Training path randomness uses a generator separate from model initialization. Validation time/noise and sampling sources are seeded from stable hashes of the configured seed and `sample_id`; deterministic validation is therefore independent of validation batch order. The exact random-number stream is backend/runtime dependent, so cross-PyTorch-version or cross-device bitwise identity is not claimed.

The first time representation is the raw scalar `t` appended to graph-level conditioning. It is not multiplied by an arbitrary diffusion timestep scale and no sinusoidal/Fourier transformation is applied in M11.

Inference integrates

$$
\frac{dx}{dt}=v_\theta(x,t)
$$

from `t=0` to `t=1` on a uniform grid. Explicit Euler uses one model evaluation per step. Heun uses a predictor and endpoint correction and therefore uses two model evaluations per step. The reference configuration uses 50 Heun steps. Adaptive-step error control is not part of the M11 baseline.

Generated standardized states are inverse-transformed with the frozen **input/state** training standardizer to the physically nondimensional state. No test-set statistics participate in this inverse transform.
