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

## 22. M13 discrete diffusion and generalized DDIM sampling

M13 constructs the discrete cosine diffusion schedule once in float64. The un-clipped cumulative schedule is normalized so the clean endpoint is one, then discrete betas are derived and clipped to

```text
1e-8 <= beta_t <= 0.999
```

before recomputing the cumulative product used by training and sampling. The stored convention has `alpha_bar[0] = 1` and training timesteps in `t = 1, ..., T`. Schedule tensors are cast and cached on the active state device/dtype when used; the baseline target path is FP32 and CUDA low-precision behavior remains unvalidated.

One integer timestep is sampled per physical graph and broadcast to its nodes through `batch_index`. The model-visible time is

$$
\tau=\frac{t}{T}\in(0,1],
$$

rather than the raw integer timestep. This keeps scalar-conditioning magnitude comparable to the existing flow-matching baseline while preserving the discrete schedule internally.

Training noising is performed in the already standardized state space:

$$
x_t=\sqrt{\bar\alpha_t}x_0+
\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

The sampled epsilon is the model target and is not statistically transformed again. The existing per-sample, equal-channel MSE reduction is used unchanged.

Validation randomness is derived independently for each physical sample from a stable BLAKE2b hash of `(validation_seed, "validation", sample_id)`. Each validation sample therefore receives a deterministic timestep and Gaussian field independent of validation batch order. Training diffusion randomness uses a dedicated PyTorch generator seeded separately from model initialization. Stable hashing fixes seed construction, but bitwise equality across PyTorch versions, devices, or different RNG backends is not claimed.

Reverse generation starts from an independent Gaussian source for each generated sampling key. The baseline generation runner uses keys `gen_000000`, `gen_000001`, and so on rather than held-out CFD filenames. For stochastic sampling, the same per-generated-sample generator supplies all later reverse-process noise draws. The clean held-out test state is never used to construct the generated state or its RNG stream.

For `sampling.steps = K`, reverse timesteps are constructed by a linearly spaced grid from `T` to `1`, rounded to integer timesteps. Duplicate rounded timesteps are rejected. `K` must satisfy `1 <= K <= T`. Each selected reverse timestep uses one network evaluation, so the model-evaluation count is `K`.

For consecutive selected times `t > t'`, the generalized DDIM variance is

$$
\sigma_t=\eta
\sqrt{
\frac{1-\bar\alpha_{t'}}{1-\bar\alpha_t}
\left(1-\frac{\bar\alpha_t}{\bar\alpha_{t'}}\right)
},
$$

with `0 <= eta <= 1`, and the direction coefficient uses

$$
\sqrt{\max(1-\bar\alpha_{t'}-\sigma_t^2,0)}.
$$

The explicit clamp to zero guards floating-point roundoff in a theoretically non-negative expression; it is not a change to the intended sampler. `eta=0` is deterministic DDIM. `eta=1` with `K<T` remains stochastic accelerated DDIM. Only `eta=1` with `K=T` is labelled `ddpm_ancestral`.

No x0 clipping, dynamic thresholding, learned reverse variance, SNR loss weighting, or alternate prediction parameterization is applied in M13. Introducing any of those changes the numerical/scientific baseline and requires separate validation.

Training and generation use separate scripts. Generation reloads the exact persisted `standardizers.pt` and `dataset_split_manifest.json`; it does not refit statistics or regenerate the split. Generated standardized states are inverse-transformed using the frozen **input/state** standardizer. The generation artifact stores generated RNG IDs and reference test IDs separately, and benchmark comparison is population-level rather than index-paired.


## M46 all-snapshot similarity numerical conventions

- All available snapshots are included. The physical sample is one snapshot of one case;
  each case has its own empirical distribution and no node-count importance factor.
- On endpoint-inclusive Cartesian periodic meshes, the three maximum-coordinate planes
  are omitted from moments and FFTs. The native-node-to-Cartesian map is verified
  exhaustively and is independent of original node IDs.
- Full-grid moments use population conventions, with Pébay chunk aggregation.
  Velocity TKE uses volume-mean-subtracted \`u_i = (rho*u_i)/rho\`, divided by
  \`U_ref\`, without Favre weighting. Any sound-speed Mach diagnostic requires an explicit
  \`a_ref\` in the case reference YAML.
- Full-cube normalized real Fourier coefficients and Hermitian mirror weights are used
  for Helmholtz energy fractions. Plotted \`E*(kL)\` uses nearest-integer radial shells,
  shell spacing \`2*pi\`, and the inscribed Nyquist sphere. Parseval tests compare
  total spectral fluctuation energy with spatial TKE; the plotted spectrum reports
  its energy capture separately.
- Distribution draws sample a fixed number of Cartesian spatial locations per snapshot
  with a reproducible case/snapshot seed, rather than choosing native node IDs.
  Pairwise W1 is the equal-quantile 1024-midpoint approximation; dividing by the
  pooled full-field standard deviation produces a dimensionless normalization
  separate from M29's frozen-training-scaler standardized W1.
- Cached snapshots are invalidated by changes in options, reference checksum or source
  file size/mtime. Archive/pilot collisions are resolved by ID/counter with archived
  data preferred; identity is a provenance assumption rather than a hash proof.


## M48.4b: Frozen teacher-only CEDAR residual scaling (2026-11-18)

For the fixed MS_SR \`w1p25_l1\` q8 -> q4 task, the
\`configs/m48/teacher_44_w1p25_l1_q4_q8.json\` file freezes
the **all-44-teacher** scalar moments from audit job 510184.
The 22 cases contribute two teacher snapshots each. The intended weight
is one per independent snapshot, not one per node. The selection hash
authenticates the ordered list of teacher source and operator identities;
no validation/test fields enter any fit.

For each conservative field component k:

\`\`\`text
coarse_input_k = (Uc_k - mean_teacher(Uc_k)) / std_teacher(Uc_k)
lifted_input_k = (P Uc_k - mean_teacher(Uc_k)) / std_teacher(Uc_k)
FM_target_k    = (Uf_k - P Uc_k) / RMS_teacher(Uf_k - P Uc_k)
X0_k           = (I - P C) eps_k
Xt_k           = (1 - t) X0_k + t FM_target_k
v_target_k     = FM_target_k - X0_k
\`\`\`

The fixed componentwise **residual RMS** is strictly positive and
does not subtract a mean, preserving the global nullspace
\`C R = 0\` for an operator that acts identically on each component.
All physical restriction, lifting and source-noise projection are
on complete periodic fields. The training loss reduces MSE over owned
fine nodes in each patch, and training schedules each teacher snapshot
equally in each epoch, independent of 48^3 vs 80^3 fine-node count.

The same frozen normalization, seeds, FM-time schedule, patch origins,
optimizer and nominal model width/layer depths are used for static and
dynamic TwoScaleDiNAT. Their parameter counts and dependency halos are
not equal; they run as separate jobs so allocated CUDA memory can be
compared without a second resident model. This is a
**matched-configuration** comparison, not a parameter-matched ablation.
No new physical nondimensionalization is applied in this baseline,
and the archive's case names are not trusted as measured Re/Ma values.

This changes the numerical preprocessing **relative to the M48.4a
two-source engineering smoke**, which remains available for
reproducibility. The 44-source frozen scales are required for
M48.4b. The new training path is source-prepared and awaiting its
Calypso qualification, not yet a measured convergence result.


## M48.4c: configuration-C statistical GPU patch batching (2026-11-18)

This extension leaves the M48.4b frozen 44-teacher per-component
normalization, full-field CEDAR inverse and physics unchanged.
For the q8->q4 task, independent source fields and their statistical
weights are unchanged by GPU microbatching.

New **explicit** training sampling parameters:

    owned fine cube side = 12
    patches per teacher source per epoch = 16
    effective optimizer batch = 4 patches
    packed GPU microbatch = 1, 2, or 4 patches

Each patch independently draws a full-domain CEDAR-projected standard
Gaussian noise `X0=(I-PC)epsilon`, independent FM time t and a periodic
owned-core origin. There is no patchwise CEDAR projection or artificial
periodic boundary. The task target is unchanged:
`Xt=(1-t)X0+tX1`, `velocity_target=X1-X0`, `X1=(Uf-PUc)/sigma_R`.

For each independent patch b, reduce MSE over its **owned fine nodes and
all five conservative components** to `L_b`. The per-optimizer objective
is `(L_1+L_2+L_3+L_4)/4`, independent of patch read-set/halo sizes.
For GPU microbatch m, pack m disconnected fine/coarse attention graphs
with distinct per-graph FM conditioning, and accumulate m-patch
subgradients scaled by 1/4. Clip gradients and apply AdamW **once**
after all four patches. This is a numerically meaningful change in
gradient variance and optimizer update semantics, not in the
expected spatial patch loss. GPU microbatch variations can produce
small floating point differences.

44 sources * 16 patches/source = 704 patches/epoch, 176
optimizer updates/epoch. Eight full epochs would consume 5,632 patch
samples and 1,408 optimizer updates per architecture. A bounded
`--probe-grids` run is not a full epoch: it deliberately selects one
authenticated teacher case on 48^3 and one on 80^3, and records a
distinct checkpoint schedule/fingerprint. Do not compare its loss to
a converged training run.

See `docs/M48_4C_SAMPLING_GPU_BENCHMARK.md` for the geometry, exact
packed-graph edges, numerical tests, memory/throughput measurements,
and operational gate. Single-GPU performance is unvalidated until
the Calypso benchmark jobs return.


## M48.4e — Phase-A isolated GPU capacity (2026-11-19)

M48.4e Phase A does **not** change any physical formula, training loss,
CEDAR operator, projected FM source, teacher data or frozen normalization.

It measures the actual static/dynamic TwoScaleDiNAT network
forward/backward and AdamW on **synthetic** five-channel patch features,
with exact M48 local/dilated/cross-scale graph supports and periodic
seam-crossing owned fine cores. The fixed graph template is replicated
as disconnected components with graph-specific synthetic FM times and
targets. It is a GPU computational capacity test, **not** training or
a test of globally projected Gaussian source noise. The original
44-teacher manifest and frozen normalization are authenticated using
metadata and hashes only; source field payloads are never opened.

Parameter sweep: fine grids 48^3 and 80^3, owned patch sides
12/16/20/24, CUDA microbatch 1/2/4/.../256, with production-candidate
width/depth explicitly tunable. This benchmarking batch size **is the
optimizer batch size for its synthetic step**; the scientific effective
batch is not thereby selected. A later supervised training optimizer
configuration is a separate statistical decision.

CUDA work includes actual forward, backward, gradient clipping and
AdamW; CUDA event and synchronized wall times are reported separately
from CPU geometry construction, packing and host-to-device transfer.
OOM trials execute in isolated processes and are excluded from
throughput ranking. Rank qualified GPU configurations by owned nodes
per update second within a GPU reserved-memory safety budget.

Only exact dependent read nodes enter network computations and only
owned fine nodes contribute to the synthetic MSE. A GPU run at one
patch size/geometry origin cannot prove equivalent capacity for all
random origins, CEDAR preprocessing efficiency, convergence or
resolution generalization. See
`docs/M48_4E_GPU_CAPACITY_PHASE_A.md` and
`scripts/m48_4e_gpu_capacity.py`.


## M48.4e A1 — Depth and halo geometry qualification (2026-11-19)

A1 is a **CPU-only measurement** of the existing exact M48 dependency closure,
with no changes to network, filtered fields, CEDAR, or periodic indexing.
The A1 depth `d` means `d` fine-pre, `d` fine-post, `d` coarse-static
blocks, plus `d` coarse-dynamic blocks in dynamic mode. Each sequence
alternates exact one-hop and exact-two-hop attention as already implemented.

Owned cores remain periodic on the **original full lattice**, and halo
boundaries are not given artificial periodic edges. Source metadata must pass
44-teacher manifest and normalization authentication; physical periodic
lengths are case-dependent, with one authenticated representative length
per grid used for the topology-only geometry audit.

Report explicit fine/coarse node and edge counts, read-to-owned ratio,
read fractions and layer-weighted **structural work proxies**. Idealized
full-coverage ratios do not constitute measured GPU execution speed or
training convergence. See `docs/M48_4E_A1_DEPTH_HALO_GEOMETRY.md`.
