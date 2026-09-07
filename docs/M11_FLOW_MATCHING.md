# M11 HIT-slice flow matching

## Status

M11 implements the first generative task in GraphAttention: unconditional straight-path flow matching of the complete five-channel conservative HIT slice state with the existing M8/M9 sparse graph Transformers.

Software and target-training validation are pending. No generative-quality claim is made until the implementation passes the repository gate and real GPU smoke training.

## 1. Scientific objective

For one 2-D spatial slice of the 3-D HIT state, define

$$
x = [\rho,\rho u,\rho v,\rho w,\rho E].
$$

M11 learns an unconditional vector field whose ODE transports a standard Gaussian source distribution toward the empirical distribution of the full five-channel state. This is a generative task, not the M10 cross-field regression task.

The baseline follows the flow-matching formulation of Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023, arXiv:2210.02747, using a project-specific straight interpolation path. Straight source-to-data trajectories are also closely related to rectified-flow practice; see Liu et al., *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*, ICLR 2023, arXiv:2209.03003.

The implementation is adapted from the existing `diffusion4avbp` linear Gaussian flow-matching task, but uses GraphAttention's packed graph, preprocessing, loss, provenance, and sparse-transformer contracts.

## 2. State preprocessing and source distribution

The data endpoint is constructed in this order:

$$
\boxed{
\text{dimensional conservative state}
\rightarrow
\text{physical nondimensionalization}
\rightarrow
\text{train-only sample-balanced standardization}
\rightarrow
x_1
}
$$

Only after statistical standardization is the Gaussian source sampled:

$$
x_0\sim\mathcal N(0,I).
$$

This ordering is deliberate. The prior therefore lives in the same numerically normalized state space in which the model is optimized. Noise is not sampled in dimensional CFD units and then normalized afterward.

The frozen M11 state fields are, in order:

```text
rho
rhou
rhov
rhow
rhoE
```

All five channels are generated jointly.

## 3. Straight flow-matching path

For each physical graph `g` independently,

$$
t_g\sim\mathcal U(0,1).
$$

The same time is broadcast to every node in that graph. For node `i` in graph `g`,

$$
\boxed{x_{t,gi}=(1-t_g)x_{0,gi}+t_gx_{1,gi}}
$$

with constant conditional target velocity

$$
\boxed{v^*_{gi}=x_{1,gi}-x_{0,gi}}.
$$

The model learns

$$
v_\theta(x_t,t,G,R),
$$

where `G` is the supplied sparse graph topology and `R` is the model geometry used by M9.

M11 does not add a diffusion beta schedule, score target, epsilon target, SNR weighting, or discrete diffusion timestep. Those belong to a later diffusion task.

## 4. Time conditioning

The first baseline uses the raw scalar

$$
t\in[0,1]
$$

as one graph-level conditioning channel. It is concatenated with any already-declared graph-level physical conditioning and is broadcast to nodes through the existing M8/M9 conditioning path.

No sinusoidal/Fourier time embedding, learned time MLP, AdaLN, or per-layer time modulation is added in M11. This is an explicit minimal baseline, not a claim that raw scalar time is optimal. Time representation is a separate future ablation.

Because M8 and M9 already support graph-level conditioning, their attention equations remain unchanged.

## 5. Geometry and topology

M11 reuses the validated M10 HIT-slice representation without alteration:

- canonical stored coordinates `[N,2]`;
- `spatial_dim=2` for M9;
- extraction axis retained as provenance only;
- non-periodic bidirectional Cartesian 4-neighbour topology;
- no diagonals, self-loops, random edges, k-hop edges, or periodic wrap augmentation.

For M9, geometry therefore remains the canonical 2-D relative displacement

$$
\Delta r_{ij}=r_j-r_i\in\mathbb R^2.
$$

## 6. Training loss

The flow-matching velocity target has the same five channels as the standardized state. The per-node equal-channel MSE is

$$
\ell_{gi}=\frac{1}{5}\|v_\theta(x_{t,gi},t_g)-v^*_{gi}\|_2^2.
$$

M11 reuses the M6 sample reduction. Without node quadrature weights,

$$
L_g=\frac{1}{N_g}\sum_i\ell_{gi},
$$

and the optimizer objective is

$$
\boxed{L=\frac{1}{B}\sum_g L_g}.
$$

The existing `train_equal_sample_optimizer_step` is reused after the flow-matching task has constructed `x_t`, `v*`, and time conditioning. Statistical standardization is not applied a second time to the velocity problem.

## 7. Reproducible training and validation randomness

Training uses a dedicated path RNG seeded independently of model initialization. This prevents the additional M9 geometry parameters from changing the sampled training path sequence.

Validation is deterministic by `(validation_seed, sample_id)`. Each validation sample receives a reproducible time and Gaussian source independent of validation batch order.

This makes validation flow-velocity MSE comparable across repeated runs and across M8/M9 when the split, seeds, and model initialization policy are matched.

## 8. ODE sampling

Generation starts from a deterministic per-sample Gaussian source in standardized state space and integrates

$$
\frac{dx}{dt}=v_\theta(x,t,G,R),\qquad t:0\rightarrow1.
$$

M11 supports:

- explicit Euler;
- explicit Heun predictor-corrector.

The first reference configuration uses Heun with 50 uniform steps. Solver type and step count are inference choices and do not change the training objective.

The generated standardized state is inverse-transformed only through the fitted training-set state standardizer, producing the physically nondimensional five-channel state. No dimensional reconstruction is required for the first analysis artifact.

## 9. First HIT experiment

The reference data/split remains the M10 real artifact:

```text
1488 slices
372 source snapshots
33 x 33 = 1089 nodes per slice
4224 directed 4-neighbour edges per slice
train / validation / test = 1040 / 220 / 228 slices
source groups = 260 / 55 / 57
```

The provisional single-GPU batch size is 128. This is an engineering starting point selected to move the generative experiment forward; it is not a frozen scientific constant or an efficiency optimum. On smaller GPUs it may be reduced without changing the task definition.

The reference architecture remains the M8/M9 128-hidden, 8-head, 4-layer, MLP-ratio-4 configuration unless explicitly overridden.

## 10. Reported outputs

Each run writes:

- resolved Hydra configuration;
- grouped split manifest and runtime provenance;
- frozen train-only standardizers;
- `history.csv` with train and deterministic validation flow-velocity MSE;
- `best.pt` and `last.pt`;
- `generated_test.pt` containing deterministic generated test states in standardized and nondimensional coordinates plus the nondimensional reference state;
- `summary.json`.

As a first distribution diagnostic, `summary.json` reports for each channel:

- generated marginal mean/std;
- test marginal mean/std;
- empirical one-dimensional Wasserstein-1 distance between all generated and test node values in that channel.

This flattened marginal W1 is only a channel-distribution diagnostic. It does not measure spatial coherence, spectra, cross-channel structure, or joint-distribution fidelity.

## 11. Assumptions, edge cases, and deferred work

Introduced or inherited assumptions:

- the M10 fixed-slice artifact and grouped split contracts remain valid;
- the five physically nondimensional fields are standardized using training data only;
- an isotropic standard Gaussian in statistically standardized state coordinates is an acceptable first source distribution;
- raw scalar time conditioning is sufficient for the first baseline;
- one time value applies to every node in one physical sample;
- canonical 2-D geometry and non-periodic 4-neighbour topology remain the first controlled representation.

Handled explicitly:

- variable graph node counts through packed `batch_index`/`ptr` semantics;
- one independently sampled training time per graph;
- validation random variables keyed by sample ID rather than batch order;
- sampling Gaussian sources keyed by sample ID;
- existing physical/regime conditioning concatenated before flow time;
- Euler/Heun solver validation;
- model output/state shape mismatch;
- non-empty sample IDs for deterministic validation/sampling;
- exact input/target state channel-order agreement;
- output-directory collision.

Deliberately deferred:

- sinusoidal/Fourier/learned time embeddings;
- AdaLN or per-layer time conditioning;
- nonlinear probability paths or optimal-transport couplings;
- diffusion/DDPM/DDIM tasks;
- periodic wrap topology;
- slice-orientation conditioning or local vector rotation;
- BF16/FP16 training validation;
- DDP in the dedicated first M11 runner;
- adaptive ODE solvers;
- spectral, spatial-correlation, and joint-distribution generation metrics;
- multi-seed generative-quality conclusions.

Failure behavior:

- invalid task state semantics fail before path construction;
- invalid sampling solver/step/seed values fail explicitly;
- model output shape mismatch fails during sampling;
- non-finite final generated state fails instead of being silently saved;
- missing or inconsistent data/split/scaler contracts continue to fail through the existing M3-M6 validation paths.

Remaining validation gaps:

1. full `pytest`, `ruff check .`, and `ruff format --check .` after merge;
2. real-artifact one-epoch M11 M9 GPU smoke;
3. confirm `conditioning_channels=1`, `in_channels=5`, `out_channels=5`, and M9 `spatial_dim=2` in the real run;
4. inspect generated-test tensor shapes and finite values;
5. choose a full training epoch/update budget from learning curves;
6. run M8 under the same seeds/settings for the geometry ablation;
7. add multiple seeds if the final architectural difference is small.

Trade-offs:

- raw scalar time is cheap and minimally confounded but may underrepresent time compared with richer embeddings;
- batch size 128 reduces optimizer updates per epoch relative to smaller batches, so epoch counts are not comparable across batch-size changes;
- Heun roughly doubles model evaluations relative to Euler for a fixed number of steps;
- flattened marginal W1 is inexpensive and interpretable per channel but ignores spatial and joint structure;
- the first runner stays single-process to minimize implementation risk, so four-A30 DDP utilization is deferred rather than silently changing distributed semantics.
