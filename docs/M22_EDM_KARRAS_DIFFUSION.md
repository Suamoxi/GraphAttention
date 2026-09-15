# M22 EDM/Karras Continuous-Noise Diffusion

## Purpose

M22 tests whether a continuous-noise EDM formulation removes the pathological terminal
transition observed with the M13/M17 cosine DDPM schedule while keeping the same HIT slice
data, preprocessing, M12 L10 backbone, optimizer family, and grouped split.

This is a **project adaptation of an established method**, based on Karras et al.,
*Elucidating the Design Space of Diffusion-Based Generative Models* (NeurIPS 2022,
arXiv:2206.00364) and the public NVlabs/edm implementation.

M22 is not a claim that EDM is superior until the target-cluster training and generation
benchmarks are complete.

## Scientific state

The generated standardized state is

\[
x_0=[\rho,\rho u,\rho v,\rho w,\rho E].
\]

Physical nondimensionalization and train-only sample-balanced statistical scaling are
unchanged from M13.  EDM perturbations are applied after those transforms.

For every physical graph, training samples

\[
\log\sigma \sim \mathcal N(P_{\rm mean},P_{\rm std}^2),
\qquad
\epsilon\sim\mathcal N(0,I),
\]

and constructs

\[
y=x_0+\sigma\epsilon.
\]

One `sigma` is shared by all nodes in one graph, while Gaussian noise remains node/channel
wise.

## EDM preconditioning

The raw M12 network output is denoted `F`.  The task uses the EDM coefficients

\[
c_{\rm skip}=\frac{\sigma_{\rm data}^2}{\sigma^2+\sigma_{\rm data}^2},
\]

\[
c_{\rm out}=\frac{\sigma\sigma_{\rm data}}
{\sqrt{\sigma^2+\sigma_{\rm data}^2}},
\qquad
c_{\rm in}=\frac{1}{\sqrt{\sigma^2+\sigma_{\rm data}^2}},
\]

\[
c_{\rm noise}=\frac{\log\sigma}{4}.
\]

The denoised-state estimate is

\[
D_\theta(y,\sigma)
=
c_{\rm skip}y
+c_{\rm out}F_\theta(c_{\rm in}y,c_{\rm noise},G,R).
\]

The model architecture itself is unchanged.  `c_noise` is supplied through the existing
single graph-level conditioning channel.

## Loss

The EDM weighting is

\[
\lambda(\sigma)
=
\frac{\sigma^2+\sigma_{\rm data}^2}
{(\sigma\sigma_{\rm data})^2}.
\]

For node `i` in physical graph `g`, the equal-channel point loss is

\[
\ell_{gi}
=
\lambda(\sigma_g)
\frac{1}{C}
\left\|D_\theta(y_{gi},\sigma_g)-x_{0,gi}\right\|_2^2.
\]

Spatial reduction follows the repository equal-sample convention: node weights are used
when supplied, otherwise nodes are averaged inside each graph, then physical graph losses
are averaged.

## Project scaling adaptation

The original EDM image defaults use approximately

- `sigma_data = 0.5`,
- `P_mean = -1.2`,
- `P_std = 1.2`,
- sampling `sigma_min = 0.002`,
- sampling `sigma_max = 80`,
- `rho = 7`.

The GraphAttention generative state is statistically standardized to unit channel scale.
M22 therefore uses `sigma_data = 1.0` and multiplies the original sigma magnitudes by two,
preserving their ratios to `sigma_data`:

\[
P_{\rm mean}^{\rm M22}=-1.2+\log 2=-0.5068528194400547,
\]

\[
P_{\rm std}^{\rm M22}=1.2,
\quad
\sigma_{\min}=0.004,
\quad
\sigma_{\max}=160,
\quad
\rho=7.
\]

This scaling is a project adaptation, not a claim that these values are optimal for CFD.

## Karras sampling grid

For `N` nonzero sigma levels, M22 uses

\[
\sigma_i=
\left[
\sigma_{\max}^{1/\rho}
+
\frac{i}{N-1}
\left(
\sigma_{\min}^{1/\rho}-\sigma_{\max}^{1/\rho}
\right)
\right]^\rho,
\qquad i=0,\ldots,N-1,
\]

followed by the exact clean endpoint `sigma_N = 0`.

The first reference generation uses `N=18`, `rho=7`, deterministic Heun integration, and
no stochastic churn.  This requires

\[
2N-1=35
\]

network evaluations because the final step to `sigma=0` uses no second denoiser call.

The probability-flow ODE derivative for the EDM linear noise schedule is

\[
d(x,\sigma)=\frac{x-D_\theta(x,\sigma)}{\sigma}.
\]

Each non-final step uses Euler prediction and, for Heun, a second derivative evaluation at
the predicted next state.

## Why this experiment addresses the M13 endpoint failure

M13/M17 use a discrete cumulative cosine schedule whose final beta is clipped to `0.999`.
M18/M20 showed that the resulting first reverse transition is the dominant catastrophic
instability for the current CFD graph model.

M22 has no discrete beta sequence and therefore no special clipped terminal beta.  The
largest noise level is a finite `sigma_max`; the sampler then follows a smooth finite sigma
grid down to zero.

This removes the specific `beta_T=0.999` mechanism.  It does not guarantee improved
sample quality.

## Controlled-comparison choices

Unchanged from the current M13/M19 controlled baseline:

- five conservative channels;
- HIT physical nondimensionalization;
- train-only sample-balanced statistical scaling;
- grouped source-snapshot split;
- M12 alternating local/exact-two-hop geometric sparse Transformer;
- hidden width 128, four heads, ten layers, MLP ratio 4;
- AdamW configuration;
- 1000 training epochs;
- single-process FP32 network training.

Changed together because they are mathematically coupled in canonical EDM:

- continuous additive-noise parameterization;
- EDM input/output/noise preconditioning;
- log-normal training noise distribution;
- EDM weighted denoised-state objective;
- Karras power-law sigma discretization at inference;
- Euler/Heun ODE sampling.

Therefore M22 is a **formulation comparison**, not a schedule-only ablation.  M21 remains
the cleaner schedule-only comparison inside epsilon-prediction DDPM.

## Reproducibility

Training entry point:

```text
scripts.train_slice_edm
```

Training Slurm:

```text
scripts/slurm/m22_edm_karras_train.slurm
```

Generation entry point:

```text
scripts.generate_slice_edm
```

Generation/benchmark Slurm:

```text
scripts/slurm/m22_edm_karras_generate_benchmark.slurm
```

The generation job benchmarks the full 228-sample held-out population and, when the M19
stable `t=999` DDPM benchmark is available, runs the repository relative benchmark with
M19 as baseline and M22 as candidate.

## Assumptions and deliberate exclusions

Assumptions:

- statistical standardization makes `sigma_data=1` a meaningful first scalar data scale;
- one graph-level noise magnitude is appropriate for one physical sample;
- the existing scalar graph-conditioning interface is adequate for `log(sigma)/4`;
- FP32 network evaluation with float64 sampler state is numerically acceptable.

Handled explicitly:

- variable graph sizes in packed batches;
- sample-level loss reduction;
- deterministic validation keyed by sample ID;
- deterministic generation keyed independently from reference sample IDs;
- finite/positive sigma validation;
- exact final clean endpoint at `sigma=0`;
- Euler and Heun deterministic solvers.

Deferred:

- stochastic EDM churn (`S_churn`, `S_min`, `S_max`, `S_noise`);
- EMA training weights from the original EDM recipe;
- tuning `sigma_data`, `P_mean`, `P_std`, sigma range, or `rho` for CFD;
- DDP and AMP/BF16/FP16 target validation;
- EDM2 learned loss uncertainty/log-variance;
- conditional generation.

## Validation status

Repository implementation and unit tests are present, but CERFACS target execution is
required before marking M22 `TARGET_VALIDATED`.  The first cluster gate is:

```bash
ruff check .
pytest tests/unit/test_edm_diffusion_task.py
pytest
```

The scientific performance claim remains **unvalidated** until the M22 training,
generation, and distribution benchmark finish.
