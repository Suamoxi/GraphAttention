# M17 — Improved DDPM formulation

## 1. Purpose

M17 introduces a second discrete-diffusion formulation without changing the M13 baseline. The objective is to test whether the Improved-DDPM training and learned reverse-variance formulation used by DGN4CFD materially improves generation quality before changing the graph backbone, time-conditioning architecture, multiscale hierarchy, or latent representation.

The implementation is a project adaptation of:

- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020;
- Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic Models*, ICML 2021;
- Lino, Pfaff & Thuerey, *Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks*, ICLR 2025.

## 2. Controlled comparison to M13

M17 deliberately preserves the following M13 choices:

- the five standardized HIT state channels;
- the cosine forward schedule with `T=1000` and `s=0.008`;
- one diffusion timestep per physical graph;
- the forward noising equation;
- normalized scalar time conditioning `t/T` appended to graph-level conditioning;
- the selected graph backbone and geometry configuration;
- equal physical-sample weighting;
- train-only statistical standardization;
- grouped train/validation/test split semantics.

The first M17 experiment therefore changes the diffusion formulation, not the graph architecture or time embedding.

## 3. Forward process

For standardized clean state `x0`, timestep `t in {1,...,T}`, and Gaussian noise `epsilon`, M17 uses the same forward process as M13:

\[
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\]

## 4. Two-headed model output

M13 predicts only epsilon. M17 predicts two fields per state channel:

\[
[\hat\epsilon_\theta, v_\theta]=f_\theta(x_t,t,G).
\]

For `C` physical state channels the model therefore returns `2C` output channels.

The first `C` channels predict epsilon. The second `C` channels parameterize the reverse-process variance.

## 5. Learned-range reverse variance

Define

\[
\tilde\beta_t=\beta_t\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}.
\]

Following Improved DDPM, the model variance is parameterized in log space between the posterior lower bound and the forward beta upper bound. The raw network variance output is mapped as

\[
f_t=\frac{v_\theta+1}{2},
\]

then

\[
\log\Sigma_\theta
=f_t\log\beta_t+(1-f_t)\log\tilde\beta_t^{\mathrm{clip}}.
\]

As in the standard learned-range implementation, `f_t` is not explicitly clamped to `[0,1]`; the network can in principle extrapolate beyond the nominal bounds. This behavior is scientifically meaningful and should be monitored rather than silently clipped.

At the first reverse step the posterior variance is exactly zero. The lower log-variance used for learning is therefore clipped by reusing the first finite posterior log-variance, matching the Improved-DDPM/DGN4CFD convention.

## 6. Reverse mean

The reverse mean remains epsilon-parameterized:

\[
\mu_\theta(x_t,t)=
\frac{1}{\sqrt{\alpha_t}}
\left(
 x_t-
 \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
 \hat\epsilon_\theta(x_t,t)
\right).
\]

Thus M17 changes the learned reverse variance and training objective while retaining epsilon prediction for the reverse mean.

## 7. Hybrid objective

The per-sample objective is

\[
L=L_{\mathrm{simple}}+\lambda_{\mathrm{vlb}}L_{\mathrm{vlb}},
\]

with default

\[
\lambda_{\mathrm{vlb}}=10^{-3}.
\]

The simple term is equal-channel epsilon MSE reduced inside each physical graph before graphs are averaged:

\[
L_{\mathrm{simple},g}
=\operatorname{mean}_{i,c}
(\hat\epsilon_{gic}-\epsilon_{gic})^2.
\]

For `t>1`, the VLB contribution is the KL divergence between the exact forward posterior

\[
q(x_{t-1}|x_t,x_0)
\]

and the learned reverse Gaussian

\[
p_\theta(x_{t-1}|x_t).
\]

For `t=1`, the KL term is replaced by a continuous Gaussian decoder negative log likelihood.

The epsilon prediction is detached inside the VLB branch. Consequently:

- the simple MSE trains the epsilon head;
- the VLB term trains the variance head without perturbing epsilon through that branch.

This separation follows the Improved-DDPM strategy used by DGN4CFD.

## 8. Loss-aware timestep sampling

M17 implements the loss-second-moment timestep sampler from Improved DDPM.

Before warm-up, timesteps are sampled uniformly. After every timestep has accumulated the configured history, sampling probabilities are proportional to

\[
p_t\propto\sqrt{\mathbb E[L_t^2]}.
\]

A small uniform component is mixed into the probabilities. The default values follow DGN4CFD:

- history length per timestep: `10`;
- uniform mixture probability: `0.001`.

Each sampled graph loss is multiplied by

\[
w_t=\frac{1}{T p_t},
\]

so the expected optimizer objective remains the uniform-timestep hybrid objective even though difficult timesteps are sampled more frequently.

The sampler history is training state. The first M17 runner does not implement checkpoint resume, so sampler state is not currently restored from checkpoints.

## 9. Equal-sample CFD weighting

All M17 loss components preserve the repository convention that a fine mesh must not receive more statistical weight merely because it has more nodes.

Losses are reduced over channels and nodes inside each physical sample first. Optional node quadrature weights are respected when present. The resulting sample losses are then averaged across graphs.

The loss-aware timestep importance weight multiplies the already sample-reduced loss.

## 10. Sampling

The first M17 sampler intentionally supports only the full learned-variance ancestral process:

- `steps = T`;
- `eta = 1`;
- one model evaluation per reverse timestep;
- learned diagonal variance used at every stochastic reverse transition;
- no stochastic noise added at the final `t=1 -> 0` transition.

Reduced-step DDIM or spaced learned-variance diffusion is deliberately deferred. Mixing a new accelerated sampler into the first formulation experiment would add another scientific variable.

## 11. What M17 does not change

M17 does **not** yet implement DGN4CFD's:

- sinusoidal diffusion-step embedding;
- per-message-passing-layer time injection;
- multiscale graph hierarchy;
- graph pooling/unpooling;
- latent VGAE;
- Dirichlet-boundary diffusion treatment;
- learning-rate plateau scheduler;
- accelerated spaced learned-variance sampler.

These remain separate experiments.

## 12. Implementation paths

- task and learned-variance sampler: `src/graph_attention/tasks/improved_diffusion.py`;
- task config: `configs/task/hit_improved_diffusion.yaml`;
- training runner: `scripts/train_slice_improved_diffusion.py`;
- generation runner: `scripts/generate_slice_improved_diffusion.py`;
- scientific tests: `tests/unit/test_improved_diffusion_task.py`.

## 13. Initial validation requirements

Before any learning-quality claim:

1. run the full unit test suite;
2. run a one-epoch GPU smoke test;
3. verify finite simple, VLB, and hybrid losses;
4. verify that the loss-aware sampler warms as expected;
5. generate with the full learned-variance ancestral chain;
6. run `generation_distribution_v2` and the relative benchmark against the M13 epsilon-only baseline;
7. re-run the fixed-timestep and reverse-trajectory diagnostics after they are explicitly adapted to two-headed model output.

The current M15/M16 diagnostics are defined for the epsilon-only M13 output contract and must not be assumed compatible with M17 until updated.

## 14. Assumptions and deferred edge cases

- The first runner is single-process FP32, matching the current M13 runner.
- `T>=2` is required because the learned lower variance uses the first finite posterior variance to clip the `t=1` log variance.
- Learned variance is diagonal and predicted independently per node/channel.
- The raw learned-range variable is not clamped.
- Full ancestral sampling is the only currently supported M17 generation path.
- Resume training, DDP, AMP/BF16, spaced learned-variance sampling, and improved-diffusion trajectory diagnostics are deferred.
- The implementation preserves the current raw normalized scalar time conditioning so any M17/M13 difference is not attributable to a richer time embedding.
