# M21 — Linear-beta DDPM schedule ablation

## 1. Purpose

M18-M20 showed that the existing cosine DDPM baseline becomes catastrophic when the reverse chain includes the terminal `1000 -> 999` transition, while the same trained model becomes numerically sensible when initialized at `t=999`. M21 isolates the diffusion schedule as the next controlled variable.

The experiment keeps the M13 model, epsilon-prediction objective, preprocessing, split, optimizer, normalized scalar time conditioning, and ancestral sampler unchanged. Only the forward beta schedule changes.

## 2. Schedule

The M21 reference schedule is

```text
T = 1000
beta_start = 1e-4
beta_end = 2e-2
beta_t = linspace(beta_start, beta_end, T)
```

with

$$
\alpha_t = 1-\beta_t,
\qquad
\bar\alpha_t = \prod_{s=1}^{t}\alpha_s.
$$

No beta clipping is applied. The configured endpoint is itself the maximum beta.

For the reference values,

```text
alpha_bar_T ~= 4.03583e-5
sqrt(alpha_bar_T) ~= 6.3528e-3
beta_T = 0.02
1 / sqrt(alpha_T) ~= 1.01015
```

Thus the terminal forward marginal is already close to Gaussian while the final reverse transition remains finite and mild compared with the cosine baseline's clipped `beta_T=0.999` endpoint.

## 3. Controlled comparison

Reference comparison:

```text
M13 cosine:
  T=1000
  cosine_s=0.008
  epsilon prediction
  M12 L10 backbone

M21 linear:
  T=1000
  beta_start=1e-4
  beta_end=2e-2
  epsilon prediction
  same M12 L10 backbone
```

The intended full M21 generation uses all 1000 adjacent reverse transitions with `eta=1` and starts from `N(0,I)` at the actual terminal timestep. Unlike the M19 Gaussian restart, no trained transition is omitted.

## 4. Interpretation

M21 tests whether the observed endpoint instability is primarily caused by the discretized/clipped cosine schedule rather than by DDPM as a generative formulation.

A stable full-chain M21 result would support the hypothesis that the pathological `beta_T=0.999` cosine endpoint is the dominant numerical/statistical problem for the current epsilon-prediction graph model.

A stable schedule alone does not establish that the linear schedule is optimal. Quality must still be evaluated with the full population benchmark, including marginal W1, mean/std errors, spectra, local spatial statistics, descriptor fidelity/coverage, and physical admissibility.

## 5. Implementation

- `src/graph_attention/tasks/linear_diffusion.py`
- `configs/task/hit_diffusion_linear.yaml`
- `scripts/train_slice_linear_diffusion.py`
- `scripts/generate_slice_linear_diffusion.py`
- `tests/unit/test_linear_diffusion_task.py`

The subclass reuses the existing M13 task and sampler and replaces only `_alpha_bar_cpu` with the exact cumulative product implied by the configured linear betas.

## 6. Validation status

Implemented in the repository. Target validation on Calypso/GH200 remains required:

```bash
ruff check .
pytest tests/unit/test_linear_diffusion_task.py
pytest
```

No learning-quality claim exists until the full M21 training, generation, and benchmark finish.
