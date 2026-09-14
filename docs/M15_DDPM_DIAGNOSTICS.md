# M15 — DDPM fixed-timestep diagnostics

## 1. Purpose

The current DDPM epsilon-prediction runs can achieve finite validation epsilon-MSE while producing highly unstable reverse samples. M15 is a diagnostic benchmark intended to localize that failure before changing the diffusion objective, noise schedule, prediction parameterization, sampler, or model architecture.

The diagnostic does not retrain the model and does not alter the reverse sampler. It evaluates a frozen checkpoint on the held-out test split at fixed diffusion timesteps.

## 2. Forward process

For standardized clean state `x0`, Gaussian noise `epsilon`, and discrete cumulative noise coefficient `alpha_bar_t`, the task uses

\[
x_t = \sqrt{\bar\alpha_t}\,x_0
      + \sqrt{1-\bar\alpha_t}\,\epsilon.
\]

The model predicts `epsilon_hat_theta(x_t,t)`.

For the diagnostic, one deterministic Gaussian field is generated per test sample and reused at every selected timestep. This removes an avoidable source of cross-timestep variance: changes in the reported errors are then attributable to the timestep/noise level rather than to drawing a different diagnostic noise realization.

## 3. Epsilon error

The direct training-space metric is

\[
E_\epsilon(t)
=\operatorname{MSE}(\hat\epsilon_\theta,\epsilon).
\]

Reduction is performed within each physical graph first and then averaged over graphs, preserving equal-sample semantics.

The diagnostic reports both per-channel values and an equal-channel mean.

## 4. Clean-state reconstruction

The epsilon prediction implies

\[
\hat x_0
=
\frac{x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon_\theta}
     {\sqrt{\bar\alpha_t}}.
\]

The corresponding reconstruction error is

\[
E_{x_0}(t)=\operatorname{MSE}(\hat x_0,x_0).
\]

This is diagnostically important because a moderate epsilon error can map to a very large clean-state error at low signal-to-noise ratio.

## 5. Exact epsilon-to-x0 amplification

From the forward and reconstruction equations,

\[
\hat x_0-x_0
=
-\sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}
(\hat\epsilon_\theta-\epsilon).
\]

Therefore

\[
E_{x_0}(t)
=
\frac{1-\bar\alpha_t}{\bar\alpha_t}
E_\epsilon(t).
\]

M15 records both the amplitude amplification

\[
A(t)=\sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}
\]

and the MSE amplification factor

\[
A^2(t)=\frac{1-\bar\alpha_t}{\bar\alpha_t}.
\]

The reported ratio `x0_mse_over_expected_from_epsilon` should be approximately one. A material discrepancy would indicate an inconsistency in the diagnostic/reconstruction path rather than ordinary model error.

## 6. Oracle reconstruction gate

The same reconstruction formula is evaluated using the exact injected noise `epsilon` instead of the network prediction.

\[
\hat x_0^{oracle}
=
\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon}
     {\sqrt{\bar\alpha_t}}.
\]

Its MSE should remain near floating-point numerical precision. If it does not, the forward noising coefficients or reconstruction equation must be investigated before interpreting model errors.

## 7. Additional state diagnostics

For every selected timestep and channel the CSV also records population means of per-sample:

- clean-state mean, standard deviation, and maximum absolute value;
- injected-noise mean, standard deviation, and maximum absolute value;
- noisy-state mean, standard deviation, and maximum absolute value;
- predicted-epsilon mean, standard deviation, and maximum absolute value;
- reconstructed-x0 mean, standard deviation, and maximum absolute value;
- reconstructed-x0 / clean-state standard-deviation ratio.

These quantities are intended to reveal whether the denoiser remains numerically bounded while the implied clean state diverges.

## 8. Default timestep grid

For `T=1000`, the default normalized fractions map to

`[1, 10, 50, 100, 250, 500, 750, 900, 990, 1000]`.

This gives dense coverage near the high-noise end where epsilon-to-x0 amplification can become extreme.

## 9. Implementation

- Numerical primitives: `graph_attention.evaluation.diffusion_diagnostics`
- CLI: `python -m scripts.diagnose_slice_diffusion`
- Config: `configs/diagnose_slice_diffusion.yaml`
- Tests: `tests/unit/test_diffusion_diagnostics.py`

The CLI reconstructs the original model, split, geometry, and frozen training standardizers from the persisted diffusion run and evaluates the requested checkpoint only.

## 10. Interpretation ladder

The intended diagnosis is:

1. **Oracle reconstruction is not near numerical precision** -> investigate diffusion coefficients/noising/reconstruction implementation.
2. **Epsilon MSE becomes large only in specific timestep regions** -> investigate timestep conditioning, sampling of training timesteps, and potentially timestep/SNR loss weighting.
3. **Epsilon MSE is moderate but x0 MSE explodes exactly according to the analytical amplification factor** -> the reverse failure is consistent with epsilon-prediction error amplification at low SNR; investigate alternative parameterization/weighting before increasing model capacity again.
4. **Fixed-timestep diagnostics look acceptable or show low-SNR amplification but do not establish how sampler states evolve** -> use M16 reverse-trajectory tracing to measure the actual production sampler state and update magnitudes through the full chain.

No automatic causal claim is produced by the script; it provides the measurements needed to choose the next controlled experiment.

## 11. Assumptions and limitations

- The current CLI targets the existing `PrecomputedSlicePTDataset` diffusion workflow and the persisted held-out test split.
- The diagnostic imports the existing task's internal noising/model-call helpers so it uses the same equations as training/sampling rather than maintaining a parallel diffusion implementation.
- The diagnostic is single-process and uses float32 on the configured device, matching the current generation path.
- M15 itself diagnoses fixed-timestep denoising only. Reverse-trajectory behavior is a separate M16 diagnostic so the one-step and accumulated-error questions remain distinguishable.
