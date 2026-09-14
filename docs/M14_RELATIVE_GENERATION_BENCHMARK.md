# M14 — Relative generation benchmark reporting

## 1. Purpose

The existing `generation_distribution_v2` benchmark remains the authoritative source of raw generation diagnostics. M14 adds a post-processing comparison layer that expresses those diagnostics on dimensionless, physically interpretable error scales before model-to-model relative improvements are quoted.

The motivation is that a statement such as “model B is 3x better” can be misleading when both errors are already negligible. Therefore every relative model-to-model improvement is reported together with:

1. the baseline dimensionless error;
2. the candidate dimensionless error;
3. the absolute change in that error;
4. the relative error reduction.

No single weighted composite score is introduced.

## 2. Marginal metrics

For channel `c`, let `W1_c` be the empirical one-dimensional Wasserstein distance and `sigma_r,c` the pooled test-reference standard deviation.

The normalized distribution error is

\[
E_{W_1,c} = \frac{W_{1,c}}{\sigma_{r,c}}.
\]

The normalized mean-bias magnitude is

\[
E_{\mu,c} = \frac{|\mu_{g,c}-\mu_{r,c}|}{\sigma_{r,c}}.
\]

The standard-deviation ratio error is

\[
E_{\sigma,c} = \left|\frac{\sigma_{g,c}}{\sigma_{r,c}}-1\right|.
\]

The benchmark also records the arithmetic mean of each error over channels. These channel means are reporting summaries only; they do not imply equal physical importance of all variables.

## 3. Spectrum metrics

For each configured spectral band and channel,

\[
E_{P,c,b}=\left|\frac{P_{g,c,b}}{P_{r,c,b}}-1\right|.
\]

A mean band error is reported across channels.

For the full radial spectrum, M14 also reports an RMS log-ratio error,

\[
E_{\log P}
=
\sqrt{
\frac{1}{N}
\sum_{c,k}
\left[
\log\left(\frac{P_{g,c,k}+\varepsilon}{P_{r,c,k}+\varepsilon}\right)
\right]^2
},
\]

where `epsilon` is the configured numerical floor. The log-ratio form treats reciprocal over- and under-prediction symmetrically.

## 4. Local spatial metrics

For each channel, population means of the existing per-sample statistics are compared.

Spatial-standard-deviation error:

\[
E_{\sigma_x,c}
=
\left|
\frac{\langle\sigma_{x,g}\rangle}
     {\langle\sigma_{x,r}\rangle}
-1
\right|.
\]

Nearest-neighbour correlation error:

\[
E_{\rho_{NN},c}
=
\left|
\langle\rho_{NN,g}\rangle
-
\langle\rho_{NN,r}\rangle
\right|.
\]

First-difference RMS error:

\[
E_{\Delta,c}
=
\left|
\frac{\langle RMS_{\Delta,g}\rangle}
     {\langle RMS_{\Delta,r}\rangle}
-1
\right|.
\]

Arithmetic channel means are recorded for presentation convenience.

## 5. Cross-channel correlation

For the unique off-diagonal entries of the generated and reference channel-correlation matrices,

\[
E_C
=
\sqrt{
\frac{1}{M}
\sum_{i<j}
\left(C^g_{ij}-C^r_{ij}\right)^2
}.
\]

## 6. Nearest-reference calibration

The existing benchmark already reports generated-to-test and test-to-generated mean nearest-neighbour distances normalized by the test-to-test leave-one-out mean distance.

M14 reports the absolute deviation of these calibrated ratios from one. This is a calibration diagnostic, not a monotonic standalone quality score: values below one can also arise from over-concentration or memorization. The underlying fidelity and coverage distances must therefore remain visible.

## 7. Physical metrics

The existing `physical_metrics.csv` declares how each quantity should be compared. M14 converts those rules to non-negative error magnitudes:

- `bias_over_reference_std` -> absolute normalized bias;
- `generated_over_reference` -> absolute ratio error from one;
- `absolute_difference` -> absolute difference.

Physical metrics are not averaged into a global score.

## 8. Model-to-model comparison

For a baseline error `E_b` and candidate error `E_c`, the comparison file reports

\[
\Delta E = E_c-E_b,
\]

and the absolute error reduction

\[
R_{abs}=E_b-E_c.
\]

The relative reduction is

\[
R_{rel}=\frac{E_b-E_c}{E_b}.
\]

A positive `R_abs` or `R_rel` means the candidate is better. Relative reduction must always be interpreted alongside `E_b`, `E_c`, and `R_abs`.

Example:

\[
E_b=3\times10^{-7},\qquad E_c=1\times10^{-7}
\]

corresponds to a 66.7% relative reduction but only a `2e-7` absolute reduction in the already dimensionless error scale.

## 9. Implementation

- Core computation: `graph_attention.evaluation.relative_benchmark`
- CLI: `python -m scripts.compare_generation_benchmarks`
- Config: `configs/compare_generation_benchmarks.yaml`
- Tests: `tests/unit/test_relative_benchmark.py`

The tool reads existing benchmark directories and never reruns generation or changes the original benchmark files.

## 10. Assumptions and limitations

- The compared runs must use the same sample space, reference population, channel semantics, mesh node count, and 2-D grid shape.
- The implementation accepts the whitespace-padded CSV headers present in some earlier benchmark artifacts by stripping field names and values when reading.
- Channel-mean summaries are deliberately simple arithmetic means and are not a physical weighting of the state variables.
- No arbitrary threshold is used to suppress large relative improvements at tiny errors. Instead the absolute dimensionless errors and absolute error change are always persisted next to the percentage reduction.
- No composite score is defined because weighting heterogeneous marginal, spectral, local, correlation, nearest-reference, and physical diagnostics would introduce an additional scientific choice.
