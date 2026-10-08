# Physics of the 3-D HIT spatial-correlation and structure-function plots

This note describes **the implemented estimators**, their mathematical properties,
what their curves mean physically, and where classical turbulence theory requires
additional assumptions. It supplements [SCIENTIFIC_SPEC.md](SCIENTIFIC_SPEC.md)
sections 23-24 and [M40_3D_VOLUME_BENCHMARK.md](M40_3D_VOLUME_BENCHMARK.md).
The same estimators are used in the M37 (DiNAT-DiT), M38 (Full DiT), and M40
(local self-attention) full-volume generation benchmarks.

Primary code:
- src/graph_attention/evaluation/spatial_statistics_3d.py
- src/graph_attention/evaluation/plotting_3d.py
- src/graph_attention/evaluation/generation_benchmark_3d.py
- configs/benchmark_generation_3d.yaml

## 1. Data, geometry and averaging conventions

The benchmark begins with nondimensional conservative fields
(rho, rhou, rhov, rhow, rhoE). Velocity is calculated pointwise,
u = (rhou/rho, rhov/rho, rhow/rho). Every density value must be finite and
strictly positive. Calculations are **volume/gridpoint-weighted**, NOT
Favre-density-weighted.

The stored mesh is 33 x 33 x 33 nodes, containing repeated maximum-coordinate
planes in a periodic box. For statistics the implementation drops the repeated
endpoint on each axis and uses 32 x 32 x 32 unique periodic points. This is
a postprocessing choice only, and it does not add periodic edges to the graph
attention training geometry.

For each *individual sample* s, define

    u'_s(x) = u_s(x) - <u_s>_V

where <.>_V is an arithmetic mean over all 32^3 periodic voxels. Let
sigma^2_{s,i} = <u'^2_{s,i}>_V and
Q_s = sum_i sigma^2_{s,i} = <|u'_s|^2>_V. Zero variance in any
component is rejected.

The available separations are positive Cartesian directions,
r_a = m * Delta * e_a, for axis a = x,y,z and integer m = 0,...,16.
The box length is L = 32 Delta, so the plotted abscissa is

    r/L = m/32, from 0 to 1/2 in increments of 1/32.

np.roll with periodic wrapping computes u'(x+r_a) from u'(x). Every
spatial point contributes. The code averages the three axis directions;
this is an axis-averaged HIT estimator, **not** full solid-angle radial
pair enumeration. There is **no temporal correlation** in these plots.

## 2. Implemented two-point velocity correlations

For each snapshot s and separation m, the per-axis correlations are

    C_{s,a,b}(r) = <u'_{s,b}(x) u'_{s,b}(x+r e_a)>_V
                    / sigma^2_{s,b}.

The three saved normalized quantities are

    R_LL,s(r) = (1/3) sum_a C_{s,a,a}(r)

    R_NN,s(r) = (1/6) sum_a sum_{b != a} C_{s,a,b}(r)

    R_uu,s(r) = (1/3) sum_a
       <u'_s(x) dot u'_s(x+r e_a)>_V / Q_s.

Here longitudinal means velocity *parallel* to separation and transverse
means perpendicular, not velocity components relative to a fixed global
viewing direction. All three functions are explicitly set to one at r=0.

Physical interpretation:
- R near +1: velocity fluctuations remain highly alike after displacement r.
- R near 0: little two-point linear covariance at r.
- R below 0: tendency toward opposite-sign fluctuations at that distance.
- Faster decorrelation: shorter velocity coherence scales; too-slow
  decorrelation means too much persistent large-scale organization.
- The integral length scale in classical turbulence relates to the area
  under an appropriate longitudinal correlation function. The present
  half-box finite-domain curves do not guarantee reliable integral-scale
  estimation.

The normalization is performed independently for each velocity component
in R_LL and R_NN **for each snapshot**, before axis averaging. R_uu uses
the total fluctuation variance, giving variance-weighted components. These
are the exact code semantics, not a common variance pooled over snapshots.
All are dimensionless. Normalization removes amplitude information: a
model can match R(r) while having the wrong kinetic energy.

For homogeneous, isotropic, incompressible turbulence with equivalent
component variances, the well-known theoretical relation

    R_NN(r) = R_LL(r) + (r/2) * dR_LL(r)/dr

provides a consistency check. The discrete axis-averaged, possibly
compressible/LES sample need not satisfy it exactly.

## 3. Implemented velocity structure functions

For each direction a, define the signed increment

    delta_a u_b(x;r) = u_b(x+r e_a) - u_b(x).

Spatially constant mean velocity cancels, so increments of u and u' are
identical. The longitudinal increment is delta_a u_a.

The per-snapshot longitudinal structure functions are

    S_{2,L,s}(r) = (1/3) sum_a <(delta_a u_a)^2>_V

    S_{3,L,s}(r) = (1/3) sum_a <(delta_a u_a)^3>_V

    S_{4,L,s}(r) = (1/3) sum_a <(delta_a u_a)^4>_V.

The total-vector second-order structure function is

    S_{2,|u|,s}(r) = (1/3) sum_a <sum_b (delta_a u_b)^2>_V.

The plotted longitudinal flatness is computed PER SNAPSHOT AFTER
averaging directions:

    F_{L,s}(r) = S_{4,L,s}(r) / [S_{2,L,s}(r)]^2.

It is undefined at r=0 where S2=0 (saved as NaN).
This is not generally the average of the per-axis flatness ratios.

Physical meanings:

- S2_L quantifies the squared change in velocity along the separation
  direction. Small S2 indicates similar velocities; larger S2 indicates
  increasingly different velocities.
- S2_vector measures total velocity-vector increment power and includes
  longitudinal plus two transverse component contributions.
- S3_L retains the sign of increments, so it measures the *asymmetry* of
  longitudinal velocity differences. For the chosen convention
  delta u_L = u_L(x+r)-u_L(x), classical high-Re locally isotropic,
  homogeneous, stationary incompressible turbulence in an inertial range
  predicts Kolmogorov's 4/5-law:
      S3_L(r) = -(4/5) * epsilon * r.
  Negative S3 is consistent with the direct 3-D forward energy cascade.
  An observed departure from this law is NOT proof of a bad model:
  the benchmark is on a low-resolution 32^3 unique grid, using nondimensional
  fields, and an LES/possibly compressible forced HIT context. It does not
  estimate epsilon or identify a validated inertial interval.
- S4_L measures high-order moments and is especially sensitive to rare,
  strong velocity gradients or increments.
- F_L is the *raw kurtosis* (not excess kurtosis) of longitudinal increments;
  Gaussian distributed increments have F=3. Values much above 3 are a common
  indicator of intermittency, but noise, sample size and short scale ranges
  matter, so do not treat a threshold as a conclusive turbulence classifier.

Velocity-based structure functions carry units of velocity^p (or powers of
the nondimensional velocity in this dataset); flatness is dimensionless.
No Favre weighting or density factor is present in these estimators.

## 4. Exact correlation–structure function identity

Periodicity ensures <u_a(x+r)^2>_V = <u_a(x)^2>_V.
Expanding the square gives, exactly per sample and axis,

    <(delta_a u_b)^2>_V
        = 2 * sigma^2_b * [1 - C_{a,b}(r)].

Consequently the **code's axis-averaged** estimators obey

    S2_L(r) = (2/3) sum_a sigma^2_a * [1 - C_{a,a}(r)].

If sigma_x^2 = sigma_y^2 = sigma_z^2 = sigma_u^2 (isotropic equality),

    S2_L(r) = 2 sigma_u^2 * [1 - R_LL(r)].

The vector quantities obey without an isotropy assumption,

    S2_vector(r) = 2 Q * [1 - R_uu(r)].

These identities show why S2 and R encode related second-order physics:
S2 adds the fluctuation amplitude that a normalized correlation hides.
If R decays toward zero, S2 approaches twice the relevant variance.
If R becomes negative, S2 can exceed that decorrelated level.

In isotropic incompressible turbulence, the trace correlation is a
Fourier-transform counterpart of the three-dimensional energy spectrum.
With the conventional normalized E(k), approximately under full isotropy,

    <u'(x) dot u'(x+r)> = 2 integral E(k) sin(kr)/(kr) dk,

and integral E(k) dk = Q/2. Thus Fourier energy spectra and two-point
correlations give complementary views of the same second-order statistics.
They do not encode the third-order asymmetry measured by S3.

## 5. What appears in each saved plot

The curves are aggregated AFTER computing each complete per-snapshot
statistic. At every r/L:

- Solid line: arithmetic mean over all independently generated snapshots,
  or mean over all held-out reference snapshots, respectively.
- Shaded band: per-snapshot 10th to 90th percentile, NOT a confidence
  interval for the mean and NOT paired example-by-example uncertainty.
- One normalized correlation figure each: longitudinal_correlation.png,
  transverse_correlation.png, vector_correlation.png.
- The correlations use linear vertical values and horizontal r/L.
- Four default structure-function figures:
  longitudinal_s2.png, vector_s2.png (log x AND log y);
  longitudinal_s3.png, longitudinal_flatness.png (log x, linear y).
- The S4_L curve **is computed and stored in structure_functions.csv** but
  **is not rendered into a default PNG**.
- Structure-function plots omit r=0 (where flatness is undefined and log r
  cannot be plotted). Correlation plots include r=0.
- No loss-function pairing exists between generated and reference curves.

Exact implementation functions:
save_velocity_spatial_correlation_plots,
save_velocity_structure_function_plots, and _spatial_population_rows_3d.

## 6. Interpretation and limitations for M37 / M38

Good model agreement means matching **curves, their slopes/curvature and
their snapshot variability**, not making correlations larger, S2 smaller,
S3 more negative, or flatness higher in isolation.

Suggested diagnostic questions:
1. Does generated R_LL or R_uu decay too slowly (over-coherent large scales)
   or too quickly (under-coherent spatial structures)?
2. Does S2_vector have the correct amplitude and characteristic r dependence?
   This catches wrong kinetic energy even when R is well matched.
3. Is longitudinal versus transverse behavior consistent with isotropic HIT?
4. Are S3_L sign and scale dependence reproduced, without over-claiming a
   universal 4/5-law at the available resolution?
5. Does F_L reproduce the intermittent increment tails at the smallest
   resolved separations, and are results stable across examples/seeds?
6. Do DNS/held-out-reference and generation bands overlap naturally without
   treating individual generated samples as paired targets?

Scientific caveats:

- The source case is named HIT_LES_FORCED: label plotted data as a held-out
  **simulation reference** unless a separate source establishes it is DNS.
  Some existing figure scripts call it DNS, which should not be accepted
  as proof of DNS fidelity.
- The analysis uses periodic *postprocessing* and assumes the field is
  periodic; the graph model itself may not have periodic cross-boundary edges.
- Only three positive Cartesian separation directions are sampled, not all
  directions; therefore anisotropy can be partly hidden by axis averaging.
- 32 grid points per axis provide only 16 positive spatial separation values.
  There may be no meaningful inertial range in which inertial scaling laws
  can be tested.
- The raw S3 magnitude depends on the velocity normalization, and the
  velocity increment distribution must be adequately sampled to interpret
  high moments.
- In compressible/variable-density turbulence, velocity-based equal-volume
  correlations and structure functions do not obey every incompressible
  exact balance. Density-weighted alternatives are different diagnostics.

## 7. A reproducible sanity check

The repository regression test
test_velocity_spatial_statistics_3d_match_periodic_sinusoid constructs
u_x = sin(2 pi x/N), u_y = sin(2 pi y/N), u_z = sin(2 pi z/N).

At the first separation m=1 it expects exactly

    R_LL(Delta) = cos(2*pi/N),
    S2_L(Delta) = 1 - cos(2*pi/N),
    S3_L(Delta) = 0.

Since each component variance is 1/2, this also verifies the identity
S2_L = 2*(1/2)*(1-R_LL). The test compares numerical averages of periodic
sinusoidal fields against these analytic predictions.
