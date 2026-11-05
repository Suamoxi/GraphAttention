# M43 — physically referenced similarity diagnostics (v1)

**Purpose.** Test whether explicitly prescribed thermodynamic reference values
improve the collapse of compressible HIT statistics across the six `Re54/Re94 ×
Ma0.05/0.2/0.3` cases. Reuse the already reviewed **exploratory** stationary
snapshot interval and the already sampled M3.3 field cache. No model training,
DNS restart, HDF5 rewriting, or M3.3 case-reference rewriting is performed.

This is a diagnostic experiment, **not** a change to M43 training preprocessing,
physical conditioning, or the original baseline statistics. Statistical
stationarity is not mathematically certified by a 60-snapshot segment.

## Physical representation

With `rho_ref, U_ref, L_ref` from the frozen M3.3 reference YAML, obtain
three additional values **from each simulation's actual setup**:

- `rho0`: prescribed conserved mean density, in `kg/m^3`, independently known
  before generation. It is **not** re-estimated from the stationary DNS samples.
- `a_ref`: thermodynamic reference sound speed, `m/s`, with clear specification
  of which characteristic velocity defines the nominal Mach number.
- `e_ref`: reference specific internal energy, `J/kg` or `m^2/s^2`, in the
  **same energy-zero convention as AVBP's stored `rhoE`**.

Use **reference-based** `Ma_ref=U_ref/a_ref`; compare it to the case's nominal
Mach but do not silently assume that the definitions agree. Some studies
use a three-component turbulent RMS velocity, while M3.3's velocity reference
might use a one-component RMS. A mismatch is reported, not rewritten.

The five diagnostic channels are

```
u* = u/U_ref; v* = v/U_ref; w* = w/U_ref
rho_dagger = (rho - rho0)/(rho0 * Ma_ref**2)
E_dagger = (rhoE - rho * e_ref)/(rho_ref * U_ref**2)
```

The diagnostic transform is exactly invertible: `rho = rho0 * (1 +
Ma_ref**2 * rho_dagger)`; `rhou = rho*U_ref*u*`, etc.; `rhoE = rho*e_ref +
rho_ref*U_ref**2*E_dagger`. Input momenta are validated against cached
velocities, density must be positive, and the inversion is checked.

A fixed `rho_ref` used to scale amplitudes is **not automatically** a good
center for `rho_dagger`: if `rho_ref` differs from the conserved mean, the
resulting offset is magnified by `Ma_ref**-2` (especially at Ma0.05).
The reference audit reports this offset in units of measured density RMS.
It does not automatically replace `rho0` by observed density means.

In a perfect-gas setup, deriving `e_ref` from `p_ref, rho0, gamma` may be
appropriate, but only with a documented EOS and the solver's internal-energy
zero; **the script never assumes gamma, temperature, or a caloric gas model**.

## Strict provenance: preserve M3.3 YAML file hashes

The original files in `/scratch/coop/theret/HIT_DNS_COMP_case_references/`
are bound to the `m43_qualification_v1/fields/source_signature.json` hash.
**Do not edit them.** Additional reference values go in separate sidecar
case-definition YAMLs, using the repo's existing `CaseDefinition` schema and
same `case_id` and `reference_scheme`, in:

`/scratch/coop/theret/HIT_DNS_COMP_similarity_references/`

For each case (e.g. `Re54_Ma0.05.yaml`) provide `rho0`, `a_ref`, `e_ref`
with positive explicit literal `value`, units, `definition`, provenance,
`inference_available: true`, and non-snapshot scope. The `inspect` mode
writes **non-runnable templates** with `value: null`; fill values only after
consulting the authoritative HIT setup, not from the 60 DNS target samples.
Reference-based Mach differences >5% from the case label are flagged; check
the exact definition before drawing conclusions.

If the existing frozen M3.3 YAMLs already contain the three extra values,
sidecars are optional; the code can read those fields without modification.
Any changed sidecar hash, sample signature, cutoff JSON, or sampling settings
requires a new **output directory**; no silent overwrite of scientific results.

## Calypso commands

Use an independent worktree rather than modifying the M43 training checkout:

```bash
git fetch origin
git worktree add --detach /scratch/coop/theret/GraphAttention_dns_similarity \
  origin/feature/m43-reference-similarity-diagnostics
cd /scratch/coop/theret/GraphAttention_dns_similarity
export M43_S_REPO="$PWD"
```

Stage 1 — inspect existing physical references and generate templates:

```bash
sbatch scripts/slurm/m43_similarity_diagnostics.slurm
```

This checks existing case YAMLs and audit density histories. Read

```
/scratch/coop/theret/GraphAttention_runs/m43_dns_stationarity_audit/
  m43_reference_similarity_v1/reference_audit.json
  m43_reference_similarity_v1/reference_templates_NOT_VALIDATED/*.yaml
```

Copy each template to the sidecar reference directory and replace **all**
`value: null` fields and placeholder provenance with the original
simulation-setup values. Do not use the observed late-time rho for `rho0`.

Stage 2 — compare with *explicit* reviewed exploratory cutoffs:

```bash
export M43_S_REVIEWED_CUTOFFS=/scratch/coop/theret/reviewed_cutoffs_exploratory_40.json
export M43_S_REFERENCE_DIR=/scratch/coop/theret/HIT_DNS_COMP_case_references
export M43_S_STATE_DIR=/scratch/coop/theret/HIT_DNS_COMP_similarity_references
sbatch --export=ALL,M43_S_MODE=compare scripts/slurm/m43_similarity_diagnostics.slurm
```

The first stage should be rerun after completing the sidecars to check the
Mach definitions and density offsets, before interpreting stage-2 outputs.

### Outputs (all new, separate from M43-Q)

- `reference_audit.json`: provided vs missing reference values and setup
  provenance, nominal-Mach consistency and density-centering checks.
- `similarity_summary.json`, `similarity_report.md`: six-case descriptive
  summaries with explicit limitations.
- `reference_state_marginals.csv`: reference-state transformed means, quantiles,
  standard deviations and moving-block CIs for snapshot means.
- `pairwise_reference_state_wasserstein.csv`: physical W1 distances and pooled-
  sigma normalization **before and after** the transform; first-half/second-half
  baselines and snapshot-block uncertainty.
- `pairwise_well_resolved_spectra.csv`, `well_resolved_common_k_spectra.pdf`:
  common dimensionless support restricted to **70% of the minimum directional
  Nyquist** of periodic 64^3 and 128^3 grids. Distances are computed over the
  common support, also separately in three equal-log-wavenumber bands so
  low-k energy does not conceal smaller-scale differences.
- `pairwise_matched_delta_residual.csv`, `matched_delta_residual_S2.pdf`:
  recomputed residual S2 and longitudinal correlation at nearest matched
  `Delta/L_ref`, with moving-block CIs for S2 distance.
- `reference_state_one_point_pdfs.pdf`: original and transformed density and
  energy distributions, plotted **without per-case z-scoring**.

### Limitations and interpretation

1. Spherical Fourier shells beyond directional Nyquist are incomplete on a
   Cartesian FFT grid. Using 70% of the *minimum* directional Nyquist makes
   cross-grid energy comparisons conservative; it does not equalize numerical
   dissipation or spatial sampling quality.
2. Equal `Delta/L_ref` for Re54 and Re94 does **not** give the same DNS
   fine-scale bandwidth. A matched-filter block residual still contains
   different ranges of resolved fine scales. It cannot isolate Reynolds effects.
3. Re54 uses 65^3 stored / 64^3 unique periodic nodes; Re94 uses 129^3 /
   128^3. Reynolds and mesh resolution remain confounded by construction.
4. Time-block bootstrap resamples snapshots, not spatial nodes, and is
   descriptive for a single temporal realization; its intervals are not
   exchangeability tests or p-values. Within-trajectory half-split W1 is not a
   statistical null distribution.
5. Stronger density collapse after dividing by Ma^2 **does not** imply the
   physical Mach dependence disappears. A shared unconditional model still
   cannot request a chosen Mach without conditioning or an explicitly invertible
   reference-state specification at inference.
6. The outputs diagnose similarity; no modifications to normalizer fitting,
   checkpoints, model architecture or training splits are included.
