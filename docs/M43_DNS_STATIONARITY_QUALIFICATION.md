# M43 follow-up: stationarity qualification and physically normalized case comparison

Status: implementation on feature branch, **no production postprocessing results approved**.
Based on the successful six-case DNS audit job 491738 (600 snapshots).

The existing `feature/m43-dns-stationarity-audit` computations and M43-B
training artifacts remain unchanged. This extension consumes the immutable
audit `time_history.csv`, `spectra/snapshot_NNN.npz`, and `lag` cache, then
creates a separate `m43_qualification_v1` output directory.

## Stages and safety gates

1. `qualify`: compute per-snapshot FFT spectral-band energy fractions (shells
   `1<=k/k_min<4`, `4<=k/k_min<16`, `>=16`). Assess turbulent K, enstrophy,
   density RMS, f4 unresolved-energy fraction, and all three bands. Compare
   rolling windows against the final three windows with a default 22% relative
   tolerance, autocorrelation-aware circular block bootstrap CIs (capped 35%
   acceptance threshold), and a 25% normalized drift gate. Demand three
   complete future windows, report sensitivity across windows/tolerances.
   The candidate index is the **end** of the first passing window, not its
   start; this prevents an initial transient inside that window from being
   mislabeled stationary. No cutoff is automatically accepted.
2. Review `qualification_summary.json`, `cutoff_proposals_NOT_REVIEWED.json`,
   `snapshot_classification_manifest.csv` and the existing DNS PDFs. Review
   actual physical time, late-time drift, temporal correlations and spectrum
   evolution. Convert chosen boundaries to an explicit `reviewed_cutoffs.json`.
   The original audit's candidate indices, especially Ma0.3, are not approval.
3. `sample`: require authoritative six-case M3.3 YAML references and all six
   reviewed cutoffs. Uniformly sample **unique periodic DNS nodes** only from
   reviewed stationary snapshots. Save five conservative fields after *the
   exact M3.3 convective scale* plus normalized velocities. No per-case z-score.
   Validate archive file metadata against original audit signature, case IDs,
   reference provenance/scheme and cache settings. Never write to source HDF5.
4. `compare`: calculate per-case stationary means with time-block confidence
   intervals, marginal moments, PDFs via quantiles, pairwise dimensionless
   Wasserstein-1 distances divided by pooled standard deviation and descriptive
   within-case half-trajectory W1. Compute dimensionless physical spectra on
   shared wavenumber support `k*=k L_ref` and `E*=E/(U_ref^2 L_ref)`, residual
   S2 and correlation diagnostics with matched dimensionless filter widths.
   Incommensurate residual-filter widths are flagged, not silently compared.

Statistical stationarity is an ensemble property: a single DNS trajectory,
even after bootstrap correction, cannot prove convergence. Re54 always has a
65^3 stored grid and Re94 a 129^3 grid, so Reynolds number and resolution
are **confounded**. Comparing them cannot isolate the effect of either one.
Different Mach values retain meaningful physics after nondimensionalization.
The AVBP periodic-node sampling is uniform in discrete voxel weights and is
not a verified finite-volume quadrature.

## Commands on Calypso

Create an independent checkout from the audit feature branch; do not reset or
alter a training-resumable M43-B checkout:

```bash
git fetch origin
git worktree add --detach /scratch/coop/theret/GraphAttention_dns_qualification \
    origin/feature/m43-dns-stationarity-qualification
cd /scratch/coop/theret/GraphAttention_dns_qualification
export M43_Q_REPO="$PWD"
```

Qualification requires only the uploaded audit cache and no original DNS:

```bash
sbatch scripts/slurm/m43_dns_qualify_compare.slurm
```

After reviewing **all six** cases, create a JSON file using **zero-based**
indices; these example values are intentionally placeholders and NOT
scientifically recommended:

```json
{
  "Re54/Ma0.05": {"transient_end_index": 9, "stationary_start_index": 40},
  "Re54/Ma0.2": {"transient_end_index": 9, "stationary_start_index": 40},
  "Re54/Ma0.3": {"transient_end_index": 9, "stationary_start_index": 40},
  "Re94/Ma0.05": {"transient_end_index": 9, "stationary_start_index": 40},
  "Re94/Ma0.2": {"transient_end_index": 9, "stationary_start_index": 40},
  "Re94/Ma0.3": {"transient_end_index": 9, "stationary_start_index": 40}
}
```

Re-run `qualify` with the cutoffs to write approved-interval labels. Then:

```bash
export M43_Q_REVIEWED_CUTOFFS=/scratch/coop/theret/reviewed_cutoffs.json
export M43_Q_REFERENCE_DIR=/scratch/coop/theret/HIT_DNS_COMP_case_references
sbatch --export=ALL,M43_Q_MODE=qualify scripts/slurm/m43_dns_qualify_compare.slurm
sbatch --export=ALL,M43_Q_MODE=sample scripts/slurm/m43_dns_qualify_compare.slurm
# After sampling has finished:
sbatch --export=ALL,M43_Q_MODE=compare scripts/slurm/m43_dns_qualify_compare.slurm
```

Never submit sampling/comparison until the reference YAML files and reviewed
stationarity cutoffs are available. `sample` and `compare` fail closed without
both. Existing cached samples are bound to the SHA-256 of each reference YAML,
original audit source signature, cutoff JSON and sampling settings; changed
inputs require a **new** output directory rather than silent reuse.

## Main outputs

```
.../m43_dns_stationarity_audit/m43_qualification_v1/
  qualification_summary.json
  cutoff_proposals_NOT_REVIEWED.json
  snapshot_classification_manifest.csv
  fields/source_signature.json
  fields/Re54_Ma0.05/snapshot_NNN.npz  # and the other five cases
  cross_case_summary.json
  pairwise_m3_3_wasserstein.csv
  pairwise_m3_3_residual_lag.csv
  stationary_m3_3_energy_spectra.pdf
  stationary_m3_3_residual_S2.pdf
  cross_case_report.md
```

The new comparisons are descriptive; do not use threshold crossing as proof
that a DiNAT model needs or does not need explicit physical conditioning.
Existing training splits, normalizers and checkpoints remain unchanged.
