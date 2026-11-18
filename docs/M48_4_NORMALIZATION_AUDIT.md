# M48.4b — case-resolved normalization audit (teacher only)

## Goal

Before full real-data training, quantify whether the **single pooled scaler**
used by M48.4a introduces large cross-regime amplitude or mean offsets.
This audit is observational: **no per-case scaler is fitted for training**,
no code in the original MS_SR repository is modified, and no validation or
locked-test snapshots are read.

The initial task remains optimized CEDAR `w1p25_l1`, q8 -> q4, with the
published MS_SR manifest SHA-256
`993f791f88d801e44e5feb0cee927c4c48eaa459bd6171e9e364b002bb01fdca`.
The default audit selects **all 44 eligible teacher records**, including
both ReT060 and ReT100 cases. Selection is sealed by an ordered SHA-256
digest of the source/operator identities.

## Definitions

For each teacher snapshot `s` and conservative component `k`, define
the full-grid coarse mean `mu_c,s,k`, within-snapshot population variance
`v_c,s,k`, fine residual mean `mu_R,s,k`, residual variance `v_R,s,k`,
and mean-square `p_R,s,k`.

For a collection `S` of source snapshots, the reported statistics are:

    mu_c,k = mean_s mu_c,s,k
    std_c,k^2 = mean_s [v_c,s,k + (mu_c,s,k - mu_c,k)^2]
    mu_R,k = mean_s mu_R,s,k
    std_R,k^2 = mean_s [v_R,s,k + (mu_R,s,k - mu_R,k)^2]
    rms_R,k^2 = mean_s p_R,s,k

Each **independent snapshot** receives equal weight regardless of
resolution and cell count. A case with more selected snapshots receives
more weight in the pooled result; case-balanced averaging is NOT used.
All reductions use float64, with the original MS_SR loader, full-domain
TorchCEDAR and the existing M48 operator qualification. Each selected
source is read once. No CEDAR operation is applied on a cropped patch.

For every case and each ReT group, compare against the pooled teacher
statistics using:

    mean_shift_z = (mu_c,case - mu_c,pool) / std_c,pool
    coarse_std_ratio = std_c,case / std_c,pool
    residual_rms_ratio = rms_R,case / rms_R,pool
    residual_mean_over_global_rms = mu_R,case / rms_R,pool
    residual_mean_over_case_rms = mu_R,case / rms_R,case

The physical conservative fields are `rho`, `rhou`, `rhov`, `rhow`,
`rhoE`, **as stored by MS_SR**. No new case-specific physical
nondimensionalization occurs in this audit.

## Why the residual mean is a diagnostic, not a fitted offset

Let `C` denote the complete fine-to-coarse CEDAR restriction.
With `R = Uf - P Uc` and `CP = I`, `C R = 0`.
Multiplication of a residual component by a fixed nonzero scalar
preserves this identity because `C` is linear. Subtracting a nonzero
constant from the fine residual generally does not preserve it when
`C` preserves constant fields.

The current training target stays `R / rms_R`, without mean subtraction.
The audit measures spatial residual means (and their ratio to RMS) to
determine whether they are numerically negligible. This does not assume
that `C R = 0` alone implies a zero global spatial mean; the latter
requires additional mean-conservation properties of `C`.

## Output

From the Calypso checkout:

    cd /scratch/coop/theret/GraphAttention
    git pull --rebase
    bash -n scripts/slurm/m48_normalization_audit.slurm
    sbatch scripts/slurm/m48_normalization_audit.slurm

Default output:

    /scratch/coop/theret/GraphAttention_runs/m48_normalization_audit/job_<JOB_ID>/
        normalization_audit.json
        case_comparison.csv
        snapshot_moments.csv

The JSON contains pooled, per-case, per-ReT-group and per-snapshot moments,
case-to-pool ratios, maximum C(R) defects, source identity seals,
repository and CEDAR provenance, and the extremes across cases.
The CSV files make it straightforward to plot and rank disparities
without loading CFD arrays on a login node.

## Acceptance and interpretation

- Pinned manifest, CEDAR and original source roles must verify.
- Exactly 44 teacher records must be selected under the specified filter
  and refinement (the count guard fails closed if this changes).
- All source loads and full CEDAR numerical checks must pass.
- Pooled coarse mean/std and residual RMS must match the M48.4a
  formula. Synthetic tests exercise equal snapshot weighting with
  unequal grid sizes, cross-snapshot variance, source-role exclusion,
  mixed filter rejection and source-provenance sealing.
- **No prespecified scientific threshold** determines whether a pooled
  global scaler is acceptable. Inspect mean shifts and scale ratios
  by physical case and channel; if large, evaluate physical reference
  scaling before changing the training normalization. A measured
  difference in dimensional amplitudes is not by itself a flaw.
- The process uses a known Grace GPU allocation solely for a tested
  Singularity runtime, but the audit computations are on CPU.
  Runtime and I/O costs for the 44-source audit remain unmeasured.

This work is a diagnostic extension; it does not change the numerical
training convention or claim to have qualified a newly trained model.
