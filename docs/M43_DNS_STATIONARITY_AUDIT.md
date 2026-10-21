# M43 DNS stationarity and conditional unresolved-scale audit

Status: implementation on isolated feature branch; **real GH200/DNS test not yet run**.
Original DNS, M43-B training, splits, normalizers and checkpoints untouched.

## Scientific questions

- Is the initial forced-HIT interval statistically nonstationary?
- How do unresolved DNS fluctuations evolve during that interval?
- Do unresolved statistics remain different after comparing local regions
  with comparable coarse-flow descriptors?

Matching a few descriptors is **not equivalent** to conditioning on the
complete, high-dimensional LES field. Failure to find common support is an
inconclusive result, not evidence of intrinsic regime dependence.

## Definitions and limitations

The audit reads rho, rhou, rhov, rhow, rhoE, then computes velocity.
It validates Cartesian coordinates, drops repeated periodic maximum planes,
and checks per-variable endpoint consistency before FFT analysis.

For unique periodic samples, the implemented block restriction R averages
factor^3 samples with equal discrete voxel weights. Piecewise-constant lifting
U satisfies R U = I, and the conservative residual q-U(Rq) satisfies
R delta_q = 0. This is **equal-voxel discrete conservation**, not yet a
demonstration of the AVBP finite-volume cell/dual-volume integral.

A separate spherical spectral cutoff on velocity uses
k_cut = min(pi/(factor*dx_axis)), where k has units rad / physical length.
The low/high velocity decomposition is orthogonal by Parseval;
its energy is not interchangeable with the conservative block-filter
residual energy.

Fine residual velocity uses the lifted coarse Favre velocity:
u_f - U(overline(rho*u) / overline(rho)).
The local residual velocity energy and one-step squared increments
within/across coarse-cell boundaries are computed per coarse block,
including periodic wrap.

**Lifting caution:** piecewise-constant prolongation can itself create
jumps in the residual at coarse-cell boundaries. A mismatch in residual
cross-boundary increments is not automatically a physical DNS seam.
Future studies should compare original fine velocity across the same
interfaces, and investigate smooth conservative lifting. The present
implementation is an auditable baseline estimator.

Local coarse descriptors: log(1+energy of local coarse velocity relative
to its global mean), log(1+strain magnitude), log(1+vorticity magnitude),
velocity divergence, and relative density. Local instantaneous energy
is not a temporal Reynolds turbulent kinetic energy.

The stationarity heuristic compares rolling means of kinetic energy,
u_rms and enstrophy to the final-window means. It yields only an
exploratory candidate interval: no time cutoff is chosen automatically.
The timestamp preflight inspects explicit HDF5 time metadata; snapshot
filename suffixes are *not* physical time, and unknown time units remain
unverified. Temperature, turbulent Mach, Taylor Reynolds and viscous
dissipation are left unavailable until EOS/viscosity provenance is audited.

For matching: work within each Re/Ma case and filter factor, establish
common support of log coarse energy and log strain, quantile-bin only
overlapping descriptor ranges, require at least 20 cells and 2 distinct
snapshots per regime in a shared bin, and reweight both regimes to the
same minimum shared bin probability. Reports include before/after
descriptor balance and retained sample fractions.

Bootstrap confidence intervals resample whole snapshots rather than
individual cells. They do not establish statistical independence across
neighboring time snapshots. No full conditional density claim is made.

## Running safely

Create a **separate worktree or clone**, retaining ff85188 in the active
M43-B training checkout if checkpoint resume is planned:

~~~bash
git fetch origin feature/m43-dns-stationarity-audit
git worktree add ../GraphAttention_dns_audit feature/m43-dns-stationarity-audit
~~~

Use the M43_AUDIT_REPO environment variable to point the Slurm launcher
to the audit worktree; do not change the checkout used by a resumable
M43-B training job. Run the following commands from that worktree:

~~~bash
cd /scratch/coop/theret/GraphAttention_dns_audit
export M43_AUDIT_REPO="$PWD"
~~~

Default: metadata-only preflight, no physical 3-D arrays loaded:

~~~bash
sbatch scripts/slurm/m43_dns_stationarity_audit.slurm
~~~

Run a two-snapshot single-case functionality gate in a NEW output directory:

~~~bash
sbatch --export=ALL,M43_AUDIT_MODE=case,M43_AUDIT_CASE=Re94/Ma0.2,M43_AUDIT_MAX_SNAPSHOTS=2,M43_AUDIT_OUTPUT=/scratch/coop/theret/GraphAttention_runs/m43_dns_audit_smoke scripts/slurm/m43_dns_stationarity_audit.slurm
~~~

Run the full single-case audit, separately:

~~~bash
sbatch --export=ALL,M43_AUDIT_MODE=case,M43_AUDIT_CASE=Re94/Ma0.2 scripts/slurm/m43_dns_stationarity_audit.slurm
~~~

Run the full six-case audit only with explicit confirmation:

~~~bash
sbatch --export=ALL,M43_AUDIT_MODE=all,M43_AUDIT_CONFIRM_FULL=YES scripts/slurm/m43_dns_stationarity_audit.slurm
~~~

After manual inspection, supply a cutoff JSON with zero-based snapshot
indices; for illustration ONLY (not scientific recommended values):

~~~json
{
  "Re94/Ma0.2": {
    "transient_end_index": 19,
    "stationary_start_index": 35
  }
}
~~~

Run the matched comparison without rereading DNS fields:

~~~bash
sbatch --export=ALL,M43_AUDIT_MODE=compare,M43_AUDIT_CASE=Re94/Ma0.2,M43_AUDIT_CUTOFFS=/scratch/coop/theret/reviewed_cutoffs.json scripts/slurm/m43_dns_stationarity_audit.slurm
~~~

Test the numerics:

~~~bash
python -m pytest -q tests/unit/test_m43_dns_stationarity.py
~~~

All Slurm jobs have a 12-hour limit. The CPU-only audit reserves one
Grace GPU for compatibility with the established container setup; it
does not perform GPU model training. Full six-case processing is never
launched automatically.

## Outputs

Root: /scratch/coop/theret/GraphAttention_runs/m43_dns_stationarity_audit

- preflight.json: full case/snapshot inventory, HDF5 shapes and timestamp probes
- cases/Re94_Ma0.2/source_signature.json: input/file/config/git hash
- cases/Re94_Ma0.2/snapshots/*.json: per-snapshot diagnostics
- cases/Re94_Ma0.2/spectra/*.npz: physical velocity energy spectra
- cases/Re94_Ma0.2/lag/*.npz: residual correlations and S2/S3/S4
- cases/Re94_Ma0.2/cells/*.npz: bounded sampled coarse features
- cases/Re94_Ma0.2/time_history.csv: diagnostic time series
- cases/Re94_Ma0.2/figures/*.pdf: spectra and time history
- cases/Re94_Ma0.2/summary.json: stationarity candidate, result status
- cases/Re94_Ma0.2/matched_factor_N.json: conditional matched comparisons
- cases/Re94_Ma0.2/scientific_report.md: limitations and interpretations

A completed snapshot is atomically logged and skipped on compatible
restarts. Stale input/option/Git signatures fail closed and must use a
new results directory; original DNS files are never written.

## Bottlenecks and next gates

HDF5 archive bandwidth, repeated 3-D FFTs and gradients, CPU memory
for 128^3 workspaces, trustworthy physical timestamps and time
autocorrelation, correct fine/coarse volume quadrature, and sufficient
common support with only ~100 snapshots per case.

After the real preflight, extend the report with verified EOS/viscosity,
residual-gradient distributions, spatially blocked uncertainty,
conditional distributional distances and a smooth conservative lifting
before using this audit to decide which snapshots to exclude.

**Do not remove transient snapshots from M43-B training on the basis
of the current heuristic alone.**
