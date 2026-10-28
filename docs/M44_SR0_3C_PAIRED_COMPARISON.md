# M44 SR-0.3c — Paired checkpoint comparison on free-node support

## Rationale

Both M44 SR-0.1 (unconstrained) and M44 SR-0.3 (intrinsically
point-consistent) completed 50 epochs at the same 9³ fine-node / 5³
coarse-node geometry with refinement factor 2. Both use a 10-channel,
four-block DiNAT-DiT and 50-step Heun generation. SR-0.3 removes the
125 observed coarse nodes from Gaussian noise and the FM loss, leaving
604 unobserved fine nodes. **Native FM MSE is therefore not comparable.**

The new `scripts/compare_m44_sr03_diagnostics.py` is a self-contained,
read-only JSON comparator for the two *post-SR-0.3b* checkpoint reports.
It **refuses** to silently substitute the unrestricted 729-node loss
when a report lacks `common_unobserved_node_support`.

## Inputs

On Calypso, collect these checkpoint-only read-only reports:

- Unconstrained: `/scratch/coop/theret/GraphAttention_runs/m44_sr02_diagnostics/job_484628/diagnostic_summary.json`
- Constrained: diagnostic report associated with SR-0.3 training job `484619`.

Both must have been evaluated with `M44_SR02_PATH_REPEATS=16`,
`M44_SR02_DRAWS=16`, and `M44_SR02_MAX_PATCHES=4`, using diagnostic
revision `71a9653` or later. The older uploaded unconstrained report
from revision `8fa90e7` lacks common-support metrics and is intentionally
rejected.

## Execute through grace Slurm + Singularity

**No Python is executed on the login node.** The standalone comparison
script and its `pytest` gate must both run through the configured
Singularity image inside a job on the `grace` partition.

The new launcher defaults to:

- Unconstrained diagnostic: `m44_sr02_diagnostics/job_484628/diagnostic_summary.json`.
- Constrained *training* run: `m44_sr01/job_484619`.
- Constrained diagnostic: automatically discovered by requiring an exact
  `source_run_dir` match to the training run. If zero or multiple reports
  match, the launcher fails and asks for an explicit report.
- Output: `m44_sr03c_paired_comparison/job_<SLURM_JOB_ID>/`, never overwritten.

Submit directly from the login node (only Git and Slurm commands):

```bash
cd /scratch/coop/theret/GraphAttention
git switch main
git pull --rebase
sbatch scripts/slurm/m44_sr03c_compare.slurm
```

If more than one diagnostic report refers to constrained job `484619`,
supply the exact chosen report:

```bash
sbatch --export=ALL,M44_SR03C_NEW_REPORT=/scratch/coop/theret/GraphAttention_runs/m44_sr02_diagnostics/job_<DIAGNOSTIC_JOB_ID>/diagnostic_summary.json \
  scripts/slurm/m44_sr03c_compare.slurm
```

Optional Slurm environment overrides:
`M44_SR03C_OLD_REPORT`, `M44_SR03C_NEW_REPORT`,
`M44_SR03C_CONSTRAINED_RUN`, and `M44_SR03C_OUTPUT`.
If comparing a different constrained training run, set
`M44_SR03C_CONSTRAINED_RUN` to its absolute source path alongside any
explicit report override.

The launcher checks clean Git status, source existence and identity,
prohibits reusing an existing output directory, logs input SHA-256 hashes,
runs `tests/unit/test_m44_sr03_common_support_comparison.py` in
Singularity, then runs the comparator in Singularity.
The resulting `comparison_summary.json` and `comparison_summary.md`
are written in the new job-specific directory. No checkpoint is modified.


The comparator checks source snapshot IDs and split counts, model
configuration, equal deterministic FM path counts, identical 604-node
free support, zero-velocity reference parity per snapshot and split,
ensemble sample IDs, number of Heun steps, draws, and DNS residual RMS.
It outputs `comparison_summary.json` and `comparison_summary.md` to a
new directory without touching either training run or checkpoint.

## Scientific interpretation

- Negative `constrained_minus_unconstrained_mse` means the constrained
  model fits the FM velocity target better **on the same free nodes**.
- The experiment still contrasts distinct trained protocols: in the
  unrestricted model, observed-node residual inputs are Gaussian; in
  the constrained model they are exactly zero.
- Compare residual amplitude calibration **by held-out snapshot and
  by conservative variable**. The 16-draw SR-0.3 report shows all
  conservative residual RMS ratios below one for the four test
  snapshots, especially `rhou`. Some validation channels overshoot
  their reference RMS. This is consistent with a generalization or
  amplitude-calibration problem but does not identify its cause.
- A single DNS target per coarse observation cannot validate a full
  conditional PDF. These 9³ patches cannot establish a reliable
  whole-domain kinetic-energy spectrum.
- The point-sampling constraint is **not** volume-integrated conservative
  LES restriction; `dnstoles` owns the physical coarse-graining task.
- Treat validation and train as exploratory diagnostics; do not tune
  architecture against the chronological test split.

## Correctness gate

The Slurm launcher executes the five comparator unit tests **inside the
same Singularity environment**, before invoking the JSON comparison. Do
not run these tests directly on a login node.
