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

Example invocation:

```bash
cd /scratch/coop/theret/GraphAttention
git switch main
git pull --rebase

OLD=/scratch/coop/theret/GraphAttention_runs/m44_sr02_diagnostics/job_484628/diagnostic_summary.json
NEW=$(grep -l 'job_484619' /scratch/coop/theret/GraphAttention_runs/m44_sr02_diagnostics/job_*/diagnostic_summary.json | head -n 1)
test -s "$OLD" && test -s "$NEW"
python -m scripts.compare_m44_sr03_diagnostics \
  --unconstrained "$OLD" --constrained "$NEW" \
  --output-dir /scratch/coop/theret/GraphAttention_runs/m44_sr03_paired_comparison
```

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

## Gate

```bash
python -m pytest -q tests/unit/test_m44_sr03_common_support_comparison.py
```
