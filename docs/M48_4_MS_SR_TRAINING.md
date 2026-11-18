# M48.4a — MS_SR multisource Flow Matching training qualification

## Scope

M48.1–M48.3 passed on real MS_SR data in SLURM job 510080 at commit
cedfa5f8fd22c081ca49c8e09fba1b94b6ee7116.

M48.4a implements a bounded two-source training qualification only.
It is not a scientific training campaign or generative-validation result.
MS_SR is read-only. The locked test role is excluded. No CNN/CVAE baselines.

## New implementation

- src/graph_attention/data/ms_sr_training.py: select independent teacher
  sources; compute sealed teacher-only population coarse mean/std and residual
  RMS; verify normalization provenance.
- src/graph_attention/data/ms_sr_adapter.py: add exact case and snapshot ID
  selectors without changing existing smoke-test default behavior.
- scripts/m48_ms_sr_train.py: two-mode matched training steps with one CEDAR
  full-domain inverse, global Gaussian projection, exact graph dependency
  halos, and loss only on owned fine nodes.
- tests/unit/test_m48_ms_sr_training.py: synthetic regression tests of role
  separation, snapshot selection, population statistics and periodic cores.
- scripts/slurm/m48_4_train_gate.slurm: Grace GPU preflight, syntax,
  regression suite and bounded two-source training run.

## Normalization and flow

For each independent teacher snapshot s, use the full coarse field Uc_s
and full CEDAR residual R_s = Uf_s - P Uc_s.

    mu = mean_s mean_x(Uc_s)
    sigma_c^2 = mean_s [variance_x(Uc_s) + (mean_x(Uc_s)-mu)^2]
    sigma_R^2 = mean_s mean_x(R_s^2)

The snapshots receive equal weight, even when different resolutions have
different cell counts. No validation or locked records are used to fit
normalization.

For each step, project noise on the complete periodic fine lattice:

    X1 = R / sigma_R
    X0 = (I - P C) epsilon, epsilon ~ N(0,I)
    Xt = (1-t) X0 + t X1
    velocity_target = X1 - X0

Only then extract exact target-driven two-scale halos. The global CEDAR
operator is never applied on artificially periodized patches.

The current gate uses w1p25_l1, q8 -> q4, ReT060, two teacher records
from distinct cases, 2x2x2 owned fine nodes, width 16 and depth 1.
Both architectures see the same snapshot, FM time, projected noise
and owned core per step.

## Calypso execution

    cd /scratch/coop/theret/GraphAttention
    git pull --rebase
    bash -n scripts/slurm/m48_4_train_gate.slurm
    sbatch scripts/slurm/m48_4_train_gate.slurm

Expected output, under
/scratch/coop/theret/GraphAttention_runs/m48_4_train_gate/job_<JOB_ID>/:

    normalization.json
    summary.json
    static.pt
    dynamic.pt

The script refuses to overwrite a previous output path.
Checkpoints are engineering receipts, not converged models.

## Not yet qualified

New code must pass actual Calypso tests. Full scientific training needs
data streaming, source shuffling, validation, checkpoint resume, memory
and runtime profiling. Full-lattice assembled velocity prediction,
global projection during every numerical integration evaluation,
projected Heun sampling, turbulence spectra, spatial correlations and
conservative-state admissibility remain separate M48.4 acceptance gates.
