# M44 SR-0.1 — Snapshot-disjoint small-patch conditional Flow Matching

## Scope and scientific limitations

This is an **engineering and statistical-learning baseline**, not an LES
closure validation. It reads real AVBP DNS snapshots but forms a proxy coarse
condition by selecting every `s`-th DNS node, **without spatial filtering**.
Point sampling can alias unresolved DNS frequencies. The fine target is a
`(patch_cells+1)^3` DNS patch (default `9^3` nodes), and the coarse state is
`(patch_cells/s+1)^3` selected nodes (default `5^3`). The physical domain is a
small subvolume; we are **not** reducing the original 65^3 DNS to 5^3 nodes.
Full-domain `64^3 cells -> 32^3 cells` at `s=2` is a separate experiment.

**This run supports no claims about physical LES restriction, exact global
conservative consistency, compressible-state admissibility, independent
conditional diversity, or a-posteriori LES prediction.**

## Differences from M44 SR-0 one-snapshot pilot

- The original `scripts/m44_sr0_pilot.py` remains unchanged, including its
  **15-channel** model and existing absolute-coordinate embedding.
- New `scripts/train_m44_sr0.py` loads multiple chronological DNS snapshots
  using the existing `AVBPMultiCaseHITDataset` inventory and M3.3 case refs.
- Source snapshots (not spatial patches) are split into **chronological**
  train/validation/test blocks. An optional embargo excludes selected
  snapshots between those blocks. No physical time interval is inferred from
  snapshot filename counters; temporal decorrelation remains to be examined.
- The same deterministic, nonoverlapping-in-cell-interiors patch origins are
  selected for all snapshots. Patches are spatial crops **without periodic
  halo connections**, so finite-patch boundary artifacts remain possible.
- Only training snapshot patches determine global per-channel coarse means/stds
  and residual RMS scales. Held-out snapshots are never used for fitting.
- Each training patch is its own model sample; the recorded train-only moments
  use equal-patch weighting. There is no unintended per-validation-snapshot
  standardization.
- Default conditional input is **10 channels**:
  `[5 evolving residual channels, 5 interpolated coarse state channels]`.
  The optional `--condition-mode interpolated_plus_baseline` reproduces the
  original **15-channel** input; with simple trilinear `Ubase=T Uc`, its two
  condition groups are redundant.
- The new `--absolute-coordinate-embedding` flag enables the existing M37
  absolute-coordinate embedding. Its **absence** disables that additive term,
  while retaining geometric attention edge displacements from M3.3 coordinates.
  This is an optional M37 flag; the existing M37 default is **unchanged** and
  continues to use coordinate embedding.
- The M37 local/exact-two-hop attention, zero-initialized DiT conditioning,
  and M11 linear Gaussian Flow Matching training loss / Heun generation remain
  unchanged. The default sparse attention backend is `scatter`.

## Exact mathematical contract

For a DNS fine target `Uf`, define the proxy coarse state `Uc=Q_sample Uf` by
regular node selection. Given a deterministic trilinear interpolation `T`,

    Ubase = T Uc
    r = Uf - Ubase
    z = r / residual_rms_train
    c = T((Uc-coarse_mean_train)/coarse_std_train)

The default model is

    v_theta( [z_t, c], t, fine_edges, fine_coords ),

where `z_t=(1-t)epsilon+t z`, `v*=z-epsilon` and t is passed through the
existing adaLN-Zero time embedding. `z_t` has 5 channels, `c` has 5. All field
quantities and coordinates are first physically nondimensionalized with the
*same case-level M3.3 references*, including `L_ref`. There is no explicit
coarse/fine local-spacing channel in SR-0.1.

## Run on Grace (Calypso)

On the login node:

```bash
cd /scratch/coop/theret/GraphAttention
git switch main
git pull --rebase
sbatch scripts/slurm/m44_sr0_multisnapshot.slurm
```

The Slurm launcher runs CPU correctness gates then the numerical experiment
inside the established CERFACS Singularity/Python environment. Defaults:

| Item | Default |
|---|---|
| Case | Re54/Ma0.3 |
| DNS source grid | 65^3 nodes |
| Fine patch | 9^3 nodes |
| Coarse patch | 5^3 nodes |
| Refinement s | 2 |
| Selected snapshots | 24 from the full chronological inventory |
| Patches per selected snapshot | 2, fixed origins across snapshots |
| Temporal embargo | 1 selected snapshot at each split boundary |
| Train / validation / test | chronologically blocked, approx. 70 / 15 / 15 of non-embargo snapshots |
| Epochs | 5 |
| Model | 4 layers, hidden 64, 4 heads, scatter attention |
| Spatial conditioning | 5 interpolated coarse channels; model has 10 inputs |
| Absolute positional embedding | OFF (relative attention geometry remains ON) |
| Sampling | 50-step Heun at best validation epoch |

Output location by default:

    /scratch/coop/theret/GraphAttention_runs/m44_sr01/job_<SLURM_JOB_ID>/

Files: `manifest.json` (dataset/split/geometry/provenance), `scales.pt`
(training-only channel statistics), `history.json` (training and validation
losses), `last.pt` (resumable optimizer and RNG state), `best.pt` (best
validation epoch), `validation_sample.pt` (one held-out fine-field generation
and its coordinates and baseline), and `summary.json` (test loss and
conditioning diagnostics). Files are atomically replaced where applicable;
new runs refuse to overwrite an existing directory.

For a shorter/faster engineering test:

```bash
sbatch --export=ALL,M44_SR01_MAX_SNAPSHOTS=8,M44_SR01_PATCHES=1,M44_SR01_EPOCHS=2 \
  scripts/slurm/m44_sr0_multisnapshot.slurm
```

Resume **only** if the first job completed at least one epoch and `last.pt`
exists. Specify both the prior run directory and an increased total epoch
count. The runner verifies source snapshot sizes/mtimes, reference SHA256,
mesh metadata, split, origins, architecture and optimizer settings, then
restores optimizer state and the Flow Matching random generator:

```bash
sbatch --export=ALL,M44_SR01_RESUME=1,\
M44_SR01_OUTPUT=/scratch/coop/theret/GraphAttention_runs/m44_sr01/job_<OLD_JOB_ID>,\
M44_SR01_EPOCHS=10 \
  scripts/slurm/m44_sr0_multisnapshot.slurm
```

The run directory path must be an absolute path when exporting these values.
The source training snapshot set and model configuration must be identical
for resuming. An increase in total epochs is allowed.

## What to evaluate

1. Whether the held-out validation Flow Matching MSE falls below the
   zero-velocity reference near 2 and remains stable as training proceeds.
2. Held-out test FM loss **at best validation epoch**, not lowest test loss.
3. `conditional_sensitivity.mean_abs_velocity_change`: same noise and time,
   spatially permuted coarse fields. Nonzero sensitivity is necessary but
   **not sufficient** for physically useful conditioning; it is an out-of-
   distribution perturbation diagnostic, not quality validation.
4. Baseline point-sampled coarse agreement and per-field maximum/RMS errors
   for one generated sample. Generated coarse values are **not projected**
   to equal the coarse condition; no conservation claim is made.
5. Later: physically restricted LES pairs from `dnstoles`, turbulence spectra,
   statistics, condition-matched ensembles and positive density/pressure.

## Code and correctness gates

- `src/graph_attention/data/sr0_training.py`: snapshot selection, embargo
  split, deterministic patch origins.
- `src/graph_attention/data/sr0_pilot.py`: accepts optionally pre-qualified
  patch geometry; previous API and one-snapshot pilot are preserved.
- `src/graph_attention/tasks/sr0.py`: opt-in five-channel instead of
  ten-channel spatial conditioning. Default remains previous mode.
- `src/graph_attention/models/dinat_dit.py`: optional bypass of additive
  absolute coordinate embedding; relative displacements are unchanged.
- `src/graph_attention/training/model_factory.py`: transfers explicit
  embedding flag to the DiNAT model, defaulting to previous behavior.
- `scripts/train_m44_sr0.py`: self-contained experiment runner using existing
  model and task components.
- `tests/unit/test_m44_sr0_training.py`: split invariants and config flags.
- `tests/unit/test_m44_sr0_training_runner.py`: tiny synthetic AVBP HDF5
  archive, training, deterministic validation, checkpoint and resume.

The target implementation is a CPU/GPU executable multi-snapshot **numerical
baseline**. Report actual training results only after the Slurm job is run.
