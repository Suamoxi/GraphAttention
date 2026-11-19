# M48.4f — three-way exact-halo SR patch execution comparison

**Implementation status (2026-11-19):** committed to `main`; **CUDA/Calypso
qualification pending**. Do not report timings or numerical acceptance until
the actual Slurm logs, audit tensors and summary have been inspected.

## Why this experiment exists

M47 full-DNS graphs favored sequential gradient accumulation on one GH200:
- Two 65^3 real snapshots: packed segmented 0.876119 s/step vs sequential
  0.860268 s/step; peak allocated 13.878 vs 6.442 GiB.
- Heterogeneous 65^3 + 129^3: 3.917339 vs 3.819152 s/step;
  peak allocated 60.935 vs 48.140 GiB.
- Packed segmented significantly outperformed old packed indexing,
  and passed gradient/AdamW parity. These large-graph results should NOT
  be extrapolated directly to SR patches.

M48.4e depth-2, width-64, 12^3 owned-core capacity evidence:
- Static, 48^3, GPU batch 1 vs 2: 31,898 vs 56,664 owned nodes/s;
  peak allocated 0.92 vs 1.78 GiB.
- Dynamic, 48^3: 9,418 vs 16,406 owned nodes/s;
  peak allocated 3.56 vs 7.01 GiB.
- These short measurements show packing is promising, but M48.4c's
  GPU microbatch 1 used packed batch indices even for ONE graph,
  so it was not an optimized sequential baseline.
- M48 packed models also did NOT propagate the optional M47
  expand+cat segmented-conditioning path into both fine/coarse DiNAT.

The experiment therefore distinguishes **(1) packing effectiveness**
from **(2) graph-conditioning broadcast efficiency**.

## Three arms with identical physical/optimizer objective

| Arm | SR patch execution | Fine/coarse DiT batch handling |
| --- | --- | --- |
| sequential_single | 4 independent forward/backward calls | no graph batch indices; single-graph expand |
| packed_index | one disconnected 4-patch forward/backward | original indexing with fine/coarse batch_index |
| packed_segmented | one disconnected 4-patch forward/backward | opt-in tuple(graph_sizes), fine AND coarse, expanded features then concatenated |

Each arm consumes the same four real-CEDAR-projected patch samples for each
optimizer step. The source is one authenticated 48^3 or 80^3 *teacher*
snapshot; never validation or locked-test data. Static and dynamic are
separate experiments. Fine residual Flow Matching state, projected-noise
source, FM time per patch, patch location, M48 exact two-scale dependency
closure, frozen 44-teacher normalization, model initialization and AdamW
configuration are held fixed. No GPU stochastic data augmentation.

Owned fine-core supervision only, including patch overlap duplicates as
independent draws with independent full-domain projected Gaussian sources.
The effective optimizer objective is:

    L_update = (L_patch1 + L_patch2 + L_patch3 + L_patch4) / 4
    L_patch = mean over owned fine nodes and five conservative components
              of (predicted FM velocity - projected FM target)^2

For all arms: one optimizer.zero_grad(), sum of scaled gradients, one
clip_grad_norm_(max_norm=1.0), one AdamW(lr=3e-4) optimizer.step().
TwoScaleDiNAT architecture and original CEDAR operators remain unchanged.

## Production/safety boundary

- New source: `scripts/m48_4f_patch_execution.py`. Implements the opt-in
  single-patch fast path with `batch_index=None` and the two packed paths;
  reuses the existing `pack_patches` for packed modes. No duplicate
  graph/physics preprocessing.
- `src/graph_attention/models/two_scale_dinat.py`: propagates optional
  `fine_graph_sizes` and `coarse_graph_sizes` into fine encoder, decoder,
  static coarse encoder, and dynamic coarse encoder. The existing
  `AlternatingDilatedGeometricDiT` validates sizes and implements the
  optional segmented DiT graph-feature broadcast. Default behavior is
  unchanged.
- `scripts/m48_4f_compare.py`: driver runs each execution mode in
  independent fresh CUDA subprocesses, serially, on two teacher grid sizes;
  uses `_prepare_pair` (M48.4b pinned CEDAR), `make_patch` (M48.4c
  full-domain source projection + exact halos), frozen normalization, and
  existing M47 full parameter-gradient/AdamW auditor.
- `tests/unit/test_m48_4f_patch_execution.py`: CPU/CUDA static/dynamic
  parity tests for owned outputs, full gradients and optimizer weights.
- Original M48.4b/M48.4c scientific trainer, checksums, modes and defaults
  are NOT modified.

## Measurement and acceptance protocol

Default static and dynamic runs separately on one GH200:
`width=64, heads=4, depth=2, core_side=12, effective_batch=4,
steps_per_grid=4`, teacher representatives from 48^3 and 80^3. Step 1
on each grid is a warmup and full audit; reported medians use steps 2-4.
Each worker records a first-step hash of model weights and all four patch
input/target/geometry tensors, individual forward/backward + transfers +
clip/optimizer times, full GPU update time, preparation time (includes CPU
CEDAR full-domain projections), peak GPU allocated/reserved bytes, loss
and owned-node throughput. Snapshot HDF5 load is reported separately
and excluded from per-step timing.

The driver requires:
- All three worker processes exit normally and produce full JSON reports.
- Exact starting-weight and full-input/geometry/provenance hashes.
- First-step owned-output allclose(atol=1e-4,rtol=1e-3).
- Complete first-step parameter gradient audit
  (atol=5e-4, rtol=5e-3), plus post-AdamW parameter audit
  (atol=3e-6, rtol=2e-4).
- Every per-step mean loss within 1e-4 of the optimized sequential arm.
- Direct packed indexed/segmented gradient and optimizer-update audit.
- Both 48^3 and 80^3 sources pass. Failures are reported but never
  silently reclassified as acceptable speed results.

Float32 PyTorch/CSR reduction scheduling can produce finite small
differences. Avoid bitwise-parity requirements for separate CUDA paths.
The statistical acceptability of model-generated SR fields is NOT tested.

## Run on Calypso Grace

First ensure your checkout has no local modifications:

```bash
cd /scratch/coop/theret/GraphAttention
git status --short
git pull --ff-only origin main
git rev-parse --short HEAD
bash -n scripts/slurm/m48_4f_three_mode_compare.slurm
```

Launch the static and dynamic tests separately:

```bash
sbatch --export=ALL,M48_MODE=static \
  scripts/slurm/m48_4f_three_mode_compare.slurm
sbatch --export=ALL,M48_MODE=dynamic \
  scripts/slurm/m48_4f_three_mode_compare.slurm
```

The Slurm stdout/stderr are written to:

    /scratch/coop/theret/GraphAttention/m48f_three_mode_<JOBID>.out
    /scratch/coop/theret/GraphAttention/m48f_three_mode_<JOBID>.err

Each job outputs:

    /scratch/coop/theret/GraphAttention_runs/m48_4f_three_mode_v1/
      job_<JOBID>_<MODE>/
        manifest.json
        progress.json
        summary.json
        sequential_single/
          report.json
          process.log
          audits/grid_48.pt
          audits/grid_80.pt
        packed_index/
          report.json
          process.log
          audits/grid_48.pt
          audits/grid_80.pt
        packed_segmented/
          report.json
          process.log
          audits/grid_48.pt
          audits/grid_80.pt

If tests fail at the Slurm preflight, the benchmark will not start:
share stdout/stderr. If a worker fails, the driver still records
progress and a final `failed_or_review_required` summary; share the
corresponding process.log, report.json and any surviving audit files.

## Decision to make after valid measurements

1. Is optimized SR packed segmentation faster than optimized single-patch
   accumulation after including CEDAR and exact-halo geometry costs?
2. Does the gain survive dynamic cross-scale attention at 48^3 and 80^3?
3. How much GPU headroom remains for increasing depth, owned core size
   or GPU patch microbatch beyond four?
4. Depending on this result, benchmark additional core/depth/batch sizes
   and compare full-field vs tiled execution using M48.4e A2.

**Do not change the scientific training default based on a source-only
implementation or an unvalidated short probe.**
