# M48.4e Phase A — GPU-only exact-halo TwoScaleDiNAT capacity benchmark

**Status: GPU benchmark pending; 511469/511470 were blocked by an overly restrictive metadata assumption, now corrected pending rerun.**
Do not infer the GPU-optimal batch size until a Grace run has been inspected.
This is a capacity benchmark, not a training experiment, a new dataset, or a
scientific model-quality evaluation.

## Objective and experiment scope

Determine the largest **useful** GPU workload, separately for static and dynamic
TwoScaleDiNAT, on both previously source-ordering-qualified fine-grid families
(48^3 and 80^3), across owned cubes 12^3, 16^3, 20^3, 24^3.

Sweep GPU microbatch 1, 2, 4, 8, 16, 32, 64, 128, 256 until a trial
hits CUDA OOM, the GPU reserved-memory headroom limit, or a conservative
host-memory packing guard. Each candidate runs in its **own CUDA process**.
Failures cannot poison the memory allocator in subsequent candidates.
Individual JSON reports are written as each candidate finishes, along with
a continuously updated `summary.json`, a final `trials.csv`, and exact-halo
geometry receipts. Refuse to overwrite an existing output directory.

## What is and is not executed

- Calls the **actual** M48 `TwoScaleDiNAT` model, including local/dilated
  geometric DiT attention, static coarse encoding, optional dynamic feedback,
  cross-scale interpolation attention, fine decoder, and owned-node MSE.
- Each measured iteration performs FP32 forward, backward, gradient clipping
  and an AdamW update. Warmup initializes CUDA kernels and optimizer buffers.
- Builds **actual** M48 target-driven periodic dependency closure, with one
  seam-crossing owned patch for each `(mode, Nfine, p, depth)` combination.
- For GPU capacity isolation, **replicates this same exact graph geometry**
  into disconnected components, with independent synthetic five-component
  input state, baseline, coarse state, target, and independent FM t per graph.
  All cross-graph edges are explicitly rejected. There is no artificial
  periodicity at patch faces.
- GPU model/optimizer and packed features/geometry stay resident during timed
  repeats. Timing excludes **all** CPU graph creation, packing/transfer,
  original DNS/HDF5 loading, global CEDAR noise projection, and checkpointing.
- **No raw or prepared CFD field data are read**. `MS_SR` teacher manifest and
  frozen normalization file are read and cryptographically verified solely to
  establish the two authorized grid sizes and physical periods. This protects
  data lineage but *does not* evaluate physical field or CEDAR numerical
  identities in this benchmark. Those were qualified separately in M48.4d/c.
- Training model width=64, depth=2 per subnetwork, heads=4 is an **engineering
  capacity candidate**, not an established final scientific architecture.
  Change `M48_WIDTH`, `M48_DEPTH`, `M48_HEADS` to retest the selected model.

A GPU batch is distinct from an **optimizer effective batch** in scientific
training. Every GPU capacity trial uses one AdamW update on its B synthetic
patch graphs. A later production decision on B will therefore change the
number of samples per optimizer step. No attempt is made here to preserve
M48.4c's effective batch of four during a B=64 forward/backward.

## Measurements

Per candidate:

- GPU name, total memory, initially free memory, peak allocated/reserved
  PyTorch memory and qualification against `max_gpu_fraction` (default 85%
  of initial free memory, bounded by total device memory).
- Model parameter count, number of owned fine nodes, fine/coarse read nodes,
  local/dilated and cross-scale edge counts.
- Synchronized per-update wall time (including Python launches and optimizer),
  CUDA event elapsed time, means/medians, warmup/repeat counts, and **owned
  fine nodes per measured update second**.
- Optional coarse NVML GPU/memory utilization samples (100 ms polling); both
  sample count and availability errors are recorded. For short trials these
  metrics are advisory and do not establish sustained GPU saturation.
- CPU geometry construction seconds and packed-graph transfer time recorded
  **separately** from GPU model throughput.

The fastest *within-headroom* trial is selected **per architecture, fine grid,
and patch size**. There is deliberately no single recommended patch size or
optimizer batch across physical resolutions at this stage. `status=INCOMPLETE`
means that at least one grid/core combination had no qualified candidate;
expected CUDA OOM is reported, not hidden. Unexpected worker crashes or
geometry failures cause `status=ERROR` and fail the SLURM postcheck.

The use of one representative patch topology (repeated with independent
features) gives an isolated **network capacity** comparison, but not the
cost of random spatial patch sampling. The geometry's coarse-grid parity
and seam position can affect edge counts; future end-to-end experiments must
use real random origins and independently projected full-domain CEDAR noises.

## Calypso execution

From an up-to-date GraphAttention checkout on Calypso:

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
bash -n scripts/slurm/m48_4e_gpu_capacity.slurm

sbatch --export=ALL,M48_MODE=static scripts/slurm/m48_4e_gpu_capacity.slurm
sbatch --export=ALL,M48_MODE=dynamic scripts/slurm/m48_4e_gpu_capacity.slurm
```

The two jobs run separately for isolated GPU memory measurements. By default,
CPU preparations are bounded to 8 GiB of estimated packed graph payload with a
3x temporary-allocation multiplier; a very large configuration may be skipped
rather than exhausting the SLURM host-memory allocation. The user can override
`M48_CORE_SIDES`, `M48_GPU_BATCHES`, `M48_FINE_SIDES`, `M48_WIDTH`, `M48_DEPTH`,
`M48_GPU_FRACTION`, `M48_HOST_PACK_GIB`, `M48_WARMUP`, `M48_MEASURE` to
run another bounded campaign. Use a **fresh output directory/job ID** each time.

Output:

```
/scratch/coop/theret/GraphAttention_runs/m48_4e_gpu_capacity/
    job_<JOBID>_<MODE>_w<WIDTH>_d<DEPTH>/
        summary.json
        trials.csv
        trials/<MODE>_n<GRID>_p<CORE>_b<B>.json
        trials/<MODE>_n<GRID>_p<CORE>_b<B>.log
        geometry/<MODE>_n<GRID>_p<CORE>.json
        geometry/<MODE>_n<GRID>_p<CORE>.pt
```

The `.pt` geometry templates are **synthetic benchmark graph geometry**, not
trained model checkpoints and not authorized real-source content.

## Acceptance criteria and next step

Phase A is complete only when both architecture modes have usable receipts for
both grid sizes and tested owned patch choices, when network throughput and
memory are compared fairly, and when no unexpected graph/worker failures exist.
A short measured sample and NVML polling cannot prove GPU utilization; inspect
large trials and increase measured repeats if rankings are unstable.

Then proceed to M48.4f (Phase B): independently projected CEDAR noises per
patch, realistic spatial origin sampling, CPU producer/prefetch queue, full
pipeline throughput, and a controlled optimizer batch-size/learning-rate
study. **Do not** treat the Phase-A fastest GPU graph batch as automatically
optimal for physical convergence or generative-field statistics.

## M48.4e correction after jobs 511469 and 511470

Both jobs passed their original CPU tests but failed in
`authentic_teacher_grids()` before the GPU sweep: teacher cases with the same
node grid have **different physical periodic lengths**. The previous benchmark
incorrectly required one universal physical length for each grid side.

The corrected metadata gate now validates every source's *own* fine/coarse
spacing and physical box, while allowing different physical domain lengths in
one 48^3 or 80^3 family. It preserves the first authenticated teacher record
as the **representative geometry for GPU-only timing**, and records min/max
lengths, all case-specific lengths and distinct length values in
`authenticated_grid_geometry` in `summary.json`. No across-case physical
training batches are created; the benchmark continues to pack replicas of
one physical geometry. The representative period is not asserted to be a
universal physical scale. Physical-size/coordinate generalization remains a
separate scientific question.

The launcher runs the metadata-only preflight *before* any CUDA worker.
To run it separately from the sweep (with a compatible Python environment and
`PYTHONPATH` including `GraphAttention/src`, the repo root and `MS_SR/code`):

```bash
python -m scripts.m48_4e_gpu_capacity --mode static --metadata-only \
  --ms-sr-root /scratch/ai/MS_SR \
  --normalization-file configs/m48/teacher_44_w1p25_l1_q4_q8.json
```

A small first GPU qualification after metadata and unit tests (use a
fresh SLURM job ID for each mode):

```bash
export M48_DEPTH=2 M48_CORE_SIDES=12 M48_GPU_BATCHES=1,2
export M48_WARMUP=1 M48_MEASURE=2
M48_MODE=static sbatch --export=ALL scripts/slurm/m48_4e_gpu_capacity.slurm
M48_MODE=dynamic sbatch --export=ALL scripts/slurm/m48_4e_gpu_capacity.slurm
```

For a subsequent depth-aware study, run depth 4 (or 6/8) as a separate job,
**after** the small pilot qualifies the CUDA path. Do not infer deep-network
capacity from depth-2 throughput or from only these two sample grid geometries.
