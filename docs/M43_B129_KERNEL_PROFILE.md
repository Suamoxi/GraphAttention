# M43-B129 — Unchanged M37 GPU kernel profiling gate

**Status:** profiling infrastructure written on isolated
`feature/m43-b-129-kernel-profile` branch. No GH200 profile has been
executed and no source training job resumed. Numeric model code unchanged.

## Why this before a new kernel?

The existing M43-B1b training gate showed that the 129^3, 10-layer,
hidden-128, FP32 torch_sparse M37 model has approximately:
- 1.20 s forward, 20.06 s backward, 21.27 s full step;
- 53.17 GiB peak allocated with non-reentrant block checkpointing;
- 12.78 million directed one-hop and 38.04 million exact-two-hop edges.

The expensive stage is backward plus checkpoint reforward, but this does
not yet tell us whether GPU time goes to sampled_addmm, sparse softmax,
SpMM/index_add, MLP matmul, kernel-launch overhead or graph metadata.

No speculative fused or Triton kernels are introduced in this branch.

## What is measured?

`scripts/profile_m43_b_129_kernels.py`:

1. Read the existing scientific `run_signature.json`, `last.pt`
   and train-only `standardizers.pt` from the training directory.
2. Recreate the existing M43-B model with `build_recipe(checkpointing=True)`:
   M37 10-layer, hidden=128, heads=4, physical nondimensionalization,
   geometric one-hop/exact-two-hop, FP32 torch_sparse, Flow Matching.
3. Choose one actual 129^3 snapshot by basename, initially
   `Re94/Ma0.2/solut_hit_00000051.h5`.
4. Use the existing `GraphTaskCollator`, node-weighted M6 MSE,
   `_one_batch`, `_forward`, and `load_training_checkpoint` paths.
5. Warm up one complete 129^3 optimizer update without profiler overhead.
6. Reload model/AdamW/training generator from original checkpoint so the
   measured step starts at its checkpoint state.
7. Profile one full physical-sample train step with CPU+CUPTI CUDA
   activities, record_function phase scopes, and default accuracy.
8. Collect phase times, CUDA kernel duration/count by launching scope,
   top PyTorch operators, GPU memory peak and Chrome trace.

The old checkpoint and input fields are opened for reading only. In-memory
training updates are discarded and no optimizer/weight/history files are
written. This is NOT the complete six-case optimizer update or a full
training epoch. It isolates the expensive 129^3 component and preserves
its per-node forward/backward computation.

The profiler source branch is derived from the scientific M43-B
`9440788253dc61f359a94ba3c68f50d6a7a6cb7d` commit. A Slurm
Git-diff guard rejects any change outside the profiling scripts/tests/docs.

## Results

The profile output directory (separate from scientific training) contains:

- `input_provenance.json`: original signature, checkpoint epoch, source
  snapshot ID, model/graph sizes, GPU and Python/PyTorch versions.
- `profile_report.json`: synchronized wall-clock times; unprofiled warm
  step and profiled step; exclusive PyTorch operator CUDA activity;
  CUDA kernel names, launch counts and time per phase.
- `top_cuda_kernels.txt`: ranked GPU kernel names with per-phase duration.
- `top_pytorch_operators.txt`: ranked PyTorch operators by exclusive
  device time.
- `cuda_trace.json` (optional): Chrome/Perfetto CPU/GPU trace.

Important: summed CUDA kernel durations, CPU operator time and wall
seconds are distinct metrics. A kernel profiler changes timings and can
consume substantial memory. Compare the measured step against the
non-profiled warm step, not against absolute performance expectations.
Trace writing can take space and time, so use a fresh scratch directory.
`--skip-trace` avoids materializing the Chrome JSON file but does not
disable profiling overhead.

## Running on Calypso without changing the training checkout

Create a separate worktree, leaving the original 8-epoch M43-B training
run and its Git revision/checkpoint intact:

~~~bash
cd /scratch/coop/theret/GraphAttention
git fetch origin
git worktree add --detach ../GraphAttention_m43_profile \
  origin/feature/m43-b-129-kernel-profile
~~~

Run metadata-only inspection (CPU correctness tests, then signature checks):

~~~bash
cd /scratch/coop/theret/GraphAttention_m43_profile
sbatch --export=ALL,M43_PROFILE_REPO="$PWD" \
  scripts/slurm/m43_b129_kernel_profile.slurm
~~~

When the correctness tests and inspection pass, submit the CUDA run:

~~~bash
sbatch --export=ALL,M43_PROFILE_REPO="$PWD",M43_PROFILE_MODE=profile,M43_PROFILE_CONFIRM=YES \
  scripts/slurm/m43_b129_kernel_profile.slurm
~~~

The output directory defaults to:

`/scratch/coop/theret/GraphAttention_runs/m43_b129_cuda_profile_JOBID/`

Expected resources: single Grace GH200 GPU with sufficient VRAM for the
129^3 checkpointed graph plus CUPTI profiling overhead, 4 CPUs, 2 hours.
The old M43-B training run is not submitted or modified.

If high-volume traces are too large, set
`M43_PROFILE_SAVE_CHROME_TRACE=NO`; operator/kernel tables are still
generated.

### Interpretation gate before kernel engineering

Only propose kernel changes after inspecting:

1. Which backward kernels dominate GPU time? Quantify the fraction for
   sampled_addmm/SDDMM, SpMM/index_add and scatter softmax.
2. What fraction of backward is repeated forward work from activation
   checkpointing?
3. How much CPU overhead comes from graph transfer and CSR sorting?
4. Do GPU time sums and synchronized wall time reveal a launch-bound
   or memory-bandwidth/atomic-contention-bound workload?
5. Does one kernel dominate, or do many small kernels each contribute?
6. What peak device memory does profiling add over the warmup?
7. Which next benchmark can isolate one proposed change while preserving
   the M37 operator, attention edges, loss and optimizer equivalence?

It is acceptable if the profile can identify slow *families* of kernels
without proving a particular hardware bottleneck. Detailed memory bandwidth
and atomic contention may require Nsight Compute as a subsequent step.
