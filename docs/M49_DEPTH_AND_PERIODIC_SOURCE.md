# M49 short-first experimental campaign

Use a clean standalone checkout on Grace with /scratch paths and the
existing container. The periodic-source diagnostic and relative-only 3-D
training are independent. Do not restart an already-running relative-only
experiment or resume it on another Git revision.

## Immediate jobs

- scripts/slurm/m49_periodic_noise.slurm: two inference-only populations,
  epoch 1000, 180 draws per arm, Heun 50, original versus periodically tied
  Gaussian source, 2 h limit. M49_NOISE_RUN and M49_NOISE_EPOCH can select
  a trained relative-only checkpoint later.
- scripts/slurm/m49_dinat_relative_train.slurm: existing 10-layer relative-only
  3-D ablation (1000 epochs), 12 h limit. Original open 33^3 graph and iid source.
- scripts/slurm/m49_depth_benchmark.slurm: array indices 0..4, respectively
  10x12, 9x123, 8x123, 20x12, 15x123. 1 h per task; two GPUs maximum
  concurrent by default. Reports at m49/depth_benchmark/JOB_INDEX.json.

The depth benchmark measures the actual accumulated training step on a fixed
16-snapshot subset of the frozen training population, including FP32 Triton
backward, AdamW and GPU transfers. It tries B8/A2 and B4/A4, each with and
without activation checkpointing; B2/A8 only if no safe option. It reports
host collation/topology cost separately, peak allocated/reserved memory and
sample throughput. Selection requires <=85% allocated and <=90% reserved.
The estimate is not a guarantee of long-run memory or epoch duration.

## Dependent training

Submit m49_depth_train.slurm with M49_BENCH_ID set to the completed benchmark
array job ID. Array indices match the benchmark. The default 1-4 trains
all four nonbaseline variants; 1,3,4 prioritizes the proposed 9/20/15-layer
comparison. Index 2 is the optional radius-matched 8-layer control.
Index 0 can be selected explicitly for a new matched baseline if necessary.
The report must be successful and match the pinned Git revision, runtime
and hardware. Each job uses the fastest memory-qualified configuration,
effective batch 16, FP32, original mesh, iid Gaussian, absolute encoding.
The default first stage is 250 epochs with 25-epoch checkpoints. Resume
with the same report, checkout, variant, and M49_MODE=resume and a larger
cumulative M49_MAX_EPOCHS. All allocations are at most 12 hours.

Run m49_depth_eval.slurm with the same M49_BENCH_ID and array selection,
M49_EVAL_EPOCH=25/100/250/500/750/1000 as available. It uses Heun 50,
seed 5678, the original 180 validation references, and the qualified
microbatch. No test reference fields are accessed. Keep the separate
relative-only experiment on its original positional convention.

The periodic-source experiment produces job-tagged comparison and physical
metrics JSON under the original run's generations directory. Tied source
changes only the initial draw, not the ODE. It is an out-of-training-
distribution diagnostic, not a periodic model or a training change.

No Grace runtime results are claimed by these implementations.
