# M43-B1b — Non-reentrant activation checkpointing in M37

## Motivation and previous real-GH200 evidence

The M43-B1 Grace training-memory gate (calypso-grace03, Slurm
471112, code revision 6558dc3a) established that complete
full-graph FP32 ten-layer h128/h4 DiNAT-DiT Flow Matching
AdamW training **fits at 65^3 but not 129^3** on a single
Grace GH200 with 95 GiB CUDA HBM.

| Real dataset | Nodes | One-hop / exact-2-hop edges | Result |
|---|---:|---:|---|
| Re54/Ma0.3 65^3 | 274,625 | 1.62M / 4.79M | 2 optimizer steps, GPU peak allocated 41.416 GiB |
| Re94/Ma0.3 129^3 | 2,146,689 | 12.78M / 38.04M | CUDA OOM in first forward, reported GPU peak allocated 89.330 GiB |

The first attempted additional allocation at 129^3
was 4.10 GiB when the device reported only 4.68 GiB free.
That does not establish the complete backward memory
requirement. CPU graph construction had already succeeded.

## Exactly what changes

Model: src/graph_attention/models/dinat_dit.py
Factory: src/graph_attention/training/model_factory.py
Config: configs/model/dinat_dit.yaml

New boolean option: model.activation_checkpointing,
**default false**. Only for AlternatingDilatedGeometricDiT.

When training with gradients enabled, each existing
M37 DiT block's forward computation is wrapped in
torch.utils.checkpoint.checkpoint(use_reentrant=False,
preserve_rng_state=True). The block's hidden state,
adaLN time-conditioning tensor and geometric displacement
are passed as explicit inputs to the checkpointed
function. The exact one-hop or exact-two-hop topology
and sparse CSR remain the same across recomputation.
Each block and its edges/topology are frozen into
a local callable to prevent Python's late-bound loop
closure selecting an incorrect block on backward.

There are **no parameter, layer, sparse-kernel, loss,
input, graph or trained-weight format changes**.
The activation checkpoint property is a runtime
execution switch, not a state_dict entry.
Non-reentrant checkpointing supports proper autograd
through both the hidden state and the time/coordinate
conditioning paths. Dropout RNG state is preserved.

During eval, inference or torch.no_grad(), the model
always bypasses checkpointing. Existing default
training and all M41 no-grad inference remain unchanged.
GPU custom sparse attention still requires FP32.

An opt-in model.set_block_memory_observer callback
runs after each ORIGINAL forward block, outside
the checkpointed function. It is never invoked by
backward activation recomputation. The probe callback
synchronizes CUDA, records live/peak PyTorch allocated
and reserved memory, and writes incremental JSON for
each of the ten blocks. This is a **diagnostic,
synchronized** run: it is not a clean throughput
benchmark and the cumulative peak is not the exact
memory footprint of an individual attention kernel.
If forward OOM occurs mid-block, the last completed
block and final allocator state are still recorded.

## Validation stages

### Gate 1: numerical / automatic differentiation equivalence

tests/unit/test_m43_b1b_checkpointing.py runs small
irregular sparse 3D graphs under both scatter and
torch_sparse backends and with three self-edge policies:
none, local only, and both. It tests:

- Same source weights / state dict (no new parameters).
- Outputs of uncheckpointed and checkpointed training
  with identical initialization, inputs and RNG.
- Gradient parity for ALL named parameters, including
  learned geometry bias, QKV, MLP and adaLN, plus input,
  coordinate and time-conditioning derivatives.
- Dropout preservation and TWO successive AdamW
  optimizer updates (matching updated weights).
- Per-block observer is invoked exactly once per
  original forward, not a second time during backward.
- Checkpointing bypassed in eval/no_grad modes.
- M37 reference sparse-cache unit regressions.

Small-graph tolerance: FP32 outputs 2e-6
absolute/relative, gradients 5e-5 absolute and
5e-4 relative, optimizer-updated weights 5e-6
absolute and 5e-5 relative. These are exact-path
regression tolerances, not new numerical equations.
Tests must PASS on Grace before starting large probes.

### Gate 2: real 65^3 / 129^3 full-graph comparison

Exactly four fresh CUDA subprocesses, in order:

1. 65^3 baseline M37, 2 full training steps
2. 65^3 checkpointed M37, 2 full training steps
3. 129^3 baseline M37, expected forward CUDA OOM
4. 129^3 checkpointed M37, 2 full training steps ATTEMPTED

All run on the SAME Grace allocation, select the middle
real HDF5 snapshot, use the same random seed, ten-layer
M37 configuration, FP32, batch one and probe-only
single-snapshot standardization. HDF5 reading, CPU
topology build, GPU H2D, forward, backward, optimizer,
peak CPU RSS, PyTorch CUDA allocated/reserved and
post-block memory are recorded separately.
No checkpoint weights or long training are created.

65^3 baseline and checkpointed per-step loss values
must agree within atol=rtol=2e-4. The CPU small-graph
gradient parity is the stronger numerical gate;
the real 65^3 loss check is only an extra consistency
check. Reporting 129^3 checkpoint as passed requires
TWO completed backward plus AdamW optimizer steps,
not just a successful forward.

A baseline 129^3 CUDA OOM is an accepted reference
outcome, not a reason to skip the checkpointed 129^3
test. If baseline 129^3 unexpectedly fits, record it
rather than treating it as a defect.
If the checkpointed 129^3 job still fails, its
partial per-layer profile will narrow the failing stage.

### Grace launcher

~~~bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch scripts/slurm/m43_b1b_activation_checkpointing.slurm
~~~

Grace, one GPU, four CPUs, four hours, NO --mem cap.
Uses the existing Singularity image pyg25.03.sif and
avbpML_env Python on Calypso. Requires a clean checkout.

Default output root:
 /scratch/coop/theret/GraphAttention_runs/m43_b1b_activation_checkpoint_gate/

Files:
  signature.json
  preflight.json
  progress.json
  summary.json
  runs/Re54_Ma0.3_65/baseline/report.json
  runs/Re54_Ma0.3_65/checkpointed/report.json
  runs/Re94_Ma0.3_129/baseline/report.json
  runs/Re94_Ma0.3_129/checkpointed/report.json

Each child also has a process.log. JSON reports are
incremental: failures retain previous per-block
measurements and allocator state. A process killed
before JSON completion is marked process_failed
by the parent when it survives.

An exclusive lock avoids concurrent jobs mixing
outputs. Terminal trial reports can be reused on
restart only with matching Git revision, case,
snapshot, mode, seed and step count.

Overall status checkpoint_passed means two actual
steps passed at BOTH checkpointed resolutions,
65^3 baseline passed, 65^3 diagnostic losses
matched, and 129^3 baseline produced either a
legitimate pass or a CUDA OOM. Otherwise the gate
writes checkpoint_failed and exits nonzero.

## Limits and next decision

Activation checkpointing trades additional backward
recomputation for reduced retention of activations
between layers. It cannot reduce the scratch memory
required by a single attention operation, which may
remain substantial for ~38M exact-two-hop edges.
No guaranteed memory reduction or 129^3 success
is claimed before testing.

If 129^3 passes, this establishes the MEMORY
prerequisite for subsequent scientific Experiment B
across the 600 Re54/Re94 snapshots. That later
training still requires case-specific physical
nondimensionalization, train-only sample-balanced
standardization, time/trajectory leakage-safe
splits, and per-resolution metrics. It is not
started automatically.

If 129^3 still fails, examine live/peak
per-block observations and whether failure moves
to backward. Possible next steps: use larger
activation checkpointing scopes, identify any
single-block workspace bottleneck, or implement
exact-halo patch training with verified backward
equivalence. Do not assume M41.6 inference-only
CPU streaming can backpropagate.

**Status (2026-10-19): Checkpointed 65^3 and 129^3 each completed two full training steps on Grace. Baseline 129^3 reproduced CUDA OOM. Full measured results follow; scientific multi-case training remains not started.**


## Measured M43-B1b results — 2026-10-19

Source: user-uploaded summary(20261019-095457).json.
Status checkpoint_passed. All four subprocesses on
calypso-grace04.calypso.cerfacs.fr, Slurm job 471143,
git b1615f7a1a31bbd7c28e1b817f63c663c94f30e3.
HDF5 snapshots: Re54/Ma0.3 and Re94/Ma0.3,
both solut_hit_00000051.h5. Hardware: NVIDIA GH200
with **95 GiB CUDA HBM**, regardless of the "480GB"
in the device name. M37 h128/l10/h4, 3,051,845
parameters, FP32 torch_sparse, one physical sample
per optimizer step, 2 true AdamW updates per mode.

| Measure | 65^3 baseline | 65^3 checkpointed | 129^3 baseline | 129^3 checkpointed |
|---|---:|---:|---:|---:|
| Status | ok | ok | CUDA OOM during forward_1 | ok |
| Completed optimizer steps | 2 | 2 | 0 | 2 |
| Peak GPU PyTorch allocated (GiB) | 41.41574 | **6.87797** | 89.32964 before OOM | **53.16558** |
| Peak GPU reserved (GiB) | 43.03125 | 7.41211 | 91.93164 before OOM | 58.29688 |
| CPU process peak RSS (GiB) | 1.89313 | 1.90845 | 7.82129 | 7.82355 |
| CPU graph build (s) | 5.11687 | 4.83919 | 39.12378 | 38.70553 |
| Step 2 forward (s) | 0.15948 | 0.16025 | — | 1.20439 |
| Step 2 backward (s) | 2.42607 | 2.56301 | — | 20.06284 |
| Step 2 optimizer (s) | 0.00154 | 0.00148 | — | 0.00169 |
| Step 2 total (s) | 2.58769 | 2.72529 | — | 21.26953 |

### Key quantitative findings

- At 65^3, non-reentrant checkpointing reduced GPU
  PyTorch allocated peak by **83.39%** (41.416 -> 6.878 GiB),
  peak reserved by **82.78%** (43.031 -> 7.412 GiB).
  Warm second-step time increased only **5.32%**
  (2.58769 -> 2.72529 s), though per-block instrumentation
  synchronizes CUDA and timing is diagnostic.
- On 129^3, baseline live allocated reached **36.48 GiB
  after block 1** and **68.64 GiB after block 2**, then
  OOM occurred in block 3 before the first forward
  completed. Peak after failure was 89.33 GiB allocated;
  this is NOT a complete baseline training peak.
- Checkpointed 129^3 **completed all 10 blocks and two
  complete forward/backward/AdamW updates**, at 53.166 GiB
  peak allocated and 58.297 GiB peak reserved.
  Approximately **36.70 GiB** of the 95 GiB device capacity
  remained above the recorded reserved peak; this is
  capacity arithmetic, not a guarantee for larger batches.
- Checkpointed 129^3 after-block live allocated increased
  from **7.60 GiB after block 1** to **16.82 GiB after
  block 10** in step 1. Peak observed through forward
  reached 33.196 GiB, while complete train-step peak
  reached 53.166 GiB. Backward/recomputation, not only
  the end-of-forward live state, must be budgeted.
- At checkpointed 129^3, warm forward 1.20439 s,
  backward 20.06284 s, total 21.26953 s. The backward
  stage dominates the full step (~94.3%). There is no
  valid baseline 129^3 complete-step time, so no direct
  checkpointing overhead ratio can be reported at 129^3.
- The diagnostic 65^3 Flow Matching loss values were
  IDENTICAL AS RECORDED IN THE JSON at the two measured steps:
  2.0007710456848145 and 1.9952092170715332 for both
  baseline and checkpointed. The paired gate reported
  baseline_65_loss_check=passed. Actual per-parameter
  gradient equivalence is covered by the separate
  small-graph unit test suite and cannot be inferred
  from two scalar losses alone.
- No full 600-snapshot training, scientific validation,
  conditional generation or SGS prediction has been
  conducted in M43-B1b. Probe-only per-snapshot z-scoring
  is still unsuitable for the actual experiment.

### Consequences for multiresolution Experiment B

**The training-memory gate has passed.** A single M37
architecture with activation checkpointing can now
execute full training steps on BOTH confirmed 65^3
and 129^3 graph resolutions on one Grace GH200.

The next task should be an ACTUAL controlled mixed-case
training implementation, not additional GPU-memory
optimization unless training later reveals a new issue.
Before starting:

1. Define the six-case dataset manifest (Re54/Re94
   with Ma 0.05/0.2/0.3, total 600 snapshots) and
   correct per-case physical reference scales. Do NOT
   inherit diagnostic per-snapshot z-score statistics.
2. Create temporal/trajectory-aware train/validation/test
   partitions with no correlated time leakage; explicitly
   distinguish physics generalization from mesh
   generalization because Re and resolution are confounded.
3. Use train-only, sample-balanced statistical scaling
   and ensure one consistent state representation across
   all cases/resolutions. Optionally introduce Re/Ma
   conditioning as a separately controlled intervention.
4. Mix microbatches at one graph each; checkpointing on;
   accumulate gradients by PHYSICAL SAMPLE (already in
   training infrastructure), track separate metrics per
   case and mesh resolution, and reuse static graph
   metadata across snapshots.
5. Run a short mixed-resolution training/smoke gate before
   a lengthy experiment. Inspect NaNs, gradient norms,
   reference-scale consistency, actual step throughput
   and validation per regime.

At measured step-2 rates, processing 300 snapshots of
65^3 and 300 of 129^3 would require approximately
2.00 GPU-hours of optimizer-step computation per
600-snapshot pass, EXCLUDING I/O, graph assembly,
statistical fitting, gradient accumulation effects
and validation. Treat this as an approximate planning
bound from one HDF5 snapshot in each mesh category,
not a measured full-epoch throughput. A 70% training
split with balanced case proportions would contain
420 snapshots, with ~1.4 hours of such measured step
time per training epoch before other work.

**Decision: proceed to the Experiment B dataset and
training-contract design, using activation checkpointing.
Do not launch the full 600-snapshot run until the
scientific preprocessing contract and smoke gate pass.**
