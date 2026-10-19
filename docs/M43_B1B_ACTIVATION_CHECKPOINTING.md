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

**Status: code and launch tests prepared. Grace B1b
performance and memory numbers are NOT yet available.**
