# Experimental GPU backend - CSR Triton backward

## Goal and status

**Opt-in experimental backend; synthetic GH200 benchmarks and a 65^3 real-HDF5 ten-layer training-step pilot completed. Scientific training acceptance remains pending.**
This investigation is separate from the frozen M47.1/M47.4 scientific training
continuations. It does not modify their checkpoints, numerical behavior,
input distributions or training recipes.

The new backend named 'torch_sparse_triton' uses the same original FP32 CSR
SDDMM forward, sparse softmax, and memory-bounded SpMM forward as
'torch_sparse'; only the backward reductions are changed. No BF16, periodic
topology modifications, sampling changes or other model changes are introduced.

Files:

- src/graph_attention/models/triton_sparse_backward.py
- src/graph_attention/models/sparse_transformer.py (opt-in dispatch only)
- src/graph_attention/models/dinat_dit.py (CSR cache reuse)
- tests/unit/test_triton_sparse_backward.py
- scripts/benchmark_triton_sparse_backward.py
- scripts/slurm/sparse_backward_triton_benchmark.slurm
- scripts/benchmark_triton_full_training.py (paired real-HDF5 10-layer probe)
- scripts/slurm/sparse_backward_triton_full_training.slurm

## Mathematical contract

For directed edges source-to-target, the attention CSR has row=target and
column=source. The transposed CSR reverses those roles. The per-head
formulas are, for score gradient D and output gradient G:

- dQ = D K / sqrt(head_dim), target CSR
- dK = D-transpose Q / sqrt(head_dim), transposed CSR
- dV = P-transpose G, transposed CSR
- dP_ij = dot(G_i,V_j), bounded edge/feature tiles

The existing PyTorch softmax backward calculates D and propagates gradients
through the learned geometric-bias network. The optimized backward never
allocates a full E-by-H-by-d message tensor and uses no global atomics.

Each Triton program processes a small number of CSR rows and a finite
edge-feature tile. It loops over the necessary number of edge tiles,
supporting arbitrary variable degree, isolated nodes, asymmetric directed
graphs, self-loops, and degrees larger than one tile. The algorithm makes
no Cartesian or regular-mesh assumption.

Initial tile sizes: 4 rows by 32 edges for row reductions and 64 edges for
per-edge dot products. These are correctness-first defaults, not tuned.

## Calypso invocation

From a clean main checkout:

~~~bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase origin main
git status --short
# First validate compilation and parity on the smallest graph
sbatch --array=0 scripts/slurm/sparse_backward_triton_benchmark.slurm

# Only if the first job succeeds, run the other topologies/resolutions
sbatch --array=1-3 scripts/slurm/sparse_backward_triton_benchmark.slurm
~~~

Array indices 0-3 are 65-cube one-hop, 65-cube exact-two-hop, 129-cube
one-hop and 129-cube exact-two-hop. Head count=4, head dimension=32, FP32.

The job checks that Triton/CUDA are available inside the deployed Python
environment, runs tests/unit/test_triton_sparse_backward.py and existing
DiNAT-DiT tests with deterministic CUDA reductions, then performs three
comparisons with unrestricted CUDA timing:

- isolated SDDMM backward,
- isolated SpMM backward,
- full geometric sparse-attention forward and backward.

Each comparison benchmarks 'torch_sparse' against 'torch_sparse_triton'.
Graph construction, CSR conversion and initial JIT compilation are outside
the timed region. Profile traces are separate and opt-in using
CSR_TRITON_PROFILE=YES; CSR_TRITON_REPEATS defaults to 3.

Output directory:

~~~text
/scratch/coop/theret/GraphAttention_runs/sparse_backward_triton_v1/
  65cube_one_hop_<jobid>/results.json
  65cube_two_hop_<jobid>/results.json
  129cube_one_hop_<jobid>/results.json
  129cube_two_hop_<jobid>/results.json
~~~

## Numerical acceptance gates

Unit tests compare original/new SDDMM/SpMM gradients on irregular unsorted
directed graphs including high-degree (49-edge) rows and isolated vertices.
They also check full geometric-attention outputs, input and weight gradients,
edge-displacement gradients, and a two-layer DiNAT model with activation
checkpointing.

Gradient thresholds on small examples start at rtol=3e-4..5e-4 and
atol=3e-5..5e-5. Forward thresholds are tighter. Different summation
orders are not bitwise equivalent. Investigate discrepancies instead of
simply loosening tolerances.

Initial GPU gate: all tests pass, memory does not increase unexpectedly,
and 65/129-cube tests complete without OOM. The required speedup is
a reduction in median backward and full-attention-step time, not only a
faster isolated kernel.

## Measured GH200 pilot (November 2026)

Synchronized unprofiled medians of three repetitions, synthetic Cartesian
graphs, FP32, four heads by 32 dimensions. All attention forwards unchanged.

| Graph | Nodes | Directed edges | Geometric-attention backward, original -> Triton (ms) | Backward speedup | Combined forward/backward speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 65^3 one-hop | 274625 | 1622400 | 18.081 -> 8.751 | 2.07x | 1.60x |
| 65^3 two-hop | 274625 | 4791930 | 49.771 -> 14.530 | 3.43x | 2.19x |
| 129^3 one-hop | 2146689 | 12780288 | 141.908 -> 65.848 | 2.16x | 1.62x |
| 129^3 two-hop | 2146689 | 38042874 | 391.253 -> 112.498 | 3.48x | 2.18x |

The final 129^3 two-hop GH200 run was Slurm job 511292 at revision
38abd607. Its deterministic numerical tests reported 27 passed and 1
skipped (DGL dependency unavailable). Maximum allocated geometric-attention
GPU memory: 16.205 GiB for each backend; maximum reserved: 20.611 GiB.
These results do not establish full-model training speedup or scalability for
all arbitrary-degree unstructured graphs. The JSON protocol intentionally
reports "completed_not_accepted" pending full-model evaluation.

## Next gate: real-HDF5 ten-layer M37 optimizer steps

The next probe reuses the existing M43-B1 real HDF5, GraphTaskCollator, and
Flow Matching code. It pairs the original torch_sparse backend against the
opt-in torch_sparse_triton backward with 10 layers, 128 hidden channels,
4 heads, AdamW, and non-reentrant activation checkpointing. Two fresh CUDA
subprocesses use matched HDF5 inputs, graph construction and RNG seeds;
the first optimizer step in each is a warmup, then three timed steps follow.

M43-B1 uses per-snapshot state standardization and box-normalized coordinates
for feasibility. This is NOT the original M47 physical normalization or
a scientific validation run. No checkpoints or trained model weights are
saved and no frozen M47.4 training scripts are changed.

Start from a clean checkout:

~~~bash
cd /scratch/coop/theret/GraphAttention
git pull --ff-only origin main
sbatch scripts/slurm/sparse_backward_triton_full_training.slurm
~~~

Default real case: Re54/Ma0.3 at 65^3, in
/archivenfs/phlusim/HIT_DNS_COMP. Output root:
/scratch/coop/theret/GraphAttention_runs/triton_full_training_v1/job_<jobid>/

Required analysis artifacts:

- /scratch/coop/theret/GraphAttention_runs/csr_full_train_<jobid>.out
- /scratch/coop/theret/GraphAttention_runs/csr_full_train_<jobid>.err
- <output>/preflight.json and <output>/summary.json
- <output>/torch_sparse/report.json and <output>/torch_sparse/process.log
- <output>/torch_sparse_triton/report.json and <output>/torch_sparse_triton/process.log

The Slurm script prints full ARTIFACT_* paths before GPU execution.
Use its output for analysis before attempting a 129^3 real graph.


## First full-model GH200 result: Slurm 511302 (real 65^3)

Revision 6be13c2e3b717dc2a5ec9a281865940b01df1783. The default real
Re54/Ma0.3 65^3 snapshot (solut_hit_00000051.h5) contains 274625 nodes,
1622400 directed one-hop edges and 4791930 exact-two-hop directed edges.
Both runs executed four complete 10-layer, four-head, 128-hidden-channel
Flow Matching AdamW steps with non-reentrant activation checkpointing.
The first step was excluded from timing; medians cover three later steps.

Source: /scratch/coop/theret/GraphAttention_runs/triton_full_training_v1/job_511302/
(summary.json, preflight.json and paired per-backend report.json/process.log).
Full Slurm stdout/stderr: csr_full_train_511302.out and .err.

| Stage | torch_sparse median (ms) | torch_sparse_triton median (ms) | Speedup |
| --- | ---: | ---: | ---: |
| Forward | 149.316 | 148.847 | 1.003x |
| Backward + loss | 2558.048 | 2340.270 | 1.093x |
| Optimizer | 1.667 | 1.594 | 1.046x |
| Complete step | 2709.511 | 2490.930 | 1.088x |

The full step fell by 218.58 ms (8.07%). Both full models completed with
6.878 GiB peak allocated and 7.412 GiB peak reserved memory. All four
reported probe-only loss values were identical. The comparison JSON
confirms matching real graph/input SHA-256 fingerprints, identical initial
weight hashes and equal model parameter counts. The Slurm preflight passed
three CPU tests and one deterministic checkpointed Triton integration test.

For the five local plus five exact-two-hop attention blocks, the isolated
65^3 backward reductions predict 5*(18.081-8.751)+5*(49.771-14.530) =
222.85 ms savings. The full-model observed backward savings are 217.78 ms,
within 2.3% of this prediction. Thus the reduced end-to-end speedup appears
consistent with unchanged model operations and checkpoint recomputation
rather than an unexpected loss of kernel acceleration.

This is a matched **computational feasibility probe only**, using
M43-B1's per-snapshot normalization. Identical loss *values* are not proof
of every parameter gradient or updated weight being bitwise equivalent.
No scientific validation, general unstructured graph benchmark, or
M47.4 training was attempted. The result remains "completed_not_accepted".

### Next: paired real 129^3 checkpointed probe

Do not alter the frozen M47.4 training. Use the same full-training Slurm
script with the exact existing Re94/Ma0.3:129 mesh/case:

~~~bash
CSR_FULL_CASE=Re94/Ma0.3:129 \
  sbatch --export=ALL,CSR_FULL_CASE=Re94/Ma0.3:129 \
  scripts/slurm/sparse_backward_triton_full_training.slurm
~~~

The new job writes the same eight artifacts listed above, in a unique
job_<jobid> directory. Verify peak GPU allocation, equal initial weights,
equal actual geometry hashes, and median end-to-end speedup before drawing
scalability conclusions. Full model profiling remains a separate optional
follow-up.

## Full-model 129^3 result: Slurm 511305 (real Re94/Ma0.3)

Revision e0806c52b14a9b5147b4d57d8c960755f20866c1.
The original 10-layer M37 checkpointed model was tested with the real
Re94/Ma0.3 129^3 snapshot: 2146689 nodes, 12780288 one-hop directed
edges, and 38042874 exact-two-hop directed edges. The 3 CPU and 1
deterministic CUDA preflight tests passed. Both backends completed four
Flow Matching AdamW updates, with the first discarded as a warmup.

| Stage | original median | Triton median | Speedup |
| --- | ---: | ---: | ---: |
| Forward | 1.192308 s | 1.193019 s | 0.999x |
| Backward incl. loss | 20.037588 s | 18.257388 s | 1.098x |
| AdamW | 0.001945 s | 0.001985 s | 0.980x |
| Full optimizer step | 21.232421 s | 19.452927 s | 1.091x |

Combined step time fell by 1.779494 s (8.38%). Both backends reported
53.165580 GiB peak allocated GPU and 58.296875 GiB reserved. Both
reports used the same graph/coordinate/input hashes, initial model
weight hash, and 3051845 parameters, with identical four reported loss
values. The comparison summary has status completed_not_accepted,
consistent with the M43 probe-only normalization.

The isolated 129^3 kernel timings predict five one-hop plus five
exact-two-hop backward savings of 1.7741 s. Observed full-model backward
savings are 1.7802 s, a 0.35% difference. This supports transfer of the
kernel improvement into the checkpointed model. Larger backward costs
are outside the optimized kernels; their attribution requires profiling.

Original artifacts:
- /scratch/coop/theret/GraphAttention_runs/csr_full_train_511305.out
- /scratch/coop/theret/GraphAttention_runs/csr_full_train_511305.err
- /scratch/coop/theret/GraphAttention_runs/triton_full_training_v1/job_511305/
  (summary.json, preflight.json, paired report.json and process.log)

The 65^3 and 129^3 results are single-job computational probes, not
multi-run uncertainty estimates or scientific training validation.
Do not alter the frozen M47.4 recipe based on these measurements alone.

## Next diagnostic: paired CUDA operator profile on real 65^3

Start at 65^3 to keep profiling overhead and artifact size controlled.
The new profile runner reuses M43-B1 and records one warmup plus one
separately CUDA-profiled full checkpointed optimizer step for each
backend. CUDA operator self times and call counts are summarized into
JSON; no large Chrome trace or scientific checkpoint is saved by default.
The profiler changes execution behavior, so its step durations must NOT
be compared with the earlier unprofiled timing benchmark.

~~~bash
cd /scratch/coop/theret/GraphAttention
git pull --ff-only origin main
sbatch scripts/slurm/sparse_backward_triton_full_profile.slurm
~~~

Output paths for Slurm job <jobid>:
- /scratch/coop/theret/GraphAttention_runs/csr_full_profile_<jobid>.out
- /scratch/coop/theret/GraphAttention_runs/csr_full_profile_<jobid>.err
- /scratch/coop/theret/GraphAttention_runs/triton_full_profile_v1/job_<jobid>/preflight.json
- /scratch/coop/theret/GraphAttention_runs/triton_full_profile_v1/job_<jobid>/summary.json
- /scratch/coop/theret/GraphAttention_runs/triton_full_profile_v1/job_<jobid>/torch_sparse/report.json
- /scratch/coop/theret/GraphAttention_runs/triton_full_profile_v1/job_<jobid>/torch_sparse/operator_profile.json
- /scratch/coop/theret/GraphAttention_runs/triton_full_profile_v1/job_<jobid>/torch_sparse/process.log
- /scratch/coop/theret/GraphAttention_runs/triton_full_profile_v1/job_<jobid>/torch_sparse_triton/report.json
- /scratch/coop/theret/GraphAttention_runs/triton_full_profile_v1/job_<jobid>/torch_sparse_triton/operator_profile.json
- /scratch/coop/theret/GraphAttention_runs/triton_full_profile_v1/job_<jobid>/torch_sparse_triton/process.log

The Slurm job prints every path before GPU execution. Return stdout,
stderr, summary, both operator_profile.json and both process.log for
the first analysis; also return both report.json if a mismatch or failure
occurs. Its only purpose is operator-level bottleneck identification.

## Operator profile result: 511309 (real 65^3)

Code revision b79f8cc404b8f00768e51299fe850df0b5ed2b27.
Source:
  /scratch/coop/theret/GraphAttention_runs/triton_full_profile_v1/job_511309/
Both one-step CUDA operator profiles completed with the same real HDF5
graph/input/initial weights and the same reported probe losses.

Selected top CUDA event classes in the separately profiled second step:

| Event | torch_sparse | torch_sparse_triton |
| --- | ---: | ---: |
| indexing_backward_kernel<float, 4>, 11 launches | 2045.607 ms | 2042.301 ms |
| aten::_index_put_impl_, 41 calls | 2069.127 ms | 2065.815 ms |
| aten::index_add_ | 156.170 ms | 60.493 ms |
| aten::index | 128.569 ms | 57.009 ms |
| Triton _csr_rows_reduce | N/A | 50.069 ms |
| sampled_addmm | 31.278 ms | 31.267 ms |

The aten::_index_put_impl_ and raw indexing_backward_kernel entries
largely represent the same underlying work and MUST NOT be added
together. CPU cudaStreamSynchronize time is also GPU waiting, not
independent CPU processing. Key-averaged profiler self-device totals
are not directly comparable with unprofiled wall time; the profiler
includes operator and kernel-level entries with overlapping attribution.

Model source suggests a specific reason for the 11 large indexing-backward
launches: _broadcast_graph_features in dit_common.py indexes
features[batch_index] in every one of ten DiT blocks and in the final
layer. For a single packed graph, batch_index is all zeros, producing
strong scatter-gradient contention. The exact match in launch counts
is evidence, but the current profile lacks per-call tensor shapes
and stacks; this source identification remains a hypothesis pending ablation.

## Next: packed versus native single-graph model path (opt-in)

The existing model already uses features.expand(num_nodes, -1)
when batch_index=None. Its backward reduces graph-conditioning gradients
rather than gathering/scattering into a single graph row.

M43's real-HDF5 probe now accepts --single-graph-forward as an
EXPERIMENTAL flag. It validates that the original batch is exactly one
graph with all-zero batch_index, then calls M37 with batch_index=None.
The default path is unchanged, as are model weights, attention kernels,
Flow Matching, AdamW and activation checkpointing.

The controlled four-arm ablation runs:
- torch_sparse with original packed batch_index
- torch_sparse_triton with original packed batch_index
- torch_sparse with native single-graph batch_index=None
- torch_sparse_triton with native single-graph batch_index=None

All four use the same 65^3 snapshot, seed and ten-layer architecture
in fresh CUDA processes. Four optimizer steps per arm, with the first
excluded from unprofiled timing. CPU and CUDA preflight tests check
model forward and input/conditioning/parameter gradient parity between
the two paths. The summary also compares probe-only loss sequences
and matching graph/input/initial-weight SHA-256 hashes. This is NOT
scientific training; no checkpoints are written. Do not enable this
path for multi-graph inputs.

Run from clean main:
~~~bash
cd /scratch/coop/theret/GraphAttention
git pull --ff-only origin main
sbatch scripts/slurm/sparse_backward_triton_single_graph_ablation.slurm
~~~

For Slurm job <jobid>, output root is:
  /scratch/coop/theret/GraphAttention_runs/triton_broadcast_ablation_v1/job_<jobid>/

Required artifacts for analysis:
- /scratch/coop/theret/GraphAttention_runs/csr_broadcast_<jobid>.out
- /scratch/coop/theret/GraphAttention_runs/csr_broadcast_<jobid>.err
- <output>/comparison.json
- <output>/packed/summary.json
- <output>/single_graph/summary.json
- <output>/<packed|single_graph>/<torch_sparse|torch_sparse_triton>/report.json
- <output>/<packed|single_graph>/<torch_sparse|torch_sparse_triton>/process.log

The Slurm script prints ARTIFACT_* for all exact paths before tests.
If parity fails, investigate numerical differences before proposing
any default-code optimization or extrapolating savings to larger meshes.

## Single-graph ablation result: Slurm 511501 (real 65^3)

Benchmark revision 7596ff3b7182a18408a96780b07f3292355fa03e.
Ground truth:
  /scratch/coop/theret/GraphAttention_runs/triton_broadcast_ablation_v1/job_511501/comparison.json
  (plus both mode summaries, four reports, four per-process logs)
Drive folder:
  https://drive.google.com/drive/folders/1WJPlGW1cfAkGqnSli7LmM7kHYmKRwxdz

Four fresh CUDA processes (same real Re54/Ma0.3:65 snapshot, 10-layer M37,
128 hidden features, 4 heads, non-reentrant checkpointing, FP32 and AdamW).
Per-run first step excluded from timings; the remaining three-step median:

| Mode | CSR backend | Forward (ms) | Backward + loss (ms) | Complete step (ms) | Peak allocated (GiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| Packed batch_index, B=1 | torch_sparse | 148.718 | 2565.187 | 2716.508 | 6.878 |
| Packed batch_index, B=1 | torch_sparse_triton | 149.073 | 2338.514 | 2489.562 | 6.878 |
| Native batch_index=None, B=1 | torch_sparse | 137.076 | 508.959 | 647.903 | 6.092 |
| Native batch_index=None, B=1 | torch_sparse_triton | 137.428 | 288.340 | 427.575 | 6.092 |

Speedups: packed vs native single-graph 4.193x with original CSR and
5.823x with Triton; once single-graph, Triton contributes a further
1.515x speedup. Combined original packed CSR vs native single-graph
Triton: 6.353x, or 84.26% lower full-step latency. Maximum allocated
GPU memory drops by 0.786 GiB (11.43%) in either backend. The peak
reserved allocator memory is unchanged at 7.412 GiB.

All four benchmark subprocesses report status=ok and exit_code=0.
Matching one-/two-hop graph SHA-256, coordinate/state SHA-256, initial
model-weight SHA-256 and 3051845 parameters. Four reported loss values
are exactly equal between all arms:
2.0007710457, 1.9952092171, 2.0095293522, 1.9809440374.
The summary includes a per-arm 1e-4 loss-difference acceptance gate.
The Slurm script preflights CPU/CUDA forward and input/conditioning/
parameter-gradient parity on small graphs before benchmarking.
Neither full model final parameter deltas nor real-graph per-parameter
gradients have yet been explicitly compared; equal loss values do not
establish full numerical equivalence of all parameter updates.

The earlier 65^3 profiler showed 11 launches of
indexing_backward_kernel<float,4> totaling ~2042 ms with Triton CSR.
Switching from packed batch_index to the native single-graph path lowers
the Triton backward median by 2050.174 ms. The close correspondence
strongly supports the hypothesis that _broadcast_graph_features'
features[batch_index] gradient contention dominates the former step.
However batch_index=None also selects a different (mathematically
equivalent) single-graph coordinate normalization path, so the ablation
does not isolate only one source expression. No default model or frozen
M47.4 training configuration was changed.

The performance result is computational feasibility with M43's
per-snapshot normalization, not a scientific training result, and is
accepted only as "completed_not_accepted". The next step is to repeat
the four-arm ablation on real Re94/Ma0.3:129 before any production
fast-path change. Use exactly the existing opt-in runner:

~~~bash
cd /scratch/coop/theret/GraphAttention
git status --short
git pull --ff-only origin main
sbatch --export=ALL,CSR_BROADCAST_CASE=Re94/Ma0.3:129 \
  scripts/slurm/sparse_backward_triton_single_graph_ablation.slurm
~~~

For Slurm job <jobid>, mandatory output paths:
- /scratch/coop/theret/GraphAttention_runs/csr_broadcast_<jobid>.out
- /scratch/coop/theret/GraphAttention_runs/csr_broadcast_<jobid>.err
- /scratch/coop/theret/GraphAttention_runs/triton_broadcast_ablation_v1/job_<jobid>/comparison.json
- /scratch/coop/theret/GraphAttention_runs/triton_broadcast_ablation_v1/job_<jobid>/packed/summary.json
- /scratch/coop/theret/GraphAttention_runs/triton_broadcast_ablation_v1/job_<jobid>/single_graph/summary.json
- /scratch/coop/theret/GraphAttention_runs/triton_broadcast_ablation_v1/job_<jobid>/<packed|single_graph>/<torch_sparse|torch_sparse_triton>/report.json
- /scratch/coop/theret/GraphAttention_runs/triton_broadcast_ablation_v1/job_<jobid>/<packed|single_graph>/<torch_sparse|torch_sparse_triton>/process.log

All paths print with ARTIFACT_* at the start of the Slurm job.
If the 129^3 numerical and performance gates pass, consider making
single-graph broadcast selection an explicit validated capability for
scientific training, with a full-real-graph parameter/gradient
equivalence check before enabling any production default. Keep the
multi-graph packed index path intact.

## M47 experimental heterogeneous multi-graph segmented-conditioning gate

This is a computational test, NOT an architecture/default-trainer change.

The model already supports disjoint packed graphs with different node
counts and connectivity, through GraphTaskCollator and batch_index.
However, DiTBlock/DiTFinalLayer previously expanded global adaLN
modulation via graph_features[batch_index], leading to highly contended
advanced-indexing backward for large graphs.

An explicitly opt-in graph_sizes=(N1, N2, ...) argument in M37 now
selects segmented expand+cat of global modulation. This retains packed
graph edges and independent graph-level Flow Matching time. Each
expand is a view whose backward is a per-graph reduction; torch.cat
materializes the forward feature tensor. Default calls still use the
unchanged packed advanced-indexing implementation. The native single
graph path is also unchanged.

Three execution modes are compared with the SAME real CFD inputs,
same generated Flow Matching time/noise per graph and same initial
weights in fresh CUDA subprocesses:

1. packed_index: original batch_index advanced-indexing broadcast.
2. packed_segmented: native expand+cat per contiguous graph segment.
3. sequential_accum: one single-graph fast-path forward/backward per
   physical graph; accumulate the losses/gradients, then perform a
   *single* AdamW optimizer update.

All modes use a 10-layer M37 DiNAT-DiT (128 hidden channels, 4 heads),
checkpointing, FP32, original Triton CSR backward and AdamW(lr=1e-4).
The scientific FlowMatchingTask.sample_reduced_mse already weights
the two physical graphs equally by mean channel/node loss, regardless
of unequal node counts; the sequential reference explicitly reproduces
that weighting, dividing each individual loss by B=2. The same packed
Flow Matching training problem is generated per step in all modes.
The worker uses only probe-only per-snapshot normalization and the
existing M43 one-sample initialization. No scientific M47.4 model
checkpoint, normalization, or long training run is changed.

Step 1 is a compilation warmup AND numerical audit: gradients for
every parameter and all post-AdamW parameters are saved to per-mode
audit.pt files on scratch. The driver compares these complete tensors
against sequential accumulation with per-element gradient tolerances
atol=5e-4, rtol=5e-3; updated parameters atol=3e-6,
rtol=2e-4, and reports worst deviations and mismatch counts.
It also checks all input/topology/initial-weight SHA-256 fingerprints
and the full per-step equal-graph loss sequence (atol=1e-4).
Steps 2-4 provide synchronized, unprofiled forward/backward/optimizer
and full-step median timings and peak GPU allocation. All outcomes
remain "completed_not_accepted" even when numerically acceptable:
this is computational feasibility, not scientific training.

Run in two stages; do not run mixed 129^3 until 65^3 gate passes:

### Stage 0: two different real CFD snapshots with same 65^3 mesh

~~~bash
cd /scratch/coop/theret/GraphAttention
git status --short
git pull --ff-only origin main
sbatch --array=0 scripts/slurm/m47_segmented_multigraph_ablation.slurm
~~~

Task 0 uses two distinct HDF5 snapshots from Re54/Ma0.3:65.
This is different physical field data, but the same mesh and regime.
It isolates heterogeneous per-sample inputs without changing resolution.

### Stage 1: different regimes AND different resolutions

~~~bash
sbatch --array=1 scripts/slurm/m47_segmented_multigraph_ablation.slurm
~~~

Task 1 uses the existing real Re54/Ma0.3:65 and Re94/Ma0.3:129
datasets, with distinct mesh connectivity and node counts. Combined:
2,421,314 nodes; a single GPU is used for all 3 modes.
Memory feasibility is not assumed; inspect the Slurm outcome.
Only Flow Matching time is in the M43 probe's model conditioning;
Re/Ma conditioning belongs to frozen M47 scientific experiments and
is *not* added to this numerical computational comparison.

For array job <arrayid> task <taskid>, output root:
  /scratch/coop/theret/GraphAttention_runs/m47_segmented_multigraph_v1/job_<arrayid>_<taskid>/

The Slurm script prints every exact ARTIFACT_* path:
- /scratch/coop/theret/GraphAttention_runs/m47_seg_batch_<arrayid>_<taskid>.out
- /scratch/coop/theret/GraphAttention_runs/m47_seg_batch_<arrayid>_<taskid>.err
- <output>/manifest.json
- <output>/summary.json
- <output>/packed_index/{report.json,process.log,audit.pt}
- <output>/packed_segmented/{report.json,process.log,audit.pt}
- <output>/sequential_accum/{report.json,process.log,audit.pt}

Return stdout/stderr, manifest/summary and the three reports/logs.
The raw audit.pt files can remain on scratch unless parity fails;
the driver already compares *all* first-step gradients/updates and
records mismatch counts in summary.json. If the gate fails, include
audit.pt to reproduce exact parameter-level discrepancies.

Do not extrapolate a two-graph test into arbitrary variable-degree
unstructured-mesh scaling or change scientific training defaults
without checking task-1 results and physical normalization choices.

### 511548 sequential-CUDA diagnostic (preflight, not an experimental result)

Slurm 511548 at e567477 stopped during the small-graph preflight:
9 passed, 2 failed, 1 skipped. Both original CSR and Triton CSR
packed/segmented forward parity was exact. The separately executed
CUDA sequential reference differed in forward by at most 5.29345e-5
and in loss by 3.45707e-6. Several first-step AdamW update differences
were accurately explained by the first-step gradient formula (residual
below 3e-8), and the remaining strict preflight failure was a
sequential CUDA parameter-gradient discrepancy for
final_layer.projection.weight. The job DID NOT reach real-HDF5 timings.

The experimental preflight was revised to keep strict packed-vs-
segmented CPU/CUDA gradient/update gates but to DIAGNOSE per-parameter
sequential CUDA gradient discrepancies rather than suppress a real
multi-graph experiment based on a tiny CUDA execution-shape test.
It logs max-absolute/rms deltas and counts mismatches using the exact
real-HDF5 audit tolerances. The complete REAL data-path test still
compares every gradient and every post-AdamW parameter to the
sequential reference, and marks the experiment failed_or_review_required
if numerical gates are violated. An additional direct packed-index-vs-
segmented full-parameter audit is always recorded separately on success.
No scientific trainer/default optimizer changes were made.

### 511557 two-real-snapshot segmented multigraph result (Stage 0 PASS)

Real Re54/Ma0.3:65 snapshots solut_hit_00000051.h5 and
solut_hit_00000052.h5, same 65^3 mesh/topology, different solution
fields. Computation revision eeec0d2ee58d4fcb570b9bcfcbc474994b858e00.
Drive artifacts:
https://drive.google.com/drive/folders/17IXucZQ1KAlZAXJ0tNEKPJkeAFBx-xl8
Scratch root:
  /scratch/coop/theret/GraphAttention_runs/m47_segmented_multigraph_v1/job_511557_0

All three fresh-GPU subprocesses completed with no errors. Both
samples have 274625 nodes; the disjoint packed graph has 549250 nodes,
3244800 local directed edges and 9583860 exact-two-hop directed edges.
Two real snapshots share physical Re/Ma and mesh: Stage 0 does NOT
establish mixed-regime or mixed-resolution behavior. Same input/topology
hashes and 3051845 starting parameters across all modes.

Triton CSR + 10-layer M37, checkpointed, FP32, AdamW(lr=1e-4):
synchronized unprofiled median over optimizer steps 2-4; step 1
excluded for warmup and full-tensor gradient/update audit.

| Execution | Forward (ms) | Backward + loss (ms) | Full step (ms) | Peak allocated (GiB) | Peak reserved (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| packed_index | 301.149 | 2640.989 | 2943.275 | 13.878 | 15.699 |
| packed_segmented | 291.555 | 583.127 | 876.119 | 13.878 | 15.699 |
| sequential_accum | 281.606 | 577.104 | 860.268 | 6.442 | 7.656 |

Packed-index -> segmented: 3.359x faster full step; the backward
improvement is ~2057.861 ms. Segmented is ~1.84% SLOWER than sequential
accumulation on this particular single-GH200 B=2 workload, and uses
2.15x the allocated peak GPU memory. While graph/step timings were
stable, only three post-warmup measurements are available; do not
generalize the 1.84% distinction beyond the tested configuration.

All 3051845 named parameter gradients and updated weights passed
the complete first-step audit:
- packed_segmented vs packed_index: maximum gradient difference 0;
  maximum post-AdamW parameter difference 0; equal all four losses.
- packed_segmented vs sequential_accum: maximum gradient difference
  2.4139881e-6 (final_layer.projection.weight), maximum post-AdamW
  parameter difference 2.1827873e-11; maximum 4-step loss difference
  2.3841858e-7. Zero elements exceeded gradient tolerances
  atol=5e-4,rtol=5e-3 or post-update tolerances
  atol=3e-6,rtol=2e-4.

Interpretation: segmented expand+cat eliminates the large packed-index
backward bottleneck without changing the tested model numerics, but
it DOES NOT demonstrate an efficiency advantage over sequential
single-graph gradient accumulation for two large graphs on one GPU.
The packed approach materializes graph-wide modulation and carries
both graphs' activations simultaneously. The sequential reference
remains the recommended memory-efficient baseline for this regime.

Stage 1 is justified as a **separate mixed-resolution feasibility
experiment**, NOT as evidence that packed batching will be faster.
It may exceed resource limits; never infer success from Stage 0.
Submit a separate job after updating a clean main branch:

~~~bash
cd /scratch/coop/theret/GraphAttention
git status --short
git pull --ff-only origin main
sbatch --array=1 scripts/slurm/m47_segmented_multigraph_ablation.slurm
~~~

For array job <jobid>, task 1:
- /scratch/coop/theret/GraphAttention_runs/m47_seg_batch_<jobid>_1.out
- /scratch/coop/theret/GraphAttention_runs/m47_seg_batch_<jobid>_1.err
- /scratch/coop/theret/GraphAttention_runs/m47_segmented_multigraph_v1/job_<jobid>_1/summary.json
- /scratch/coop/theret/GraphAttention_runs/m47_segmented_multigraph_v1/job_<jobid>_1/manifest.json
- <output>/<packed_index|packed_segmented|sequential_accum>/report.json
- <output>/<packed_index|packed_segmented|sequential_accum>/process.log
- <output>/<packed_index|packed_segmented|sequential_accum>/audit.pt

No scientific M47 trainer, normalization constants, or conditioning
parameters were changed. Probe-only normalization is used. Re/Ma values
are dataset labels for Stage 1; the M43 probe conditions on Flow
Matching time rather than explicit physical Re/Ma parameters.


### 511565 heterogeneous real-graph segmented-broadcast Stage 1 result

**Numerical and memory feasibility PASSED; no packed batching advantage
over sequential accumulation on one GH200. Scientific acceptance remains
"completed_not_accepted" (short M43 probe-only normalization and FM time
conditioning; no explicit Ma/Re physical conditioning).**

Job 511565, computation revision
8746dd0ad701226d822d6486ea8c30920904ec2f, real samples:
- Re54/Ma0.3 65^3 = 274,625 nodes,
  /archivenfs/phlusim/HIT_DNS_COMP/Re54/Ma0.3/SOLUT/solut_hit_00000051.h5
- Re94/Ma0.3 129^3 = 2,146,689 nodes,
  /archivenfs/phlusim/HIT_DNS_COMP/Re94/Ma0.3/SOLUT/solut_hit_00000051.h5
- combined packed graph: 2,421,314 nodes, 10-layer M37/DiNAT-DiT,
  hidden=128, 4 heads, checkpointing, Triton CSR backward, FP32,
  AdamW(lr=1e-4), equal *per-physical-graph* loss.

Artifacts in Drive:
https://drive.google.com/drive/folders/1NYTsSmYi_zbc3HiMxXqbKDgrNmfxopPc

Absolute scratch output root:
  /scratch/coop/theret/GraphAttention_runs/m47_segmented_multigraph_v1/job_511565_1
Main files: summary.json, manifest.json, plus for each of
packed_index, packed_segmented, sequential_accum:
report.json, process.log, audit.pt.

Three processes exit 0, status "ok". Each had 4 optimizer steps;
step 1 excluded. Synchronized unprofiled medians of steps 2-4:

| Mode | Forward (s) | Backward + loss (s) | Full step (s) | Peak alloc (GiB) | Peak reserved (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| packed_index | 1.343699 | 18.596750 | 19.942264 | 60.934964 | 66.765625 |
| packed_segmented | 1.318111 | 2.597638 | 3.917339 | 60.934964 | 66.765625 |
| sequential_accum | 1.268721 | 2.548612 | 3.819152 | 48.140202 | 61.804688 |

The segmented conditioning is 5.090768x faster than packed index
(full step), principally reducing backward by 15.9991 seconds/step.
Sequential accumulation is still 2.57% faster than packed segmented,
with 12.795 GiB (26.6%) less peak *allocated* memory.

Full first-step audit across all 3,051,845 model parameter elements:
- packed_index vs packed_segmented: exactly equal gradients, post-AdamW
  parameters, and recorded losses (max differences all 0).
- either packed mode vs sequential_accum: max |gradient delta| =
  6.3568354e-5 (final_layer.projection.weight), max |updated-parameter
  delta| = 2.9103830e-11, max 4-step loss delta = 2.3841858e-7.
  No elements violated audited gradient (atol=5e-4, rtol=5e-3) or
  post-update (atol=3e-6, rtol=2e-4) tolerances.
These are numerical tolerances, not a claim of exact sequential parity.

Stage 0 (two distinct same-regime 65^3 real snapshots) yielded packed
segmented 0.876119s/step vs sequential 0.860268s/step. Mixed-resolution
Stage 1 similarly prefers sequential (3.917339s vs 3.819152s). We
therefore should keep the benchmark-only segmented capability available
but should not enable it as the scientific training default for
single-GPU heterogeneous CFD batches.

Next work, if scientifically needed:
1. Confirm the scientific trainer can accumulate B individual
   equal-sample losses (each scaled 1/B) before ONE optimizer update.
   Respect task-native normalization and physical conditioning in
   scientific runs, not the per-snapshot probe normalization.
2. Benchmark DDP graph-per-GPU with correct per-graph weighting and
   communication overhead, if more than one GPU is available.
3. Consider fused graph-conditioning modulation (avoids materializing
   N x 6H broadcast) only if packed single-GPU execution becomes a
   requirement at larger graph counts; no evidence of benefit here.

Do not claim the 65^3 / 129^3 computational comparison validates
generalization across regimes or physical-condition fidelity.
