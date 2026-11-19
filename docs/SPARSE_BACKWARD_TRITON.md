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
