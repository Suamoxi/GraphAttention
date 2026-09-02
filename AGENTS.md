# AGENTS.md

## Scope

These rules apply to every implementation, bug fix, refactor, benchmark, scientific change, configuration change, and documentation change in this repository.

The repository is a scientific machine-learning framework for CFD fields on meshes of varying size, resolution, and topology. Scientific correctness, numerical correctness, scalability, traceability, and reproducibility take precedence over convenience.

## Priority order

When priorities conflict, use the following order unless a task explicitly justifies a different trade-off:

\[
\boxed{
\text{scientific correctness}
\rightarrow
\text{numerical correctness/stability}
\rightarrow
\text{target-workload efficiency}
\rightarrow
\text{clarity}
\rightarrow
\text{maintainability}
}
\]

Dependency minimization is desirable but is not allowed to force an architecturally inferior or materially slower implementation on the intended workload.

## Minimal-change decision ladder

Before writing new code, evaluate these questions in order and stop at the first option that fully satisfies the requirement:

1. **Does this functionality need to exist?**
   - If the requested outcome is already satisfied, make no code change.
   - Do not implement speculative features, abstractions, or future-proofing without a concrete use case.

2. **Does an equivalent implementation already exist in the repository?**
   - Search existing modules, tests, configurations, utilities, and patterns.
   - Reuse or extend existing code instead of introducing a parallel implementation.

3. **Can the Python standard library solve it clearly?**
   - Prefer the standard library when performance and scientific requirements are adequately met.

4. **Can the primary framework solve it adequately for the target workload?**
   - Prefer native PyTorch/Hydra/Lightning functionality when it is correct, sufficiently efficient, and clear.
   - “Native” is not a justification if a specialized implementation provides a material and demonstrated advantage.

5. **Can an installed or mature specialized dependency solve it better?**
   - Evaluate correctness, GPU memory, throughput, scaling, mixed precision, autograd, DDP support, portability, and maintenance burden.
   - Do not reimplement mature optimized functionality solely to avoid a dependency.

6. **Can the requirement be expressed directly and locally?**
   - Prefer a direct expression over a new abstraction when the abstraction has only one real use.

7. **Only then implement the smallest coherent new solution.**

## Scientific ownership boundaries

The repository must preserve explicit ownership between the following layers.

### Data — what exists

The data layer owns:

- raw CFD file discovery and reading;
- physical fields supplied by the simulation;
- native mesh coordinates and connectivity;
- boundary metadata supplied by the data;
- case metadata and reference quantities;
- field catalogue information;
- split manifests and dataset provenance.

The data layer must not decide:

- k-hop depth;
- sparse-attention topology;
- random-edge augmentation for a model;
- learning targets;
- diffusion noising;
- optimizer behavior.

### Geometry — how space is related

The geometry layer owns deterministic spatial and topological transforms derived from supplied meshes, including where required:

- canonical sparse connectivity;
- relative positions and distances;
- mesh metrics;
- exact-hop edge sets;
- deterministic augmented connectivity;
- graph partitions;
- source-to-query or coarse-to-fine correspondences;
- geometry-derived resolution descriptors.

Geometry code must not know the learning objective, loss, optimizer, or trainable model parameters unless a future method explicitly introduces learned geometry and documents that exception.

### Task — what is learned

The task layer owns:

- ordered input-field selection;
- target-field selection;
- conditioning-variable selection;
- task-specific target construction;
- temporal increments;
- diffusion noising and parameterization;
- flow-matching targets;
- super-resolution residual construction;
- wall-query construction;
- interpretation of model outputs for loss/evaluation.

### Model — how the supplied representation is transformed

The model layer owns learnable transformations only. It may consume physical state, sparse topology, geometry/resolution information, and conditioning through explicit interfaces.

The model must not:

- discover raw files;
- infer channel semantics from integer indices;
- fit dataset normalization statistics;
- choose training targets;
- implement optimizer stepping;
- depend on mesh identity or node numbering.

### Trainer — how optimization is executed

The training layer owns:

- backward propagation;
- gradient accumulation;
- DDP synchronization;
- AMP/BF16 mechanics;
- optimizer/scheduler stepping;
- checkpoint orchestration;
- logging of optimization/runtime state.

The trainer should remain agnostic to task semantics wherever practical.

## Variable-mesh and batching requirements

Variable mesh size is a core requirement, not an optional extension.

The canonical model representation is a packed collection of disconnected sparse graphs. A microbatch may contain samples with different:

- node counts;
- edge counts;
- connectivity;
- resolution;
- geometry;
- topology.

Padding to a common node count or neighborhood size is not the default representation and may only be introduced when required by a demonstrated algorithmic or backend constraint.

GPU microbatches are constrained by explicit computational budgets such as total nodes and total edges. Computational microbatch size and statistical optimizer batch size are distinct concepts.

Packing must not silently alter the intended training distribution or sample weighting.

## Statistical-loss requirements

A fine mesh must not receive more statistical importance solely because it contains more nodes.

Per-sample loss must be reduced within each physical sample before combining samples into the optimizer-level objective. Where scientifically justified, use physical quadrature/control-volume/surface weights rather than an unweighted node mean.

Gradient accumulation must preserve the intended sample-level weighting when different microbatches contain different numbers of graphs.

In DDP, do not naïvely average rank-local mean losses when ranks contribute different numbers or weights of physical samples. The effective objective must correspond to the explicitly defined global weighting.

## Field semantics

Never infer scientific meaning from tensor position, shape, HDF5 group ordering, filename conventions, or node IDs.

Raw CFD files may contain many arrays with identical shapes but different roles. Every supported field must be represented through explicit semantic metadata.

The dataset defines what fields are available. The task defines which ordered fields are used as input, target, or conditioning.

Readers should load only requested fields whenever the storage format permits.

Stored fields and framework-derived fields must retain distinct provenance.

Do not implicitly concatenate physical state, geometry, boundary information, diagnostics, and global regime variables into one anonymous feature tensor.

## Physical preprocessing

Physical nondimensionalization and statistical ML scaling are separate operations.

The required order is:

\[
\text{dimensional CFD}
\rightarrow
\text{physical nondimensionalization}
\rightarrow
\text{training-set statistical scaling}
\]

Reference-scale definitions are part of the scientific specification. Reference values may differ numerically between cases, but their semantic definitions and derivation rules must remain explicit and reproducible.

Do not use reference quantities or conditioning variables that require unavailable target or future information at inference.

Training-set scalers must be associated with named dimensionless fields/components and frozen for validation, testing, and inference.

## Node-renumbering equivariance

Node indices are implementation identifiers, not physical quantities.

Renumbering mesh nodes while consistently renumbering all node-associated data and connectivity must not change the physical prediction, up to documented numerical tolerance.

Do not introduce scientific dependence on:

- raw node IDs;
- storage order;
- the first/last nodes in an array;
- edge-list order;
- arbitrary exporter numbering.

Any exception requires a documented physical meaning and a corresponding scientific test.

## Scientific change classification

Every change must be classified as one of the following.

### Software-only

Examples:

- internal renaming;
- I/O refactoring with identical data semantics;
- logging cleanup;
- bug fix with no numerical/scientific effect.

Requirements:

- ordinary unit/integration tests as appropriate;
- no scientific-spec update unless behavior actually changes.

### Numerically meaningful

Examples:

- normalization changes;
- epsilon values;
- mixed-precision behavior;
- numerical approximations;
- sampling details;
- reduction/weighting changes.

Requirements:

- update `docs/NUMERICAL_CONVENTIONS.md` if the convention changes;
- add or update numerical reference tests;
- document expected tolerances and precision behavior.

### Scientifically meaningful

Examples:

- loss definition;
- attention equation;
- positional encoding;
- topology/dilation strategy;
- diffusion parameterization;
- wall target;
- super-resolution formulation;
- conditioning semantics.

Requirements:

- update `docs/SCIENTIFIC_SPEC.md`;
- update `docs/TRACEABILITY.md`;
- add or update scientific reference tests;
- state whether the change is an established method, project adaptation, or project hypothesis.

Architecture changes should be introduced and ablated one at a time unless a combined change is scientifically unavoidable and explicitly justified.

## Scientific traceability

No mathematically meaningful implementation may exist only in code.

For every significant scientific mechanism, preserve:

- the mathematical definition;
- assumptions;
- source/reference when applicable;
- project-specific modification;
- implementation location;
- validation/test location;
- experiment or benchmark evidence when available.

Use `docs/TRACEABILITY.md` as the authoritative map.

## Source and claim discipline

Distinguish clearly between:

- **Fact** — established by cited source, repository state, or exact computation;
- **Inference** — reasoned expectation not yet measured;
- **Measured result** — empirical result from a specified benchmark/experiment;
- **Project hypothesis** — research idea to be tested.

Do not convert an inference into a performance or scientific claim merely because the code compiles or asymptotic complexity appears favorable.

## Performance policy

Performance is a first-class requirement.

When a performance-sensitive implementation is under consideration, evaluate where relevant:

- forward latency;
- forward+backward latency;
- training throughput;
- inference throughput;
- peak allocated memory;
- peak reserved memory;
- scaling with nodes;
- scaling with edges;
- scaling with batch composition;
- scaling across GPUs;
- mixed-precision behavior;
- kernel launch/memory materialization patterns;
- autograd/DDP compatibility.

Do not prefer native PyTorch solely because it avoids a dependency. Do not prefer a specialized dependency solely because it advertises sparse support.

Performance claims use the evidence states defined in `docs/BENCHMARK_PROTOCOL.md`:

- `ANALYTICAL`;
- `LOCAL_BENCHMARK`;
- `TARGET_VALIDATED`.

If target-cluster measurements are unavailable, implementation may proceed when scientifically/numerically correct, but target performance must remain explicitly unvalidated.

## Reproducibility

Every meaningful run must preserve enough information to reproduce the scientific preprocessing, model, data split, and optimization setup. Follow `docs/REPRODUCIBILITY.md`.

At minimum preserve:

- resolved configuration;
- git SHA;
- repository dirty/clean state;
- environment/runtime versions;
- dataset manifest;
- split definition;
- field selection/order;
- physical reference definitions;
- field transformations;
- statistical scalers;
- random seeds.

## Testing requirements

Keep distinct categories of tests:

- software unit tests;
- numerical reference tests;
- scientific-property tests;
- integration tests;
- benchmark/smoke tests where appropriate.

Mandatory scientific properties include node-renumbering equivariance for graph models.

Variable-mesh batching must include tests where different graph sizes/topologies coexist in the same packed microbatch.

Batching tests should verify that model outputs on disconnected packed graphs agree, within tolerance, with processing the same graphs separately when the model contains no intentional cross-sample coupling.

## Configuration policy

Expose scientific choices, not implementation plumbing.

Use Hydra composition and explicit named fields/parameters. Avoid custom registries or factory hierarchies unless Hydra/native mechanisms cannot satisfy a real requirement.

Do not add speculative configuration knobs. Every option must correspond to a supported and tested behavior.

## Documentation-before-claim rule

A new scientific or performance capability is not considered complete merely because code exists.

Before claiming completion, verify the corresponding documentation, tests, reproducibility state, and benchmark status required by this file.
