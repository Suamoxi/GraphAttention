# M0 Frozen Specification

This document is the compact record of the decisions frozen during M0. Detailed rules live in the other documentation files.

## M0.1 — Scientific scope

The repository is a scientific machine-learning framework for modelling CFD fields on meshes of varying size, resolution, and topology, with a focus on efficient sparse geometric neural architectures.

It supports multiple learning paradigms, including deterministic regression, wall-quantity prediction, super-resolution, operator learning, and generative modelling such as diffusion and flow matching.

The core abstraction is:

\[
\boxed{\text{CFD field} + \text{mesh geometry/topology} + \text{task}}
\]

Mesh connectivity and geometry are first-class inputs. Learned parameters must not depend on a particular mesh size, node numbering, or topology.

## M0.2 — Variable-mesh batching and statistical consistency

The canonical representation is a packed collection of disconnected sparse graphs.

Different meshes may coexist in the same microbatch.

Computational microbatches are constrained by explicit node/edge budgets:

\[
N_{\mathrm{total}}\le N_{\max},
\qquad
E_{\mathrm{total}}\le E_{\max}.
\]

Computational microbatch size and statistical optimizer batch size are distinct. Gradient accumulation must preserve the intended sample-level objective.

Per-sample losses are normalized independently, preferably using physically meaningful quadrature/control-volume/surface weights when available.

Sampler and packer are separate mechanisms. DDP weighting must correspond to the global statistical objective rather than naïve rank-local mean averaging.

## M0.3 — Graph vertices

For the initial repository:

\[
\boxed{\text{one graph vertex} = \text{one CFD mesh node}}
\]

Native CFD connectivity is preserved. Cells/faces may provide geometry and numerical metadata but are not initially separate learned graph entities.

## M0.4 — Field catalogue and task-specific channels

Raw CFD datasets may expose many quantities with different scientific roles even when shapes are identical.

The dataset defines what exists. The task defines what is used.

Every supported field has explicit semantic metadata, including source, support, role, component structure, units/dimensional convention, and provenance when known.

Input/output/conditioning fields are selected explicitly by name and component order. Readers should load only requested fields where possible.

Stored and derived quantities remain distinguishable.

## M0.5 — Physical nondimensionalization and regime conditioning

Preprocessing follows:

\[
\boxed{
\text{dimensional CFD}
\rightarrow
\text{physical nondimensionalization}
\rightarrow
\text{training-set statistical scaling}
}
\]

Reference-scale numerical values may differ between cases, but their semantic definitions must be explicit and consistent.

Reference quantities/conditioning must be available at inference and must not introduce target leakage.

Training-set scalers are computed only from training data and remain frozen afterwards.

Local physical state, geometry/resolution, and global regime conditioning remain distinct concepts.

## M0.6 — Node-renumbering equivariance

Node indices are implementation identifiers rather than physical quantities.

For a consistent node permutation `P`, graph models must satisfy up to numerical tolerance:

\[
f(PX,PR,PAP^T,\ldots)=P f(X,R,A,\ldots).
\]

This preserves mesh structure while changing only labels. Raw node IDs, storage order, and edge-list order must not carry scientific meaning unless explicitly documented.

## M0.7 — Ownership boundaries

\[
\boxed{
\begin{aligned}
\text{Data} &\rightarrow \text{what exists}\\
\text{Geometry} &\rightarrow \text{how space is related}\\
\text{Task} &\rightarrow \text{what is learned}\\
\text{Model} &\rightarrow \text{how it is learned}\\
\text{Trainer} &\rightarrow \text{how it is optimized}
\end{aligned}
}
\]

A model change should not require rewriting raw file readers. A task change should not require rewriting geometry algorithms. A file-format change should not require modifying scientific model equations.

## M0.8 — Performance and scalability

Priority:

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

Native PyTorch is preferred only when adequate for the target workload. Specialized libraries are acceptable when they provide meaningful capability/efficiency advantages.

Performance evidence levels are:

- `ANALYTICAL`;
- `LOCAL_BENCHMARK`;
- `TARGET_VALIDATED`.

Target performance must not be claimed before target-cluster measurement.

## M0.9 — Scientific provenance and citation

Scientific mechanisms distinguish:

- established method;
- project adaptation;
- project hypothesis;
- measured result.

Significant methods preserve reference, mathematical definition, repository modification, implementation path, validation path, and evidence status.

No mathematically meaningful implementation may exist only in code.

## M0.10 — Scientific change policy

Changes are classified as:

- software-only;
- numerically meaningful;
- scientifically meaningful.

Numerically meaningful changes update numerical conventions/tests as needed.

Scientifically meaningful changes update the scientific specification, traceability mapping, and scientific tests.

Architecture changes should normally be introduced and ablated one at a time.
