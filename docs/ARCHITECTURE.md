# Architecture

## 1. Purpose

This document defines the software boundaries and dependency direction of the CFD mesh scientific machine-learning framework.

The repository is designed around:

$$
\boxed{\text{CFD field} + \text{mesh geometry/topology} + \text{task}}
$$

It is not organized around a single learning paradigm such as autoregressive simulation, operator learning, or diffusion. These are tasks that consume common scientific data and geometry infrastructure.

## 2. Architectural principles

The architecture must support:

- arbitrary mesh node counts within model/backend limits;
- arbitrary supplied mesh connectivity;
- multiple meshes in one training run;
- different meshes in one packed microbatch when computationally feasible;
- task-independent model backbones;
- task-specific targets and conditioning;
- explicit physical preprocessing;
- sparse computation without default padding;
- reproducible experiments;
- evidence-based performance evaluation.

## 3. Ownership model

The repository is divided conceptually into five main layers.

### 3.1 Data — what exists

The data layer represents information provided by the CFD data source.

Responsibilities:

- file format readers;
- field discovery/catalogue;
- loading explicitly requested fields;
- native mesh coordinates/connectivity;
- boundary and case metadata supplied by the simulation;
- case reference quantities;
- split manifests and sample identity;
- data provenance.

Non-responsibilities:

- architectural graph dilation;
- random/long-range attention edges;
- diffusion noising;
- learning targets;
- optimizer/training logic.

A file format change should be handled here without requiring scientific model equations to change.

### 3.2 Geometry — how space is related

The geometry layer transforms supplied mesh information into deterministic spatial/topological structures.

Responsibilities may include:

- canonical `edge_index` construction;
- relative edge displacement;
- distances and local geometric descriptors;
- exact-hop edge sets;
- deterministic sparse topology augmentation;
- graph partitioning when required;
- local or directional resolution measures;
- source-to-query, volume-to-wall, or coarse-to-fine correspondences.

The geometry layer does not define training targets or optimization.

A geometry function may define how a 2-hop edge set is constructed. Whether a model layer uses that edge set is a model/configuration decision.

### 3.3 Task — what is learned

The task layer defines the scientific objective.

Responsibilities:

- input field selection and order;
- target field selection and order;
- conditioning-variable selection;
- temporal-difference targets;
- supervised regression targets;
- wall-query targets;
- super-resolution residual targets;
- diffusion noising/targets;
- flow-matching interpolation/vector-field targets;
- interpretation of model outputs for loss and evaluation.

Changing from deterministic regression to diffusion should not require rewriting the raw dataset reader or sparse-attention implementation.

### 3.4 Model — how the supplied representation is transformed

The model contains trainable transformations.

It receives explicit representations of:

- local physical state;
- sparse topology;
- geometry/resolution information;
- global conditioning where required.

A model must not:

- open raw simulation files;
- infer field semantics from integer indices;
- fit preprocessing statistics;
- choose targets;
- implement optimizer stepping.

### 3.5 Training — how optimization is executed

The training layer owns:

- backward propagation;
- gradient accumulation;
- AMP/BF16 mechanics;
- DDP synchronization;
- optimizer and scheduler stepping;
- checkpoint orchestration;
- runtime logging.

The training layer should not contain task-specific physics logic unless no clean task-level interface can represent the requirement.

## 4. Canonical graph representation

For the initial framework, graph vertices represent CFD mesh nodes.

For one graph:

- node data: `x` with shape `[N, C]`;
- coordinates: `coords` with shape `[N, D]`;
- connectivity: `edge_index` with shape `[2, E]`;
- optional edge/geometric attributes;
- optional node quadrature/control-volume weights;
- optional boundary metadata;
- case/sample metadata.

The graph uses supplied CFD connectivity. The model is mesh-agnostic, not mesh-free.

## 5. Packed variable-graph microbatches

The canonical batched representation is a disconnected union of graphs.

For graphs `G_1, ..., G_B`:

$$
N_{\mathrm{total}} = \sum_g N_g,
\qquad
E_{\mathrm{total}} = \sum_g E_g.
$$

Node tensors are concatenated:

$$
X_{\mathrm{packed}} \in \mathbb{R}^{N_{\mathrm{total}}\times C}.
$$

Edges are concatenated with node-index offsets:

$$
\mathrm{edge\_index}_{\mathrm{packed}} \in \mathbb{N}^{2\times E_{\mathrm{total}}}.
$$

Sample membership is retained through metadata such as:

- `batch_index [N_total]`;
- `ptr [B+1]`;
- per-graph metadata lists/tables.

No edges exist between samples unless a future method explicitly introduces cross-sample coupling, which would require a scientific specification change.

## 6. Computational versus statistical batching

A core architecture rule is:

$$
\boxed{\text{microbatch} \neq \text{optimizer batch}}
$$

### 6.1 Microbatch

A microbatch is a computational packing unit constrained by hardware/model limits. Initial hard constraints should include both:

$$
N_{\mathrm{total}} \le N_{\max},
\qquad
E_{\mathrm{total}} \le E_{\max}.
$$

The concrete values are hardware/model-specific configuration, not repository constants.

### 6.2 Optimizer batch

The optimizer batch is a statistical unit, for example a target number or total weight of independent physical samples.

Gradient accumulation may combine multiple microbatches before one optimizer step.

Packing must not alter the intended sample distribution or sample weighting.

### 6.3 Sampler and packer are separate

The sampler determines **which physical samples are selected**.

The packer determines **how selected samples fit into computational microbatches**.

Size bucketing may be used for efficiency, but it must not silently create a resolution-based curriculum or change sampling probabilities.

## 7. Oversized graphs

If one graph exceeds the configured per-device microbatch budget, the packer should fail explicitly with a diagnostic containing at least the graph node/edge counts and configured limits.

Processing a single graph larger than device capacity is a separate scalability problem that may require:

- graph partitioning;
- subgraph sampling;
- domain decomposition;
- hierarchy/multilevel processing;
- distributed single-graph execution.

These mechanisms must not be hidden inside ordinary batching.

## 8. Field semantics and model interfaces

The repository distinguishes conceptually:

$$
X^* = \text{local dimensionless physical state},
$$

$$
G^* = \text{geometry and resolution information},
$$

$$
c = \text{global physical-regime conditioning}.
$$

Runtime containers may store these together for efficiency, but they must remain separately identifiable by metadata and interfaces.

The data layer reports available fields. The task selects ordered input/target/conditioning fields.

## 9. Configuration architecture

Use one Hydra composition tree rather than many specialized top-level launch files.

Illustrative structure:

```yaml
defaults:
  - data: avbp
  - model: baseline
  - task: regression
  - optimizer: adamw
  - trainer: default
  - _self_

seed: 42
run_name: null
```

Prefer Hydra `_target_` / `hydra.utils.instantiate` for composition. Avoid custom registries/factory systems until a real limitation appears.

Configuration options should represent supported scientific or computational decisions. Do not expose speculative knobs.

## 10. Dependency direction

Preferred dependency direction:

```text
readers/data schemas
        ↓
data objects
        ↓
geometry transforms
        ↓
task preparation
        ↓
model
        ↓
training orchestration
```

Utility modules must not become a backdoor for circular dependencies or scientific logic with unclear ownership.

## 11. Initial implementation sequence

The architecture should be implemented in the following order:

1. repository/governance/documentation;
2. core data and geometry contracts;
3. synthetic + minimal AVBP data path;
4. budget-based packed graph batching;
5. task interface and trivial baseline;
6. training/DDP/AMP correctness;
7. benchmark infrastructure;
8. first true sparse 1-hop transformer;
9. geometry-aware attention;
10. dilation/global communication and later task-specific extensions.

The real sparse transformer should not precede the data, batching, statistical-weighting, and benchmark contracts on which its scalability claims depend.
