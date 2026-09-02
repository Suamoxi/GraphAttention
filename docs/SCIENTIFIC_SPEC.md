# Scientific Specification

## 1. Scientific scope

This repository is a scientific machine-learning framework for modelling CFD fields on meshes of varying size, resolution, and topology, with a focus on efficient sparse geometric neural architectures.

The framework may support:

- deterministic regression;
- wall-quantity prediction;
- super-resolution;
- operator learning;
- unconditional or conditional generative modelling;
- diffusion;
- flow matching.

The repository is not defined as an operator-learning framework. Some tasks may learn deterministic operators, while generative tasks may instead learn probability distributions, denoising functions, scores, or vector fields.

The fundamental scientific abstraction is:

$$
\boxed{\text{CFD field} + \text{mesh geometry/topology} + \text{task}}
$$

## 2. Mesh-agnostic versus mesh-free

The framework is **mesh-agnostic**, not mesh-free.

The supplied CFD mesh is scientifically meaningful input. Native connectivity and geometry should be preserved unless a task or controlled ablation explicitly replaces or augments them.

Learned parameters must not depend on:

- mesh identity;
- node count;
- edge count;
- exporter node numbering;
- a fixed topology.

The same learned weights should be applicable to different meshes whose field semantics and task definition are compatible.

## 3. Initial graph entity

For the initial framework:

$$
\boxed{\text{one graph vertex} = \text{one CFD mesh node}}
$$

Cells and faces may provide:

- volume/area weights;
- normals;
- boundary tags;
- local metrics;
- connectivity information;
- derived edge relationships.

They are not initially separate learned graph entities.

Adding cell-centred, face-centred, heterogeneous, or dual-graph representations is a scientific/architectural extension and must be documented and tested as such.

## 4. Native topology and geometry

The model may use native mesh connectivity together with geometric information.

For an edge `i -> j`, common geometric quantities include:

$$
\Delta r_{ij} = r_j-r_i,
$$

$$
d_{ij}=\|\Delta r_{ij}\|,
$$

$$
\hat r_{ij}=\frac{\Delta r_{ij}}{d_{ij}}.
$$

Additional scientifically meaningful geometric quantities may include:

- wall distance;
- surface normal;
- cell/node control volume;
- face area;
- local mesh scale;
- directional metric information;
- boundary type.

The existence of a quantity in a file does not automatically make it a model input. Its role must be explicit.

## 5. Field catalogue

Raw CFD outputs can contain many arrays with identical shapes but different meanings. Dataset shape is therefore not a sufficient scientific descriptor.

Every supported raw quantity belongs to a field catalogue that conceptually records at least:

- canonical name;
- source path;
- spatial support: node/cell/face/global;
- semantic role;
- component structure;
- units or dimensional convention when known;
- provenance;
- whether it is stored or framework-derived.

### 5.1 Semantic roles

Relevant roles include:

- primary physical state;
- species state;
- auxiliary physical quantity;
- derived physical quantity;
- geometry/boundary quantity;
- diagnostic;
- solver/computational metadata;
- forcing/internal state;
- global metadata.

For an AVBP/HIT-style file, for example:

- `/GaseousPhase/rho`, `rhou`, `rhov`, `rhow`, `rhoE` are primary state;
- `/RhoSpecies/*` are species state;
- quantities such as pressure/temperature/viscosity are auxiliary or derived physical fields;
- solver residuals are diagnostics;
- `mpi_rank` is computational metadata;
- restart forcing state is not automatically a physical model input.

## 6. Task-specific channel selection

The dataset defines **what exists**.

The task defines **what is used**.

The model input channel count is task-dependent:

$$
C = C_{\mathrm{task}}.
$$

Channel semantics and order must be explicit by field/component name. Scientific behavior must never depend on anonymous index ranges such as “channels 0:5” without an accompanying semantic contract.

The framework must support scalar, vector, and tensor groups even when their components are stored separately.

For example, `rhou`, `rhov`, and `rhow` are components of a momentum vector, not three unrelated scalar concepts.

## 7. Stored versus derived quantities

Stored and derived quantities must remain distinguishable.

For example, pressure read directly from a solver output and pressure reconstructed from a conservative state may be numerically close but have different provenance.

Any derived quantity must have an explicit, documented transformation and input dependency.

No silent substitution between stored and derived variants is permitted.

## 8. Physical nondimensionalization

Raw dimensional magnitudes from different CFD cases must not be assumed directly comparable.

The required conceptual sequence is:

$$
\boxed{
\text{dimensional CFD}
\rightarrow
\text{physical nondimensionalization}
\rightarrow
\text{training-set statistical scaling}
}
$$

Each case defines physically meaningful reference quantities appropriate to the problem, such as:

$$
U_{\mathrm{ref}},\quad
L_{\mathrm{ref}},\quad
\rho_{\mathrm{ref}},\quad
T_{\mathrm{ref}},\quad
p_{\mathrm{ref}}.
$$

Reference values may vary numerically between cases. Their physical definition and role must be explicit.

### 8.1 Example transformations

Velocity:

$$
\mathbf u^*=\frac{\mathbf u}{U_{\mathrm{ref}}}.
$$

Coordinates:

$$
\mathbf x^*=\frac{\mathbf x}{L_{\mathrm{ref}}}.
$$

Density:

$$
\rho^*=\frac{\rho}{\rho_{\mathrm{ref}}}.
$$

Momentum density:

$$
(\rho\mathbf u)^*=
\frac{\rho\mathbf u}{\rho_{\mathrm{ref}}U_{\mathrm{ref}}}.
$$

Energy density, when consistent with the chosen energy definition:

$$
(\rho E)^*=
\frac{\rho E}{\rho_{\mathrm{ref}}U_{\mathrm{ref}}^2}.
$$

Pressure fluctuation/offset form:

$$
p'^*=
\frac{p-p_{\mathrm{ref}}}{\rho_{\mathrm{ref}}U_{\mathrm{ref}}^2}.
$$

Rate-of-strain:

$$
S_{ij}^*=\frac{L_{\mathrm{ref}}}{U_{\mathrm{ref}}}S_{ij}.
$$

The precise transformation for each field is part of the scientific specification and must not be inferred only from dimensional intuition inside implementation code.

## 9. Reference-scale semantics and inference availability

Reference definitions are part of the scientific model.

A scale defined as a bulk velocity, outer velocity, forcing velocity, RMS velocity, friction velocity, or another characteristic velocity is not interchangeable with the others merely because the units match.

A preprocessing specification must not silently change the semantic definition of a reference quantity across train/validation/test/inference.

Every reference quantity and conditioning variable must be computable from information legitimately available at inference for the task.

Do not use unavailable target information, future information, or hidden DNS quantities to normalize or condition a prediction that would not have access to those quantities in deployment.

Reference quantities are case- or operating-condition-level by default. Snapshot-dependent references are allowed only when snapshot-wise invariance is an explicit scientific objective.

## 10. Physical-regime conditioning

Dimensionless regime parameters may be provided as global conditioning when they distinguish physical regimes represented in the dataset.

Examples include:

$$
Re_{\mathrm{ref}}=
\frac{\rho_{\mathrm{ref}}U_{\mathrm{ref}}L_{\mathrm{ref}}}{\mu_{\mathrm{ref}}},
$$

$$
Ma_{\mathrm{ref}}=\frac{U_{\mathrm{ref}}}{a_{\mathrm{ref}}}.
$$

Other candidates such as `Pr`, `gamma`, forcing parameters, or chemistry parameters should be included only when relevant to the governing physics and meaningfully variable across cases.

Flow-family-specific descriptors such as `Re_lambda`, `Re_tau`, or `Re_theta` do not replace the general reference-scale system. They may be additional metadata or conditioning where consistently defined.

## 11. Heterogeneous flow families

Different geometries or flow families may require different concrete definitions of reference scales.

A channel half-height, nozzle diameter, chord, energy-injection length, or another scale may all serve as `L_ref` in different flow families if their definitions are explicit and scientifically justified.

The mapping from flow family to reference definition must be explicit metadata and must not be inferred from filenames, tensor dimensions, or mesh size.

## 12. Geometry, resolution, and regime are distinct

Dimensionless regime parameters do not replace explicit geometry or mesh-resolution information.

For multiresolution or LES applications, resolution must remain explicitly representable, for example:

$$
\Delta_i^*=\frac{\Delta_i}{L_{\mathrm{ref}}},
$$

where `Delta_i` may be global, local, directional, volume-derived, or metric-derived depending on the mesh and task.

M0 does not prescribe one universal resolution metric. It requires the metric and its definition to be explicit when used.

The model interface conceptually distinguishes:

$$
\boxed{\text{local dimensionless physical state}}
$$

$$
\boxed{\text{geometry and resolution information}}
$$

$$
\boxed{\text{global physical-regime conditioning}}
$$

These categories may interact in the learned architecture but must not be implicitly conflated during preprocessing.

## 13. Node-renumbering equivariance

Node numbering is not a physical property.

Let `P` be a permutation matrix corresponding to a consistent renumbering of mesh nodes. A node-level graph model must satisfy, up to documented numerical tolerance:

$$
f(PX, PR, PAP^T, \ldots)=P f(X,R,A,\ldots).
$$

Here:

- `X` represents node-associated physical state;
- `R` represents node coordinates or node-associated geometry;
- `A` represents graph connectivity;
- all node/edge-associated metadata is transformed consistently.

This requirement preserves the physical mesh. It only changes implementation labels.

Forbidden hidden dependencies include:

- learned embeddings indexed by raw node ID;
- selecting special nodes solely because they appear first in storage order;
- topology augmentation that changes physically under a pure node relabelling;
- scientific behavior depending on edge-list order.

Edge ordering may still cause tiny floating-point reduction differences; tests should use an appropriate numerical tolerance.

## 14. Other invariances/equivariances

Translation, rotation, reflection, and scale invariance/equivariance are not assumed automatically.

If an architecture claims one of these properties, the claim must include:

- mathematical justification;
- assumptions;
- implementation traceability;
- a scientific property test.

For example, using only relative displacement can provide translation-invariant geometric relationships, while absolute coordinates do not.

## 15. Scientific-task independence

The backbone must not assume a unique learning objective.

Examples:

### Deterministic regression

$$
X\rightarrow Y.
$$

### Temporal increment prediction

$$
X_t\rightarrow X_{t+1}-X_t.
$$

### Super-resolution

$$
X_c \rightarrow X_f
$$

or preferably in some settings:

$$
X_f = \mathcal I(X_c)+\delta X_\theta.
$$

### Conditional generative modelling

$$
p_\theta(X_f\mid X_c,G_c,G_f).
$$

### Diffusion

The task constructs noisy states and the appropriate denoising/score/velocity target according to an explicitly specified parameterization.

### Flow matching

The task constructs the interpolation path and target vector field according to an explicitly specified formulation.

The same sparse geometric backbone may support several tasks, but task-specific mathematics must remain in the task layer.
