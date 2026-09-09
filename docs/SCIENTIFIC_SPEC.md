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

## 16. M5 deterministic node-regression baseline contract

The first concrete task contract is deterministic node-to-node regression on a packed collection of CFD mesh nodes. The task selects ordered named input and target fields from the `FieldCatalog`; the resulting model-facing channel order is the declared field order followed by each field's declared component order.

For a packed batch, the task representation is:

$$
X \in \mathbb{R}^{N_{\mathrm{total}}\times C_{\mathrm{in}}},
\qquad
Y \in \mathbb{R}^{N_{\mathrm{total}}\times C_{\mathrm{out}}}.
$$

When physical nondimensionalization is enabled, every graph is transformed with its own authoritative M3.3 reference state before task fields are concatenated. A packed microbatch therefore does not imply one shared numerical `rho_ref`, `U_ref`, `L_ref`, or `T_ref` across all constituent cases.

Requested dimensionless regime conditioning is graph-level:

$$
c \in \mathbb{R}^{B\times C_{\mathrm{cond}}}.
$$

A conditioning quantity must be explicitly declared available at inference for every graph. The same conditioning name must also carry the same physical definition across graphs used together; identical names with different definitions are not treated as interchangeable quantities.

The M5 null geometric baseline is a node-local affine map. Without global conditioning:

$$
\boxed{\hat y_i=W x_i+b}
$$

and, when graph-level conditioning is explicitly configured:

$$
\boxed{\hat y_i=W[x_i,c_{g(i)}]+b}.
$$

This baseline intentionally ignores coordinates and connectivity. Its purpose is to validate task semantics, global-conditioning plumbing, packed-versus-independent execution, and node-renumbering equivariance before graph-aware models are introduced. It is not a claim that local affine regression is an adequate CFD model.

For the baseline, a consistent node permutation satisfies exactly up to floating-point arithmetic:

$$
f(PX)=P f(X).
$$

M5 does not define a training loss, variable-mesh statistical weighting, optimizer objective, or train-set statistical scaler. Those numerical/scientific choices must remain explicit and are introduced only after their weighting semantics are frozen.

## 17. M6 deterministic regression objective

For the M6 baseline, statistical standardization is fitted from the training split with equal total influence per physical sample rather than equal influence per mesh node. This prevents mesh resolution alone from changing the statistical preprocessing objective.

For standardized target channel $c$, the node-level regression loss is equal-channel MSE:

$$
\ell_{gi}=\frac{1}{C}\sum_{c=1}^{C}(\hat y_{gic}-\hat y^{\mathrm{target}}_{gic})^2.
$$

The loss is then reduced inside each physical sample before samples are combined. Without physical quadrature weights:

$$
L_g=\frac{1}{N_g}\sum_i\ell_{gi}.
$$

When scientifically justified non-negative node integration weights are supplied:

$$
L_g=\frac{\sum_i w_{gi}\ell_{gi}}{\sum_i w_{gi}}.
$$

The initial optimizer-level statistical objective assigns equal weight to each independent physical sample:

$$
\boxed{L_{\mathrm{opt}}=\frac{1}{G}\sum_{g=1}^{G}L_g}.
$$

Computational microbatch composition is not part of this scientific objective. Splitting the same selected physical samples into different node/edge-budget microbatches must preserve the optimizer gradient within numerical tolerance. The same global sample objective must also be preserved across DDP ranks with unequal sample counts.

This equal-sample objective is the initial project convention, not a universal CFD loss prescription. Future tasks may require physical sample weights, uncertainty weighting, surface/volume-specific quadrature, or different channel metrics; each such change is a separate scientifically meaningful task definition.

## 18. M8 sparse one-hop transformer attention

M8 is a project adaptation of Transformer-style scaled dot-product attention to explicitly supplied sparse graph neighborhoods. The score mechanism follows Vaswani et al., *Attention Is All You Need* (NeurIPS 2017, arXiv:1706.03762). Neighborhood-restricted attention is consistent with established graph-attention practice; Veličković et al., *Graph Attention Networks* (ICLR 2018, arXiv:1710.10903) provides relevant graph-neighborhood context, although M8 uses scaled dot-product rather than additive GAT scoring.

For a supplied directed edge `j -> i`, head `h` uses

$$
q_i^{(h)}=W_Q^{(h)}h_i,
\qquad
k_j^{(h)}=W_K^{(h)}h_j,
\qquad
v_j^{(h)}=W_V^{(h)}h_j,
$$

with score

$$
s_{ij}^{(h)}=
\frac{q_i^{(h)\mathsf T}k_j^{(h)}}{\sqrt{d_h}}.
$$

Normalization is performed only over supplied incoming edges for target node `i`:

$$
\alpha_{ij}^{(h)}=
\frac{\exp(s_{ij}^{(h)})}
{\sum_{\ell:(\ell\rightarrow i)\in E}\exp(s_{i\ell}^{(h)})},
$$

and the head message is

$$
m_i^{(h)}=
\sum_{j:(j\rightarrow i)\in E}
\alpha_{ij}^{(h)}v_j^{(h)}.
$$

The block uses pre-normalization residual updates:

$$
\tilde h=h+\operatorname{SparseMHA}(\operatorname{LN}(h),E),
$$

$$
h'=\tilde h+\operatorname{MLP}(\operatorname{LN}(\tilde h)).
$$

M8 consumes `edge_index` as supplied. It does not add self-loops, symmetrize edges, deduplicate edges, create cross-sample edges, or use coordinates/geometric edge features. A node with no incoming attention edges receives a zero sparse-attention message; its local state remains available through the residual path. Explicit self-loops, if supplied by a future geometry transform, are treated as ordinary edges.

The absence of geometry in M8 is intentional. M8 establishes the scientific and computational effect of one-hop sparse attention in isolation. Relative position, distance, mesh scale, directional information, or other geometric attention terms are separate M9 scientific changes and must not be attributed to M8.

The M8 model must remain equivariant to consistent node renumbering and numerically insensitive to pure edge-list reordering within the tolerance expected from sparse floating-point reductions. Packed disconnected execution must agree with independent-graph execution when the supplied topology contains no cross-sample edges.

## 19. M9 relative-displacement geometric attention

M9 extends the M8 attention score with one learned geometric term and leaves the supplied topology and value aggregation unchanged.

For directed edge `j -> i`, the geometry layer defines the target-to-source displacement

$$
\Delta r_{ij}=r_j-r_i.
$$

For physically nondimensionalized tasks this is evaluated from the already nondimensional coordinates, so

$$
\Delta r^*_{ij}=\frac{r_j-r_i}{L_{\mathrm{ref}}}.
$$

Each attention layer maps the displacement to one additive bias per attention head:

$$
b_{ij}=\phi_\theta(\Delta r_{ij})\in\mathbb R^{H_{\mathrm{heads}}},
$$

and uses

$$
\boxed{
s_{ij}^{(h)}=
\frac{q_i^{(h)\mathsf T}k_j^{(h)}}{\sqrt{d_h}}
+b_{ij}^{(h)}.
}
$$

The geometry input is **relative displacement only**. Euclidean distance is not concatenated because it is mathematically derivable from the displacement vector; adding explicit distance later is a separate inductive-bias ablation. Absolute coordinates, normalized direction, local mesh scale, metric tensors, and geometry-conditioned value vectors are not part of the M9 baseline.

Because a global translation cancels in `r_j-r_i`, the geometric contribution is translation invariant. M9 does not claim rotation, reflection, or scale invariance/equivariance: the Cartesian displacement components change under those transformations and the learned geometry map is an ordinary MLP.

The geometry map is a per-layer MLP with hidden width equal to the attention head count. This keeps its edge activation `O(E * num_heads)` rather than introducing a second `O(E * hidden_dim)` geometric representation. The final geometry projection has no additive bias because a constant per-head score shift shared by all incoming edges cancels under softmax.

M9 is a project adaptation combining Transformer scaled dot-product attention with explicit relative mesh geometry. Pfaff et al., *Learning Mesh-Based Simulation with Graph Networks*, ICLR 2021, arXiv:2010.03409, provides relevant precedent for using relative mesh-position information as learned graph-edge geometry. The exact additive per-head score-bias equation above is the repository's project-specific formulation.

## 20. M10 controlled HIT-slice learning ablation

M10 introduces no new attention equation. It defines a controlled experiment for comparing the already frozen M8 and M9 backbones on the same physical data, preprocessing, topology, optimizer settings, and split.

The first task is

$$
\boxed{[\rho u,\rho v,\rho w,\rho E]\rightarrow\rho}.
$$

The source data are precomputed fixed-grid 2-D slices generated by the `diffusion4avbp` Cartesian-slice preprocessing pipeline. Each artifact stores the five conservative fields plus explicit metadata describing its source 3-D snapshot, slicing axis, actual slice coordinate, base slice index, channel names, coordinate policy, and fixed grid shape. Scientific interpretation is taken from this metadata, not inferred from filename tokens.

All slices sharing one `source_stem` are treated as one indivisible split group. The reference split uses seed 42, 70% of source groups for training, 15% for validation, and the remaining groups for testing. This reproduces the source project's grouped-split convention and prevents correlated slices from one 3-D snapshot appearing in multiple splits.

The source artifact defines one shared canonical 2-D coordinate system for every slice:

$$
\boxed{r_i=(x_i,y_i)\in\mathbb R^2.}
$$

Those stored coordinates are used directly by M10. The extraction metadata `axis`, `axis_id`, `slice_index`, `slice_coordinate`, and `base_slice_index` are provenance only and are not embedded into model coordinates or supplied as conditioning in the first ablation.

For the geometry-aware M9 model, relative displacement is therefore evaluated in the canonical 2-D slice frame:

$$
\Delta r_{ij}=
\begin{bmatrix}
x_j-x_i\\
y_j-y_i
\end{bmatrix},
$$

so `spatial_dim=2` for this experiment.

This choice deliberately prevents M9 from receiving hidden slice-orientation information that M8 does not receive. Reconstructing a 3-D coordinate from the extraction plane would make one displacement component identically zero according to slice orientation, allowing M9 to infer whether a sample came from an x-, y-, or z-normal plane. That is outside the intended first M8-vs-M9 comparison.

The physical state still contains the three global momentum components `rhou`, `rhov`, and `rhow`; M10 is therefore a 2-D spatial slice of a 3-D state rather than a 2-D CFD simulation. The canonical 2-D coordinate frame is not claimed to preserve the original global 3-D directional basis. Explicit slice-orientation conditioning or slice-local vector rotation, if later desired, is a separate scientific ablation.

Because the slice artifact does not contain graph connectivity, the first experiment uses a deterministic non-periodic bidirectional Cartesian 4-neighbour graph. For logical node `(a,b)`, valid immediate neighbours are `(a+1,b)`, `(a-1,b)`, `(a,b+1)`, and `(a,b-1)`. No diagonal edges, self-loops, random edges, k-hop augmentation, or periodic wrap edges are added in this initial comparison.

The preprocessing and loss are inherited unchanged from M3.3 and M6:

$$
\text{raw conservative slice fields}
\rightarrow
\text{HIT physical nondimensionalization}
\rightarrow
\text{train-only sample-balanced statistical scaling}
\rightarrow
\text{sample-reduced MSE}.
$$

The M8-versus-M9 scientific hypothesis is that the M9 relative-displacement score bias can improve held-out prediction quality over topology-only M8 attention under otherwise matched conditions. The first run is a single-seed controlled ablation, not a robustness claim. If the measured difference is small, multiple independent seeds are required before attributing it to the architecture.

## 21. M11 unconditional HIT-slice flow matching

M11 introduces the first generative objective while keeping the M8/M9 sparse attention equations and the M10 slice geometry/topology unchanged. The generated state is the complete ordered conservative vector

$$
\boxed{x=[\rho,\rho u,\rho v,\rho w,\rho E].}
$$

The data endpoint is constructed by physical nondimensionalization followed by train-only sample-balanced statistical standardization. In that standardized state space, M11 samples

$$
x_0\sim\mathcal N(0,I),\qquad t_g\sim\mathcal U(0,1)
$$

independently for each physical graph `g`. The same time is broadcast to all nodes of one graph. The straight conditional path and its target velocity are

$$
\boxed{x_{t,gi}=(1-t_g)x_{0,gi}+t_gx_{1,gi}},
$$

$$
\boxed{v^*_{gi}=x_{1,gi}-x_{0,gi}}.
$$

The model learns the vector field

$$
v_\theta(x_t,t,G,R).
$$

The first baseline supplies raw scalar `t` as one graph-level conditioning channel through the existing M8/M9 conditioning path. No sinusoidal/Fourier time embedding, learned time MLP, AdaLN, or per-layer time modulation is part of the frozen M11 baseline.

The velocity objective reuses the M6 sample reduction. With five equal-weight state channels and no physical quadrature weights,

$$
\ell_{gi}=\frac{1}{5}\|v_\theta(x_{t,gi},t_g)-v^*_{gi}\|_2^2,
$$

$$
L_g=\frac{1}{N_g}\sum_i\ell_{gi},
\qquad
\boxed{L=\frac{1}{B}\sum_g L_g}.
$$

Validation times and Gaussian sources are deterministic functions of `(validation_seed, sample_id)` so validation does not depend on batch order. Sampling starts from a deterministic sample-ID-keyed Gaussian source and integrates `dx/dt=v_theta` from `t=0` to `t=1` with either explicit Euler or Heun; the reference inference setting is Heun with 50 uniform steps.

M11 does not define a diffusion beta schedule, discrete diffusion timestep, epsilon/score/x0 target, SNR weighting, DDPM sampler, or DDIM sampler. Those remain a separate later diffusion task so the first generative experiment changes only the task mathematics needed for straight-path flow matching.

## 22. M13 unconditional HIT-slice diffusion

M13 adds a diffusion task while leaving the M9/M12 graph Transformer equations unchanged. It is a project adaptation of the DDPM formulation of Ho et al., *Denoising Diffusion Probabilistic Models* (NeurIPS 2020, arXiv:2006.11239), the cosine cumulative-noise schedule introduced by Nichol and Dhariwal, *Improved Denoising Diffusion Probabilistic Models* (ICML 2021, arXiv:2102.09672), and the generalized DDIM reverse process of Song et al., *Denoising Diffusion Implicit Models* (ICLR 2021, arXiv:2010.02502).

The generated state is the same ordered five-channel conservative vector used by M11:

$$
\boxed{x_0=[\rho,\rho u,\rho v,\rho w,\rho E].}
$$

Physical nondimensionalization and train-only sample-balanced statistical standardization are applied before diffusion noising. The frozen first baseline uses `T=1000` discrete timesteps, cosine offset `s=0.008`, and epsilon prediction. One timestep is sampled independently per physical graph:

$$
t_g\sim\mathcal U\{1,\ldots,T\},
\qquad
\epsilon_{gi}\sim\mathcal N(0,I).
$$

With cumulative signal coefficient $\bar\alpha_t$, the task constructs

$$
\boxed{
x_{t,gi}=\sqrt{\bar\alpha_{t_g}}x_{0,gi}
+\sqrt{1-\bar\alpha_{t_g}}\epsilon_{gi}.
}
$$

The network predicts the exact sampled noise:

$$
\epsilon_\theta(x_t,\tau,G,R),
\qquad
\tau_g=\frac{t_g}{T}\in(0,1].
$$

The normalized scalar `t/T` is appended through the existing graph-level conditioning path. This is an intentional controlled-comparison choice: the generative process changes from M11 flow matching to diffusion without simultaneously introducing sinusoidal/Fourier embeddings, a learned time MLP, AdaLN, or per-layer time modulation.

The objective reuses the M6 equal-sample reduction. For five equal-weight channels,

$$
\ell_{gi}=\frac{1}{5}
\left\|\epsilon_\theta(x_{t,gi},\tau_g)-\epsilon_{gi}\right\|_2^2,
$$

$$
L_g=\frac{1}{N_g}\sum_i\ell_{gi},
\qquad
\boxed{L=\frac{1}{B}\sum_g L_g}.
$$

Validation uses deterministic `(t,epsilon)` pairs derived from `(validation_seed, sample_id)` so validation is independent of loader batch order. Model selection uses validation epsilon MSE. The held-out test epsilon MSE is a denoising-objective diagnostic; generative quality is evaluated separately from independently generated samples.

For reverse sampling, the task first reconstructs

$$
\hat x_0=
\frac{x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon}
{\sqrt{\bar\alpha_t}}.
$$

For a selected previous timestep $t'<t$, generalized DDIM uses

$$
\sigma_t=\eta
\sqrt{
\frac{1-\bar\alpha_{t'}}{1-\bar\alpha_t}
\left(1-\frac{\bar\alpha_t}{\bar\alpha_{t'}}\right)
},
$$

$$
x_{t'}=
\sqrt{\bar\alpha_{t'}}\hat x_0
+\sqrt{1-\bar\alpha_{t'}-\sigma_t^2}\,\hat\epsilon
+\sigma_t z,
\qquad z\sim\mathcal N(0,I).
$$

The sampling stochasticity parameter is restricted to $0\le\eta\le1$. `eta=0` is deterministic DDIM. `eta=1` with a reduced timestep grid is stochastic accelerated DDIM. Only `eta=1` while visiting every one of the `T` reverse transitions is labelled the ancestral-DDPM limit.

Training and reverse generation are deliberately separate workflows. One trained checkpoint can therefore be evaluated with multiple `(steps, eta, seed)` choices without retraining. Generation reloads the exact saved train-only standardizers and the exact stored test split rather than refitting or resplitting. Generated IDs are independent deterministic RNG keys (`gen_...`); held-out CFD test IDs are stored separately and do not define generated-to-reference target pairs.

For the first controlled M11-versus-M13 sampling-cost comparison, the reference diffusion setting is deterministic DDIM with 100 model evaluations. This is compared against M11 Heun with 50 steps, which also uses 100 model evaluations. Full `T=1000, eta=1` ancestral sampling is a separate cost/quality reference, not the equal-NFE baseline.

M13 does not initially include SNR-weighted loss, `v` prediction, `x0` prediction, learned reverse variance, classifier-free guidance, conditional generation, x0 clipping, richer time embeddings, DDP, or low-precision target validation. Those are separate scientific or numerical changes and must not be attributed to the frozen baseline.
