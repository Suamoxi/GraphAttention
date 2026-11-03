# M45: Static and dynamic two-scale DiNAT — qualification implementation

## Scientific contract

HIT only, periodic nested Cartesian coarse/fine meshes. For an integer factor
`r >= 2`, `n_f = r * n_c` unique periodic nodes on each spatial axis, with
`h_f = h_c/r`. The physical data producer (`dnstoles`) supplies **Q** and
**B**; this graph module does *not* re-derive, reinterpret, or assume a filter
width. When `Q B = I`, `U_c=Q U_f` and `R_f=U_f-B U_c` imply `Q R_f=0`.
A supplied pair must be validated against the selected physical operators.

The graph model receives preprocessed, dimensionless/standardized inputs:

- `C_c = (U_c - mean_c)/std_c` (`[N_c,5]`)
- `C_f = (B U_c - mean_c)/std_c` (`[N_f,5]`)
- `X_1 = (U_f - B U_c)/rms_R` (`[N_f,5]`, training only)
- Flow Matching `X_t=(1-t) epsilon+t X_1`, `v*=X_1-epsilon`
- `s_r=log2(r)`, a single dimensionless, **provisional** refinement descriptor
- global, L_ref-normalized coordinates, the periodic box length, fine and
  coarse DiNAT one-hop and exact-two-hop edges, and sparse bipartite supports.

The neural network returns only the five fine normalized FM velocity channels.
The physical output is reconstructed outside the model as `B U_c+rms_R*X_1`.
The first implementation does not project FM noise/velocity into `ker Q`;
therefore generated coarse consistency is a **diagnostic**, not a guarantee.

## Two architectures

Both use the same fine preprocessing and decoder:

1. `H_c^0 = E_c(C_c, s_r)` — coarse DiNAT encoder, no FM time.
2. `H_f^1 = T_f_pre([X_t,C_f], [t,s_r])` — fine DiNAT encoder.
3. Static: `H_c = H_c^0`.
4. Dynamic: `Z_c=A_down(H_c^0,H_f^1)` (coarse queries, fine keys/values),
   `H_c=T_c_dynamic(H_c^0+Z_c,[t,s_r])` — time-conditioned coarse DiNAT.
5. `Z_f=A_up(H_f^1,H_c)` (fine queries, coarse keys/values).
6. `v=T_f_post(H_f^1+Z_f,[t,s_r])` — fine DiNAT velocity head.

Both cross-attentions use geometry-defined sparse bipartite supports, a
minimum-image relative displacement bias normalized by coarse spacing, and
`log(interpolation_weight)` as a score prior. This feature-transfer topology
is **not** the physical Q or B. Each coarse query's fine support is normalized
independently by attention softmax. The same cross-scale support is used in
opposite directions for the initial HIT qualification; more general meshes may
need different upward/downward supports.

The generic `AlternatingDilatedGeometricDiT` is used for each transformer.
Feature-encoder output heads have a nonzero initial projection (unlike the
zero-initialized final FM velocity head) so features are not identically zero
at initialization. Absolute coordinate embedding and patch-wise bounding-box
normalization are disabled to allow consistent patch/full-graph coordinates;
periodic displacements use the minimum-image convention.

## Partition design

For fine owned nodes `O_f`, backwards from the velocity outputs:

    fine_post = DiNAT_post_dependencies(O_f)
    coarse_up = keys_for_fine_queries(fine_post)
    if dynamic:
        coarse_dynamic_inputs = DiNAT_coarse_dynamic_dependencies(coarse_up)
        fine_down = keys_for_coarse_queries(coarse_dynamic_inputs)
        coarse_static_inputs = DiNAT_coarse_static_dependencies(coarse_dynamic_inputs)
    else:
        fine_down = empty
        coarse_static_inputs = DiNAT_coarse_static_dependencies(coarse_up)
    fine_read = DiNAT_pre_dependencies(fine_post union fine_down)

The training loss is evaluated only on `O_f`. All temporary halo outputs may
be computed, but halo nodes have **zero loss weight**. Shared coarse nodes can
be read/recomputed by multiple fine partitions; there is **one global owner**
for each coarse node during synchronized inference. Stage-specific dependency
sets need not contain all bipartite sources for *irrelevant* halo queries;
those outputs must not influence owned outputs. Cross-attention returns zero
on queries with no present keys, rather than producing NaNs.

`two_scale_patch_dependencies` and `extract_two_scale_patch_geometry` implement
small-graph **qualification** of these semantics. They assume the global
one-hop and exact-two-hop edges already exist; induced local two-hop graphs
are taken from those original edges, never recalculated after cropping.

Production-sized `257^3` meshes must not use `build_hit_two_scale_geometry`:
it has an explicit small-graph size guard. Use target-driven M41 partition
construction and lazy geometry generation instead of allocating the complete
exact-two-hop graph on the CPU.

## Current implementation and remaining work

Implemented in this first stage:

- Periodic nested Cartesian HIT mesh and sparse scale-transfer topology
  (small-graph qualification only).
- Static and dynamic DiNAT vector-field modules, with separate coarse/fine
  encoders and sparse bipartite attention.
- Refinement-only `[t, log2(r)]` vs `[log2(r)]` conditioning.
- Backward dependency closure and small-graph local geometry extraction.
- Periodic displacement support for ordinary DiNAT forward.
- Unit tests for topology, interpolation priors, gradients and closure.

Not yet implemented:

- A production `dnstoles` Q/B dataset loader or validation of the chosen
  sparse Q/B against actual HIT snapshots.
- An M44-compatible paired, core-masked FM training runner or checkpoint
  orchestration; the new models are not yet used in an M44 job.
- Exact small-graph forward/gradient parity with the full **real** M37
  backbone (required before using local patches for a scientific run).
- Stage-synchronized and disk-streamed *periodic* M41 inference. Existing
  M41 partition edge displacement handling needs the same minimum-image
  convention added to the ordinary DiNAT forward.
- Exact coarse consistency projection, especially for Q/B with nonlocal
  right-inverse solves.
- Multi-ratio training, capacity-matched static/dynamic comparisons, and
  physical turbulence statistics validation.

Do not report either architecture as trained or large-mesh qualified until
these gates have been completed.

