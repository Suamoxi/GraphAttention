# M48.1–M48.3 — Real MS_SR / CEDAR / TwoScaleDiNAT qualification

**Status:** implemented in GraphAttention `main`; **not yet executed on the
MS_SR deployment or a Grace GPU**. Do not report scientific training or
converged numerical results until the Slurm receipt exists.

## Objective and data contract

Integrate one actual **converged (user assumption)** periodic AVBP conservative
snapshot from the independently published **MS_SR** campaign into both
M45 `TwoScaleDiNAT` architectures. This is an interface, projection and
full-vs-patch **numerical qualification**, **not** a training campaign.

The MS_SR campaign is read-only:

- Manifest: `/scratch/ai/MS_SR/dataset/manifest.json`.
- Reference SHA-256:
  `993f791f88d801e44e5feb0cee927c4c48eaa459bd6171e9e364b002bb01fdca`.
- Pinned CEDAR revision:
  `ae96eece81204b7eb55365eade4838ae1e337b38`.
- MS_SR Python code: `/scratch/ai/MS_SR/code/`.
- Pinned CEDAR package: from a verified Python installation, or an optional\n  source checkout at `/scratch/ai/MS_SR/cedar/src/` if present.\n  The source checkout is **not** required when the pinned `cedar-cfd` is installed.
- Source role: `teacher` only (not validation or `locked_in_regime`).
- Filter: `w1p25_l1` (\(\omega,\lambda)=(1.25,1)\).
- Pair: source-relative factors `(4,8)`, i.e. **q8 -> q4**.
- First source: first chronological `ReT060/` teacher/filter/factor match.
  Snapshot ID and case must come from the actual manifest, not be guessed.
- First physical node grids: \(24^3\to48^3\), 5 conservative channels
  \(\rho,\rho u,\rho v,\rho w,\rho E\).
- Existing 44 teacher source/filter records for this pair are eligible for
  a subsequent training campaign; the present job uses **one** only.

The adapter reuses **MS_SR's `CampaignDataset` and `TorchCEDAR` classes**
from their existing files; no duplicate CEDAR filtering implementation is
introduced. It validates MS_SR role, case/snapshot identity, channel order,
operator hashes, physical spacing and full field shape before evaluation.

**Filesystem gate:** the existing GraphAttention Grace Slurm environment may
not mount `/scratch/ai/MS_SR`, or the GraphAttention Python environment may
not have the MS_SR `cedar-cfd` package installed. The Slurm launcher requires
the real MS_SR data and code, then imports CEDAR and checks MS_SR's pinned
source-file hashes using `preprocess.check_cedar()`. It accepts an installed
CEDAR package, the optional original `cedar/src` checkout, or an explicitly
supplied `M48_CEDAR_SRC`. Never substitute a synthetic field.

## M48.1 — Read-only real pair

`src/graph_attention/data/ms_sr_adapter.py` selects one published pair by
role, filter, factors and case prefix. Converts MS_SR channel-first
`[1,5,Nx,Ny,Nz]` into flat GraphAttention `[N,5]` in global xyz order.

Data sources retain physical double precision until CEDAR qualification.
Full source identities and operator/manifest hashes are included in the
JSON receipt. The adapter does not write to MS_SR or create new HDF5 copies.

## M48.2 — Exact joint CEDAR operator

MS_SR's existing `TorchCEDAR` evaluates restriction \(C\), matched joint
right inverse

\[
P=C^\top(CC^\top)^{-1}
\]

and the full-domain projection \(\Pi=I-PC\).

On a real target \(U_f\) and coarse input \(U_c\):

\[
B=PU_c,\quad R=U_f-B,\quad C U_f\approx U_c,\quad
C B\approx U_c,\quad C R\approx 0,\quad B+R\approx U_f.
\]

The smoke also generates a standard Gaussian \(\varepsilon\) on the
**entire periodic fine lattice** and checks

\[
X_0=\Pi\varepsilon,\qquad C X_0\approx 0,\qquad
\Pi^2\varepsilon\approx\Pi\varepsilon.
\]

Checks are performed in float64 with a per-conservative-component
relative maximum norm and default threshold \(10^{-9}\). This is a
conservative software gate, not a measured precision claim. Record raw
absolute and component-scaled errors in `summary.json`.

No CEDAR projection is applied on cropped or artificially periodized
patches. The CEDAR global inverse may be dense in physical space even
though the graph attention is local.

For the single smoke example, normalization is computed from **that
teacher field alone**, as an interface test. It is not suitable for
multi-field scientific training. Population normalization must be fit
over teacher sources and frozen before model comparison.

## M48.3 — Sparse, target-driven two-scale periodic graph

`src/graph_attention/geometry/two_scale_targeted.py` builds a
`TwoScalePatchGeometry` **without materializing the global fine graph**.

Given a set of global fine-owned nodes, reverse the exact computation
dependency chain, reusing M45's semantics:

1. Fine decoder's 1-hop / exact-2-hop incoming dependencies.
2. Coarse keys of upward fine-query cross attention.
3. For dynamic mode, incoming coarse dynamic layers and fine keys of the
   downward attention. Static mode omits this branch.
4. Incoming coarse static dependencies.
5. Incoming fine pre-encoder dependencies.

Global node IDs and physically scaled global lattice coordinates are
preserved. No new periodic boundaries are created at patch faces. The
actual cross-scale attention prior remains M45's trilinear sparse support
and is **distinct from** the physical CEDAR observation \(C\).

Unit tests compare every dependency set, every incoming edge support
(including periodic seams and exactly-two-hop exclusion) and trilinear
cross-attention weight against the **existing full-graph constructor** on
small grids. They also compare owned-velocity outputs and all parameter
gradients of static/dynamic M37-backed `TwoScaleDiNAT` against a
whole-graph reference. Tolerances follow prior M45 unit gates.

This establishes a small-grid numerical equivalence target, not a
partitioned training or multi-GPU performance result. The large real
field only evaluates one patch's complete backwards dependency halo.

## One-step real-data Flow Matching smoke

Read an entire real pair and compute exact CEDAR residual and globally
projected source. With fieldwise constant per-channel teacher scales:

\[
X_1=D_R^{-1}(U_f-PU_c),\quad
X_0=(I-PC)\varepsilon,\quad
X_t=(1-t)X_0+tX_1,\quad
u_{\mathrm{FM}}=X_1-X_0.
\]

Per-channel scalings commute with this spatial CEDAR operator because
it applies the same scalar filter to all five conservative components.

Take a **small owned fine core crossing the global x-periodic seam**,
construct a complete dependency halo, run one static and one dynamic
`TwoScaleDiNAT` optimizer step at fixed \(t\), and evaluate mean-square
FM loss **only on owned fine nodes**. Inputs and targets are taken from
the already globally prepared field; no local re-projection.

The smoke records walltime for graph construction and each optimizer
step, node counts, loss before the optimizer step, gradient norm,
parameter-update check and peak reserved CUDA memory.

For this qualification, model width=16, heads=4, depth=1 for each
subnetwork, owned-core side=2. These are engineering smoke settings,
not a scientific architecture selection.

## Slurm execution

On the GraphAttention login host **only submit jobs**, never run its
PyTorch computation on a login node:

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
bash -n scripts/slurm/m48_ms_sr_smoke.slurm
sbatch scripts/slurm/m48_ms_sr_smoke.slurm
```

If MS_SR is mounted elsewhere, supply a verified location:

```bash
sbatch --export=ALL,M48_MS_SR_ROOT=/path/to/the/verified/MS_SR \
  scripts/slurm/m48_ms_sr_smoke.slurm
```

If CEDAR cannot be imported from the default GraphAttention Python runtime,
point to a **verified pinned** source directory or compatible Python runtime:

```bash
sbatch --export=ALL,M48_CEDAR_SRC=/verified/path/to/cedar/src \
  scripts/slurm/m48_ms_sr_smoke.slurm
```

`M48_PYTHON` may alternatively name a compatible environment containing both
GraphAttention's required PyTorch dependencies and the pinned `cedar-cfd`.
A CEDAR import alone is insufficient: the launch script verifies the exact
MS_SR source-file hashes before executing tests or loading datasets.

The launcher requires the exact MS_SR files, then runs Python syntax
checks, `tests/unit/test_m48_ms_sr.py` on the allocated node and only
then executes the real CEDAR/data/model pipeline. The tests use local
mock fields for algebra checks and small whole periodic graphs for
topology comparisons; they do **not** replace the real MS_SR operator
test.

Expected output:

```text
/scratch/coop/theret/GraphAttention_runs/m48_ms_sr_smoke/job_<JOB_ID>/
  summary.json
```

Slurm logs:

```text
/scratch/coop/theret/GraphAttention/m48_ms_sr_smoke_<JOB_ID>.out
/scratch/coop/theret/GraphAttention/m48_ms_sr_smoke_<JOB_ID>.err
```

Refuse existing output to avoid overwriting any prior qualification.

## Acceptance gates and next milestones

**Required before scientific training:**

1. All unit tests pass, including full-versus-targeted static/dynamic
   owned outputs **and gradients**, on small reference graphs.
2. MS_SR manifest/role/source/filter/operator IDs verified.
3. Full-domain CEDAR identities and projected source pass numerical
   tolerances, independently for all five conservative components.
4. Both network modes complete a finite, nonzero-gradient optimizer
   step on a real owned core with complete periodic dependencies.
5. Real patch memory and walltime measured and acceptable.
6. Record all source/operator/code versions and experimental settings.

**Not yet implemented in this qualification:**

- Teacher-wide normalization, shuffled/balanced real MS_SR sampling,
  repeated epochs, and training-seed comparisons.
- Global assembled projected Heun time stepping; each FM step will
  require global assembly and global \(I-PC\) projection, not patchwise
  substitution.
- Memory-safe state streaming and fully scale-dependent multi-case
  training for \(80^3\), \(96^3\) or larger real fine fields.
- Physical regime conditioning (Mach, Reynolds, filter width) and
  direct comparability with MS_SR's CNN/CVAE published checkpoints.
- Gradient parity of *global projected multi-patch inference/training*;
  local owned-output network parity alone does not establish this.
- Thermodynamic EOS admissibility or stable AVBP time stepping.
- Statistically reliable inference ensembles and spectral validation.

If all gates pass, next implement M48.4 with teacher-only population
normalization, repeated patch-core training and a frozen validation split.
Keep the locked-test role untouched for final comparison.

This first M48 step is intentionally restricted to **operator + graph +
one-step training correctness**, not claims of a trained CFD generator.
