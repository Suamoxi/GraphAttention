# M48.4d - independent AVBP -> CEDAR -> GraphAttention ordering qualification

Status: **implementation and synthetic tests passed locally; real Calypso
source audit pending**. No claim about original AVBP ordering until an
`ordering_audit.json` receipt reports `status=PASS` on both native grids.

## Why this is needed

M48 periodic DiNAT assumes global Cartesian IDs
`ID(i,j,k)=(i*N+j)*N+k`, with periodic offsets modulo `N`. The preceding
synthetic M48 tests compare graph paths using this same index convention;
they cannot detect a consistently wrong source-to-Cartesian permutation.

MS_SR **does** call the pinned `PeriodicHexLattice` class to reorder native
AVBP node IDs before publishing `qf`. The independent qualification must
verify that these representatives correspond to physical `(x,y,z)` positions
and to the original hexahedral connectivity - rather than assume the
CEDAR mapping is right.

## Inputs and read-only scope

- Original staged AVBP pilot: `/scratch/ai/HIT_DNS_pilot`, including
  `data/<case>/MESH/mesh.mesh.h5`, original
  `GaseousPhase/{rho,rhou,rhov,rhow,rhoE}` HDF5 snapshots, and verified
  source-role assignment.
- MS_SR published manifest: `/scratch/ai/MS_SR/dataset/manifest.json`.
- The original MS_SR-pinned `cedar-cfd` implementation; neither a new filter
  nor a replacement spatial reordering is introduced.
- One actual **teacher** `w1p25_l1` `q8 -> q4` record from each of
  `ReT060` and `ReT100`, determined by the sealed manifest, not guessed.
- Full original unique native grids are `192^3` and `320^3`;
  `q4/q8` training pairs are `48^3/24^3` and `80^3/40^3`.

The audit reads complete original source fields and complete cell
connectivity. It never writes to pilot or MS_SR locations, never uses
validation/locked-test roles, and never modifies CEDAR or GraphAttention
training data. Source, mesh and published HDF5 SHA-256 hashes are checked
again by default. Removing this check via `--skip-file-hashes` is an
explicit weaker diagnostic, not an accepted qualification.

## Independent mathematical checks

1. Derive `(i,j,k)` **solely from AVBP `Coordinates/x,y,z`** and published
   *native* physical grid spacing. The original native mesh includes the
   duplicated upper periodic endpoint, so coordinate level `N` maps to `0`.
   Reject points that do not lie on the axis-aligned `0..N` lattice to within
   the declared `1e-3` cell-width tolerance.
2. Compare **every** original AVBP node's coordinate-derived periodic
   ID against the original CEDAR `source_to_unique`, and verify that
   `representative_ids[k]` really corresponds to Cartesian ID `k`.
   No CEDAR map is used to construct the expected indices.
3. Read **all** original `Connectivity/hex->node` cells, including periodic
   face/edge/corner seams. Verify exactly one eight-corner hexahedron at
   each `N^3` periodic Cartesian cell, with no missing or duplicated cells.
   Verify seam counts `N^2` per face axis, `N` for each pair of axes,
   and one three-axis corner seam cell. No local AVBP vertex ordering
   assumption is made.
4. For each of the five float64 conservative fields, compare **every**
   native published `qf[i,j,k]` exactly against the corresponding original
   `GaseousPhase/*` value selected by `representative_ids[ID(i,j,k)]`.
   Check all original periodic endpoint aliases against their representatives
   with the source's `1e-10` relative tolerance.
5. Apply the original pinned CEDAR staged restriction on the entire native
   `qf` component and compare the rebuilt `q2`, `q4`, `q8` values with the
   published filter hierarchy. The tolerance for full-field verification is
   `1e-10 * max(1, max(abs(stored)))`, matching the producer's conservative
   component scale convention. No patchwise periodic extension is used.
6. Load the selected `q8 -> q4` pair through the **actual M48**
   `load_ms_sr_pair` and `flat_channels`, and require bitwise equality at
   every `q4` and `q8` node/component against the published MS_SR arrays.
   Recheck the M48 pair's global CEDAR projection identities.

A permutation of source node IDs is acceptable if and only if the
coordinate-derived order and published arrays remain correct. A transposition
of x and z, a mistaken representative map, a corrupt physical hexahedron,
or missing periodic seam fails the audit.

## Qualification and launcher

The new module is `scripts/m48_4d_ordering_audit.py`; synthetic tests are
`tests/unit/test_m48_4d_ordering_audit.py`. The launcher is
`scripts/slurm/m48_4d_ordering_audit.slurm`.

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
bash -n scripts/slurm/m48_4d_ordering_audit.slurm
sbatch scripts/slurm/m48_4d_ordering_audit.slurm
```

Use environment variables `M48_PILOT_ROOT`, `M48_MS_SR_ROOT`,
`M48_CEDAR_SRC` or `M48_PYTHON` only if the corresponding original
read-only sources are mounted elsewhere. Paths inside Singularity are
`/scratch/...` (not host-only `/gpfs-calypso/scratch/...`). The launcher
uses Grace **without a GPU allocation** and requests four CPUs and 32 GB
host memory. The full native `320^3` comparison is CPU/memory intensive.

Expected outputs:

```
/scratch/coop/theret/GraphAttention_runs/m48_4d_ordering_audit/job_<JOBID>/
    ordering_audit.json
    ordering_audit_ReT060.json
    ordering_audit_ReT100.json
```

A case-specific receipt is written only after that case passed. The summary
is marked `FAIL` with the diagnostic if an exception occurs. A successful
receipt proves actual physical source-coordinate ordering, original
periodic mesh connectivity, full field ordering, filter hierarchy and
GraphAttention loader layout for the two selected teacher examples. It
**does not** prove the ordering of every one of the 44 source files if
those use independent mesh files; the same audit can be extended to all
cases as a subsequent gate. No additional training is launched.

## Scope and assumptions

- The current pilot is an axis-aligned uniform Cartesian box with duplicate
  periodic endpoints. The audit intentionally fails on rotated, unstructured
  or nonperiodic meshes. Their geometry requires a different index contract.
- Five conservative components, CEDAR `w1p25_l1` and q8/q4 are fixed to
  the current M48 training selection.
- AVBP alias tolerance is `1e-10` relative L2, as in MS_SR.
- Training's local periodic neighbor and cross-scale graph definitions are
  unaffected; this checks that their *inputs* use the assumed ordering.
- Any failed audit must block expansion to long mixed-resolution training
  until the mapping or dataset is corrected and independently qualified.
