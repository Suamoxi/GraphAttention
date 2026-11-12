# M46 — Stationarity-qualified cross-case HIT similarity audit

## Scope and evidence status

This is a **read-only analysis implementation**, not a model/conditioning change.
**Policy update:** M46 retains every path in `inventory.json` for provenance,
but statistical analyses operate **only on explicitly reviewed stationary-candidate
snapshots**. The former all-snapshot analytical policy is superseded.

- Source inventory: six Re54/Re94 cases from `/archive2/cfd-phlusim/HIT_DNS_COMP` and all ReT060/ReT100 cases found in that archive and in `/scratch/ai/HIT_DNS_pilot/data/`.
- All discovered snapshots remain in the full inventory, with no data deletion. A separate reviewed selection manifest determines which snapshots enter the primary statistical comparison. The full inventory is **not** an analysis dataset.
- **Within a source**, distinct HDF5 files with the same numeric suffix are **both retained**, with all such counter collisions listed in `within_source_counter_collisions`. Unnumbered files such as `last_solution.h5` are also retained if they are distinct filesystem objects, with a null numeric counter and an entry in `unnumbered_snapshots`. Identical filesystem objects reached via two filenames (symlinks/hardlinks, same inode) are counted once, and their aliases are recorded.
- **Between archive and pilot**, a matching `(case_id, numeric snapshot counter)` is selected from the archive, even if the zero-padding differs. The suppressed pilot paths are recorded in `duplicates`. Unnumbered paths cannot be matched across independent sources using time counters, so both remain until provenance establishes identity. This is a provenance convention, **not evidence of byte-identical arrays**. No scientific or stationarity filter is applied.
- Physical reference scales come exclusively from explicit M3.3 case YAML files. The tool **does not** infer velocity, density or Mach from `ReT...`, `Mt...`, `Xi...`, grid resolution, or a posteriori statistics.
- `inventory` works with no case YAML. `select` requires only the inventory and the reviewed window policy. `analyze` requires case reference YAMLs **only for cases selected by the reviewed policy**; all unreviewed cases remain pending, not silently accepted. ReT case setup references and stationary intervals remain to be qualified.
- The primary results are exploratory stationary-window distributions. Reviewed M43 cutoffs do not prove exact statistical stationarity or temporal independence.

## Usage

From the repository root, submit the **inventory** as a Slurm job (never run
Python on the login node). The wrapper uses the proven project Singularity
image and explicitly selects `avbpML_env/bin/python3`:

```bash
sbatch scripts/slurm/m46_hit_inventory.slurm
```

The inventory launcher selects the first compute-node-visible archive
containing Re54, Re94, ReT060 and ReT100, trying
`/archivenfs/phlusim/HIT_DNS_COMP` (used by M43 on Grace), then
`/archive2/cfd-phlusim/HIT_DNS_COMP`. It does not silently accept a
legacy-only archive. The pilot root defaults to
`/scratch/ai/HIT_DNS_pilot`; selected paths are printed in Slurm logs.
The inventory writes to
`/scratch/coop/theret/GraphAttention_runs/m46_hit_similarity`. Override
paths via `M46_ARCHIVE_ROOT`, `M46_PILOT_ROOT`, and `M46_OUTPUT_ROOT`
at submission time using `sbatch --export=ALL,...`.

Inspect `inventory.json`: expected case count is 32 when the six legacy and 26 pilot cases are present. The snapshot count is computed from actual filenames, not hard-coded.

Supply one authoritative case definition per **qualified analysis case** in the specified reference directory, named `Re54_Ma0.05.yaml`, `ReT060_Mt0.59_Xi0.3.yaml`, etc. Each file must be a valid M3.3 `CaseDefinition`, have a matching `case_id`, use the same reference scheme, and declare inference-available `rho_ref`, `U_ref`, and `L_ref`, with the declared `L_ref` equaling the physical cubic box side. A named `a_ref` is optional and required only for a reference-sound-speed turbulent Mach diagnostic. Do **not** use a posteriori `Mt` labels as inference conditioning metadata.

Submit the **analysis** through Slurm after the inventory succeeds and
the complete per-case reference YAML directory exists:

```bash
sbatch scripts/slurm/m46_hit_analyze.slurm
```

For the memory-lighter, **no-spectra** pass, use:

```bash
sbatch --export=ALL,M46_NO_SPECTRA=YES scripts/slurm/m46_hit_analyze.slurm
```

This places results in `m46_hit_similarity_no_spectra` and copies the
inventory manifest from the original output directory. A subsequent
full-spectra run writes into the original `m46_hit_similarity` directory.
Use `M46_REFERENCE_DIR` to override the default reference directory and
`M46_ANALYSIS_OUTPUT` to override the analysis output; each job retains
per-snapshot caches to support repeat submissions. Both wrappers execute
tests and the analysis *inside* the allocated Slurm job, not on login nodes.
The requested `grace` GPU is a cluster-allocation convention copied from
existing job scripts; no GPU kernels are used in this analysis.

Every **selected** snapshot receives exact full-grid field moments and turbulent kinetic energy. By default selected snapshots also receive an FFT spectrum and solenoidal/dilatational energy split. FFTs at `320^3` unique points are memory- and CPU-intensive: use a high-memory CPU node, not the login node. For the lower-memory first execution of qualified field statistics and pairwise distribution comparisons, use `M46_NO_SPECTRA=YES` in the analysis Slurm submission. Results are resumable per snapshot; changing analysis options or reference-file contents invalidates existing caches explicitly.

## Mathematical definitions

For each case the conservative nodal fields are nondimensionalized using the existing `ConvectiveNondimensionalizer` (M3.3):

\[
\rho^*=\rho/\rho_{ref},\quad (\rho u_i)^*=(\rho u_i)/(\rho_{ref}U_{ref}),\quad (\rho E)^*=(\rho E)/(\rho_{ref}U_{ref}^2).
\]

All statistics and spectra exclude the maximum coordinate plane on each axis of the endpoint-inclusive periodic mesh. The Cartesian coordinate mapping is checked across all node IDs and is independent of exporter ordering. Node means are arithmetic means on this uniform grid; no arbitrary resolution weighting is applied.

Per-snapshot (non-density-weighted) velocity fluctuations are calculated from `u_i = (rho u_i)/rho`:

\[
\overline{\mathbf u^*}=\langle\mathbf u/U_{ref}\rangle_V,
\qquad k^*=\frac12\left\langle|\mathbf u^*-\overline{\mathbf u^*}|^2\right\rangle_V.
\]

The optional turbulent Mach diagnostic uses `sqrt(2*k*)*U_ref/a_ref` **only** if an explicit inference-available `a_ref` scale exists; no ideal-gas sound speed is silently inferred.

Spectra use normalized FFT coefficients on the unique periodic grid. The full-cube, mean-excluded Helmholtz projection is

\[
\widehat{\mathbf u}_d(\mathbf k)=\frac{\mathbf k\, (\mathbf k\cdot\widehat{\mathbf u})}{|\mathbf k|^2},
\qquad \widehat{\mathbf u}_s=\widehat{\mathbf u}-\widehat{\mathbf u}_d.
\]

`dilatational_fraction` is the ratio of full-cube Fourier dilatational energy to total fluctuation energy. The one-dimensional shell-integrated spectrum groups modes at nearest integer `|k|L_box/(2*pi)` up to the *inscribed* Nyquist sphere; its x coordinate is `kL_box = 2*pi*m`, and `E_star` is normalized so `sum(E_star * 2*pi)` equals energy contained in those shells. `spectrum_energy_capture` reports the fraction of full-cube kinetic energy inside those spectral shells. Therefore, the plotted spectrum must not be assumed to integrate to the total TKE at high wavenumber.

Full-grid scalar moments (mean, population standard deviation, skew, excess kurtosis) are computed in bounded HDF5 chunks. For approximate distribution comparison, each snapshot contributes the same number of uniformly selected **Cartesian** interior nodes (default 2048), independent of native node IDs. Per-case samples are pooled over all its snapshots, and each case contributes one distribution to each comparison. Reported Wasserstein-1 is the average absolute difference of 1024 equal-probability quantiles; it is **an estimator**, not exact full-field W1. The normalized variant divides by the pooled standard deviation of the compared cases' full-field distributions. This is distinct from M29's training-standardized Wasserstein definition.

## Outputs and reproducibility

- `inventory.json`: source paths, mesh dimensions, every selected snapshot, same-inode aliases, archive/pilot overlap decisions, and retained within-source counter collisions.
- `per_snapshot/<case>/<snapshot>.npz`: cached exact field moments, TKE, optional spectra, sampled distribution values, reference checksum and source file metadata.
- `case_field_statistics.csv`, `case_field_pdfs.csv`, `case_pairwise_wasserstein.csv`, `case_regime_parameters.csv`.
- `snapshot_spectra.csv`: per-snapshot nondimensional kinetic-energy spectra when FFTs are enabled.
- `analysis_summary.json`: reference convention, weighting and estimator descriptions.

The outputs compare qualified empirical regimes and help select explicit conditioning. They do not modify M43-B training code or any generator, and they do not establish causal effects of `Xi` when controller settings differ. Forcing families must remain distinct in subsequent interpretation: the bundled legacy selection compares only `random_forcing` cases. ReT `lowk_split_forcing` cases enter only after explicit review of their stationary windows.

## Tests and validation status

`tests/unit/test_m46_hit_similarity.py` covers streaming moment algebra, Parseval and Helmholtz limits for analytic sine modes, Cartesian coordinate reordering, one synthetic case with M3.3 references, and archive-over-pilot snapshot deduplication. Synthetic tests validate the implementation; they do not substitute for a Calypso full-volume performance/physics run. In particular, 320^3 CPU FFT peak RSS, archived-vs-pilot numerical identity, real reference completeness and actual AVBP results remain **target-unvalidated**.

Thermodynamic temperature and pressure diagnostics are deferred: the common pilot inventory contains the five conservative fields and no authoritative equation-of-state reconstruction is introduced here.
### M46 repeated-timestep filename handling (job 503080)

The legacy M43 `_find_snapshots` helper intentionally fails when a case's
files contain repeated numeric timestep counters. The M46 inventory must
retain both distinct snapshots rather than suppress a potentially valid
realization. It now uses its own numeric-filename discovery helper, retaining
same-source numeric suffix collisions explicitly in the JSON manifest. Identical
filesystem aliases are detected by device+inode and listed separately. Across
archive and pilot, the existing archive-first counter policy remains in
effect and cannot establish physical equality without independent checks.

This fixes the exception observed on Grace for
`ReT060/Mt0.23_Xi0.0`; that case's specific colliding filenames
and their scientific equivalence remain unknown until the new inventory
reports them. The six-case M43 dataset reader has not been modified.

### M46 nonnumeric solution filenames (job 503136)

The scan of `ReT060/Mt0.23_Xi0.0` encountered
`last_solution.h5`, which has no parseable timestep suffix. M46 retains
unnumbered source files, recording `counter: null` and
`unnumbered_snapshots` in the inventory rather than excluding them as
transients or failing filename parsing. The metadata-only inventory still
validates HDF5 schemas, and physically identical symlink/hardlink aliases
are counted once using device/inode identity. Distinct unnumbered files
remain eligible with **unknown physical time**; consequently, time-series
ordering and cross-source pairing for those files cannot be inferred.
The strict M43 reader remains unchanged.

### M46 archive/pilot filenames without counters

Unnumbered files in different sources cannot be paired using timestep counters.
If archive and pilot both contain a distinct `last_solution.h5`, the pilot
sample receives a source-qualified ID (e.g.,
`ReT060/Mt0.59_Xi0.3:reT_pilot__last_solution`) and both are retained.
An unnumbered symlink to a numbered pilot snapshot that was already rejected
by archive-first precedence is also rejected, using the pilot inode registry.
Without independent provenance, distinct unnumbered files can still contain
byte-identical physical states; this implementation does not claim otherwise.


## Reviewed snapshot eligibility — revised M46 policy

The input inventory remains immutable and records all 32 cases and 1,360
discovered snapshots. It does **not** authorize using every snapshot in
physical statistics.

The bundled policy `configs/m46_stationary_selection_reviewed_legacy.json`
encodes the previously reviewed M43 exploratory decision for each of the
six Re54/Re94 cases: snapshots 1–10 are transient, 11–40 are
transition/uncertain, and **41–100 are selected**. Hence 360 legacy
snapshots enter the primary comparison. The remaining 240 legacy
snapshots are retained in the provenance file but excluded from statistical
aggregates. The boundary is an exploratory reviewed cutoff, not a proof of
stationarity.

**No corresponding reviewed stationary selection is available for the
26 ReT060/ReT100 cases**. Their 760 snapshots remain
`pending_case_qualification`; the 4–8-snapshot pilot sequences must not be
treated as stationary merely because they are available. Do not copy
the legacy snapshot-number cutoff onto the ReT sequences.

Run selection through a Slurm job, after the completed inventory:

```bash
sbatch scripts/slurm/m46_hit_select.slurm
```

This writes `qualified_manifest.json`,
`snapshot_eligibility.csv` (one row per discovered snapshot),
and `eligibility_summary.json` alongside the complete `inventory.json`.
Selected windows require complete, unique canonical `solut_hit_NNNNNNNN.h5`
counters by default; crash, restart or unnumbered files are separately
reported and never automatically added to stationary windows.

The analysis launcher now refuses to operate until the qualified manifest
exists, verifies it against the complete inventory and selection-policy
checksums, recomputes the reviewed case rule on the full inventory to reject
manually altered selections, and loads case references only for selected cases.
Separate output directories are
`m46_hit_similarity_qualified_no_spectra` (recommended first pass)
and `m46_hit_similarity_qualified` (FFT pass). Existing
all-snapshot caches must **not** be relabeled as qualified analyses.

To qualify ReT cases later, extend a copied policy with defensible
case-specific counter intervals and evidence provenance, then submit
selection using `M46_SELECTION_POLICY`. Do not modify the M43
reference files or training dataset in place.
