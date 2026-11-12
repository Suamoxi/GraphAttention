#!/bin/bash
#SBATCH --job-name=m46_analysis
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err

# M46 all-snapshot analysis. Requires completed inventory and explicit M3.3
# per-case YAML references. CPU FFT is memory-intensive; no GPU calculations.
#   sbatch scripts/slurm/m46_hit_analyze.slurm
# Low-memory first pass (distinct output and automatic manifest copy):
#   sbatch --export=ALL,M46_NO_SPECTRA=YES scripts/slurm/m46_hit_analyze.slurm
set -euo pipefail

REPO=${M46_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M46_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
ENV_ROOT=${M46_ENV_ROOT:-/scratch/coop/theret/avbpML_env}
PYTHON="${M46_PYTHON:-$ENV_ROOT/bin/python3}"
REF_DIR=${M46_REFERENCE_DIR:-/scratch/coop/theret/HIT_DNS_COMP_case_references}
INVENTORY_OUT=${M46_OUTPUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m46_hit_similarity}
NO_SPECTRA=${M46_NO_SPECTRA:-NO}
DRAWS=${M46_DRAWS_PER_SNAPSHOT:-2048}
SEED=${M46_SEED:-46}

case "$NO_SPECTRA" in
    YES) DEFAULT_OUT="${INVENTORY_OUT}_qualified_no_spectra"; EXTRA=(--no-spectra) ;;
    NO)  DEFAULT_OUT="${INVENTORY_OUT}_qualified"; EXTRA=() ;;
    *)   echo "M46_NO_SPECTRA must be YES or NO" >&2; exit 2 ;;
esac
OUT=${M46_ANALYSIS_OUTPUT:-$DEFAULT_OUT}
QUALIFIED=${M46_SELECTION_MANIFEST:-$INVENTORY_OUT/qualified_manifest.json}
POLICY=${M46_SELECTION_POLICY:-$REPO/configs/m46_stationary_selection_reviewed_legacy.json}

for path in "$REPO" "$IMAGE" "$REF_DIR" "$INVENTORY_OUT/inventory.json" "$QUALIFIED" "$POLICY"; do
    test -e "$path" || {
        echo "Missing required path: $path" >&2
        echo "Run the M46 inventory Slurm job and provide the case reference YAMLs first." >&2
        exit 2
    }
done
test -x "$PYTHON" || { echo "Missing environment Python: $PYTHON" >&2; exit 2; }

if [ -f "$ENV_ROOT/bin/activate" ]; then
    source "$ENV_ROOT/bin/activate"
fi
cd "$REPO"
mkdir -p "$OUT"
# Qualification and full source inventory must be copied together.
if [ "$OUT" != "$INVENTORY_OUT" ]; then
    if [ -e "$OUT/inventory.json" ]; then
        cmp -s "$INVENTORY_OUT/inventory.json" "$OUT/inventory.json" || {
            echo "Inventory mismatch; refusing to mix data selections: $OUT" >&2
            exit 2
        }
    else
        cp "$INVENTORY_OUT/inventory.json" "$OUT/inventory.json"
    fi
fi
if [ -e "$OUT/qualified_manifest.json" ]; then
    cmp -s "$QUALIFIED" "$OUT/qualified_manifest.json" || {
        echo "Qualification manifest changed; refusing to mix selected runs: $OUT" >&2
        exit 2
    }
else
    cp "$QUALIFIED" "$OUT/qualified_manifest.json"
fi
printf '%s\n' "$(git rev-parse HEAD)" > "$OUT/analysis_git_commit.txt"
git status --short > "$OUT/analysis_git_status.txt"
printf 'M46 analysis job=%s host=%s\n' "$SLURM_JOB_ID" "$(hostname)"
printf 'references=%s inventory=%s output=%s no_spectra=%s\n'     "$REF_DIR" "$INVENTORY_OUT/inventory.json" "$OUT" "$NO_SPECTRA"

BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done
singularity exec "${BIND[@]}" "$IMAGE" "$PYTHON" -m pytest -q     tests/unit/test_m46_hit_similarity.py

# Per-snapshot caches permit resubmitting the same job after a time limit.
singularity exec "${BIND[@]}" "$IMAGE" "$PYTHON" -u -m scripts.m46_hit_similarity analyze     --output "$OUT"     --reference-dir "$REF_DIR"     --draws-per-snapshot "$DRAWS"     --seed "$SEED"     --selection-manifest "$OUT/qualified_manifest.json"     --selection-policy "$POLICY"     "${EXTRA[@]}"

test -s "$OUT/analysis_summary.json"
echo "M46 analysis complete: $OUT/analysis_summary.json"
