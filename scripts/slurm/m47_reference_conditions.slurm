#!/bin/bash
#SBATCH --job-name=m47_references
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err

# M47 reference audit; never modifies the six frozen M3.3 YAMLs.
# sbatch scripts/slurm/m47_reference_conditions.slurm
# sbatch --export=ALL,M47_REF_MODE=prepare,M47_PHYSICAL_INPUT=/scratch/coop/theret/HIT_DNS_COMP_physical_conditions.yaml scripts/slurm/m47_reference_conditions.slurm
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
ENV_ROOT=${M47_ENV_ROOT:-/scratch/coop/theret/avbpML_env}
PYTHON=${M47_PYTHON:-$ENV_ROOT/bin/python3}
ROOT=${M47_ARCHIVE_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
REFS=${M47_M3_REFS:-/scratch/coop/theret/HIT_DNS_COMP_case_references}
OUT=${M47_REF_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions}
MODE=${M47_REF_MODE:-inspect}
INPUT=${M47_PHYSICAL_INPUT:-}
case "$MODE" in inspect|prepare) ;; *) echo "M47_REF_MODE must be inspect or prepare" >&2; exit 2 ;; esac
for path in "$REPO" "$IMAGE" "$ROOT" "$REFS"; do
    test -e "$path" || { echo "Missing required path: $path" >&2; exit 2; }
done
test -x "$PYTHON" || { echo "Missing executable: $PYTHON" >&2; exit 2; }
if [ -n "$INPUT" ] && [ ! -s "$INPUT" ]; then
    echo "Requested M47_PHYSICAL_INPUT file is missing or empty: $INPUT" >&2
    exit 2
fi
if [ -f "$ENV_ROOT/bin/activate" ]; then source "$ENV_ROOT/bin/activate"; fi
BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done
cd "$REPO"
echo "M47 references job=$SLURM_JOB_ID host=$(hostname) mode=$MODE git=$(git rev-parse HEAD)"
echo "Archive=$ROOT M3refs=$REFS output=$OUT physical_input=$INPUT"
singularity exec "${BIND[@]}" "$IMAGE" "$PYTHON" -m pytest -q tests/unit/test_m47_physics.py
EXTRA=()
if [ -n "$INPUT" ]; then EXTRA+=(--physical-input "$INPUT"); fi
singularity exec "${BIND[@]}" "$IMAGE" "$PYTHON" -u -m scripts.m47_reference_conditioning \
    "$MODE" --root "$ROOT" --reference-dir "$REFS" --output "$OUT" "${EXTRA[@]}"
echo "M47 reference audit report: $OUT/reference_audit.json"