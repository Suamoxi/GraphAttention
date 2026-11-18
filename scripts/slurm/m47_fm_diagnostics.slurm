#!/bin/bash
#SBATCH --job-name=m47_fm_audit
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.err

# M47.3 diagnostic-only; checkpoints and M47.2 results are never modified.
# Run: sbatch --array=0-2 scripts/slurm/m47_fm_diagnostics.slurm
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
ENV_ROOT=${M47_ENV_ROOT:-/scratch/coop/theret/avbpML_env}
PYTHON=${M47_PYTHON:-$ENV_ROOT/bin/python3}
TRAIN_ROOT=${M47_TRAIN_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_conditioned}
OUT_ROOT=${M47_AUDIT_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_fm_diagnostics}
REF_ROOT=${M47_REF_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions}
REFS=${M47_CONDITION_REFS:-$REF_ROOT/case_references}
DATA_ROOT=${M47_ARCHIVE_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
NVAL=${M47_AUDIT_VALIDATION_PER_CASE:-1}
STEPS=${M47_AUDIT_STEPS:-50}
TRAJECTORY=${M47_AUDIT_TRAJECTORY:-YES}
case "$TRAJECTORY" in YES|NO) ;; *) echo "M47_AUDIT_TRAJECTORY must be YES or NO" >&2; exit 2 ;; esac
if [ -n "${M47_CONDITION:-}" ]; then
    echo "Unset M47_CONDITION: array index fixes the condition" >&2; exit 2
fi
case "${SLURM_ARRAY_TASK_ID:-missing}" in
    0) CONDITION=none ;;
    1) CONDITION=mach ;;
    2) CONDITION=mach_re ;;
    *) echo "Submit with --array=0-2" >&2; exit 2 ;;
esac
for path in "$REPO" "$IMAGE" "$PYTHON" "$REFS" "$DATA_ROOT" \
            "$TRAIN_ROOT/$CONDITION/train_s42/best.pt" \
            "$TRAIN_ROOT/$CONDITION/train_s42/standardizers.pt"; do
    test -e "$path" || { echo "Missing: $path" >&2; exit 2; }
done
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty code checkout" >&2; exit 2; }
TAG="v1_e2_val${NVAL}_h${STEPS}_seed20261116"
OUTPUT="$OUT_ROOT/$CONDITION/$TAG"
test ! -e "$OUTPUT" || { echo "Output already exists: $OUTPUT" >&2; exit 2; }
if [ -f "$ENV_ROOT/bin/activate" ]; then source "$ENV_ROOT/bin/activate"; fi
BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done
echo "M47.3 mode=$CONDITION output=$OUTPUT git=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m pytest -q \
    tests/unit/test_m47_fm_diagnostics.py tests/unit/test_m47_generation.py
EXTRA=()
if [ "$TRAJECTORY" = NO ]; then EXTRA+=(--no-trajectory); fi
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -u -m scripts.m47_fm_diagnostics \
    --training-root "$TRAIN_ROOT" --reference-dir "$REFS" \
    --archive-root "$DATA_ROOT" --condition "$CONDITION" \
    --output-dir "$OUTPUT" --validation-per-case "$NVAL" --steps "$STEPS" \
    --seed 20261116 "${EXTRA[@]}"
echo "M47.3 COMPLETE: $OUTPUT/summary.json"
