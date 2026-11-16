#!/bin/bash
#SBATCH --job-name=m47_gen_ens
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.err

# Default is a 2-Heun-step, one-sample, one-case SMOKE, not a long run.
# After all three smoke tasks pass:
# sbatch --array=0-2 --export=ALL,M47_GEN_MODE=full,M47_GEN_CONFIRM=YES scripts/slurm/m47_generation_test.slurm
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
ENV_ROOT=${M47_ENV_ROOT:-/scratch/coop/theret/avbpML_env}
PYTHON=${M47_PYTHON:-$ENV_ROOT/bin/python3}
TRAIN_ROOT=${M47_TRAIN_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_conditioned}
OUT_ROOT=${M47_GEN_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_generated}
REF_ROOT=${M47_REF_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions}
REFS=${M47_CONDITION_REFS:-$REF_ROOT/case_references}
DATA_ROOT=${M47_ARCHIVE_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
MODE=${M47_GEN_MODE:-smoke}
CONFIRM=${M47_GEN_CONFIRM:-NO}
STEPS=${M47_GEN_STEPS:-50}
SAMPLES=${M47_GEN_SAMPLES:-8}
SEED=${M47_GEN_SEED:-20261116}
FIELDS=${M47_GEN_STORE_FIELDS:-NO}
case "$MODE" in smoke|full) ;; *) echo 'M47_GEN_MODE must be smoke or full' >&2; exit 2 ;; esac
case "$FIELDS" in NO|YES) ;; *) echo 'M47_GEN_STORE_FIELDS must be NO or YES' >&2; exit 2 ;; esac
if [ "$MODE" = full ] && [ "$CONFIRM" != YES ]; then
    echo 'Full ensemble requires M47_GEN_CONFIRM=YES' >&2; exit 2
fi
if [ -n "${M47_CONDITION:-}" ]; then
    echo 'Unset M47_CONDITION: Slurm array index determines condition' >&2; exit 2
fi
case "${SLURM_ARRAY_TASK_ID:-missing}" in
    0) CONDITION=none ;;
    1) CONDITION=mach ;;
    2) CONDITION=mach_re ;;
    *) echo 'Submit as --array=0-2' >&2; exit 2 ;;
esac
for path in "$REPO" "$IMAGE" "$REFS" "$DATA_ROOT" \
            "$TRAIN_ROOT/comparison_s42.json" \
            "$TRAIN_ROOT/$CONDITION/train_s42/best.pt"; do
    test -e "$path" || { echo "Missing: $path" >&2; exit 2; }
done
test -x "$PYTHON" || { echo "Missing Python: $PYTHON" >&2; exit 2; }
cd "$REPO"
test -z "$(git status --porcelain)" || { echo 'Dirty code checkout' >&2; exit 2; }
if [ "$MODE" = smoke ]; then
    TAG="smoke_${SLURM_JOB_ID}"
    RUN_ARGS=(--steps 2 --samples-per-case 1 --case 'Re54/Ma0.2')
else
    TAG="ensemble_s42_h${STEPS}_n${SAMPLES}_seed${SEED}"
    RUN_ARGS=(--steps "$STEPS" --samples-per-case "$SAMPLES")
fi
OUTPUT="$OUT_ROOT/$CONDITION/$TAG"
test ! -e "$OUTPUT" || { echo "Output already exists: $OUTPUT" >&2; exit 2; }
mkdir -p "$OUT_ROOT"
if [ -f "$ENV_ROOT/bin/activate" ]; then source "$ENV_ROOT/bin/activate"; fi
BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done
echo "M47.2 mode=$MODE condition=$CONDITION output=$OUTPUT git=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m pytest -q \
    tests/unit/test_m47_generation.py
if [ "$FIELDS" = YES ]; then RUN_ARGS+=(--store-fields); fi
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -u -m scripts.m47_generate_ensembles \
    --training-root "$TRAIN_ROOT" --reference-dir "$REFS" \
    --archive-root "$DATA_ROOT" \
    --condition "$CONDITION" --output-dir "$OUTPUT" \
    --sampling-seed "$SEED" "${RUN_ARGS[@]}"
echo "M47.2 completed: $OUTPUT/summary.json"
