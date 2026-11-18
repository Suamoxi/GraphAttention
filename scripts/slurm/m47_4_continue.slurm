#!/bin/bash
#SBATCH --job-name=m47_4_train
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.err

# Stage 5 first, stage 10 after all stage-5 jobs pass.
# sbatch --array=0-2 --export=ALL,M47_4_STAGE=5 scripts/slurm/m47_4_continue.slurm
# sbatch --array=0-2 --export=ALL,M47_4_STAGE=10 scripts/slurm/m47_4_continue.slurm
# Execute the unmodified original M47.1 training revision.
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
ENV_ROOT=${M47_ENV_ROOT:-/scratch/coop/theret/avbpML_env}
PYTHON=${M47_PYTHON:-$ENV_ROOT/bin/python3}
HOST_PYTHON=${M47_HOST_PYTHON:-python3}
SOURCE_ROOT=${M47_TRAIN_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_conditioned}
OUT_ROOT=${M47_4_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_4_longer_training}
CODE_ROOT=${M47_4_CODE_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_4_pinned_code}
DATA_ROOT=${M47_ARCHIVE_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
REF_ROOT=${M47_REF_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions}
REFS=${M47_CONDITION_REFS:-$REF_ROOT/case_references}
REF_AUDIT=${M47_REFERENCE_AUDIT:-$REF_ROOT/reference_audit.json}
M46_ROOT=${M47_M46_ROOT:-/scratch/coop/theret/GraphAttention_runs/m46_hit_similarity}
QUALIFIED=${M47_QUALIFIED_MANIFEST:-$M46_ROOT/qualified_manifest.json}
INVENTORY=${M47_INVENTORY:-$M46_ROOT/inventory.json}
STAGE=${M47_4_STAGE:-}
PINNED_REV=5d472796828a8abeb51747751b777dc8cc837f35

if [ -n "${M47_CONDITION:-}" ]; then
    echo "Unset M47_CONDITION for the fixed three-arm Slurm array" >&2; exit 2
fi
case "${SLURM_ARRAY_TASK_ID:-missing}" in
    0) CONDITION=none ;;
    1) CONDITION=mach ;;
    2) CONDITION=mach_re ;;
    *) echo "M47.4 requires --array=0-2" >&2; exit 2 ;;
esac
case "$STAGE" in
    5|10) ;;
    *) echo "Set M47_4_STAGE=5 or 10" >&2; exit 2 ;;
esac
for path in "$REPO" "$IMAGE" "$PYTHON" "$DATA_ROOT" "$REFS" "$REF_AUDIT" \
            "$QUALIFIED" "$INVENTORY" \
            "$SOURCE_ROOT/$CONDITION/train_s42/last.pt"; do
    test -e "$path" || { echo "Missing: $path" >&2; exit 2; }
done
test -x "$PYTHON" || { echo "Python is not executable: $PYTHON" >&2; exit 2; }
command -v "$HOST_PYTHON" >/dev/null
cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
    echo "Refusing dirty main checkout" >&2; exit 2
fi
git cat-file -e "$PINNED_REV^{commit}" || {
    echo "M47.1 original training commit is missing from local history" >&2; exit 2;
}
SOURCE="$SOURCE_ROOT/$CONDITION/train_s42"
WORKSPACE="$OUT_ROOT/$CONDITION"
TARGET="$WORKSPACE/train_s42"
for case_name in Re54_Ma0.05 Re54_Ma0.2 Re54_Ma0.3 Re94_Ma0.05 Re94_Ma0.2 Re94_Ma0.3; do
    test -s "$REFS/$case_name.yaml" || { echo "Missing reference: $case_name" >&2; exit 2; }
done
BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done
if [ "$STAGE" = 5 ]; then
    "$HOST_PYTHON" -m scripts.m47_4_stage prepare \
        --source "$SOURCE" --workspace "$WORKSPACE" --mode "$CONDITION"
else
    "$HOST_PYTHON" -m scripts.m47_4_stage gate10 \
        --source "$SOURCE" --workspace "$WORKSPACE" --mode "$CONDITION"
fi
mkdir -p "$CODE_ROOT"
PINNED_DIR=$(mktemp -d "$CODE_ROOT/${CONDITION}_e${STAGE}_${SLURM_JOB_ID}_XXXXXX")
cleanup() { rm -rf -- "$PINNED_DIR"; }
trap cleanup EXIT
git clone --quiet --no-hardlinks "$REPO" "$PINNED_DIR"
git -C "$PINNED_DIR" checkout --quiet --detach "$PINNED_REV"
test "$(git -C "$PINNED_DIR" rev-parse HEAD)" = "$PINNED_REV"
test -z "$(git -C "$PINNED_DIR" status --porcelain)"
echo "M47.4 pinned revision=$PINNED_REV current_main=$(git -C "$REPO" rev-parse HEAD)"
echo "stage=$STAGE condition=$CONDITION source=$SOURCE continuation=$TARGET"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# The original runner reconstructs its frozen signature, then restores last.pt,
# optimizer, and the Flow Matching training generator state.
cd "$PINNED_DIR"
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m pytest -q \
    tests/unit/test_m47_physics.py tests/unit/test_m43_b_multires.py
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -u -m scripts.train_m43_b_multires \
    --mode train --root "$DATA_ROOT" --reference-dir "$REFS" \
    --qualified-manifest "$QUALIFIED" --source-inventory "$INVENTORY" \
    --selection-policy "$PINNED_DIR/configs/m46_stationary_selection_reviewed_legacy.json" \
    --reference-audit "$REF_AUDIT" --conditioning-mode "$CONDITION" \
    --run-dir "$TARGET" --seed 42 --temporal-embargo 2 \
    --accumulation 6 --lr 0.0001 --max-epochs "$STAGE" --resume
cd "$REPO"
"$HOST_PYTHON" -m scripts.m47_4_stage freeze \
    --source "$SOURCE" --workspace "$WORKSPACE" --mode "$CONDITION" \
    --epochs "$STAGE"
echo "M47.4 stage $STAGE completed: $WORKSPACE/epoch$(printf '%02d' "$STAGE")"
