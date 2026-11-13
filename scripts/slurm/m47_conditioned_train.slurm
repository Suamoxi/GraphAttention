#!/bin/bash
#SBATCH --job-name=m47_cond_train
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err

# DEFAULT: one six-case smoke optimizer update, never a long training run.
# Three-arm paired campaign: sbatch --array=0-2 scripts/slurm/m47_conditioned_train.slurm
# For long training, explicitly set M47_MODE=train and M47_CONFIRM_LONG_TRAIN=YES.
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
ENV_ROOT=${M47_ENV_ROOT:-/scratch/coop/theret/avbpML_env}
PYTHON=${M47_PYTHON:-$ENV_ROOT/bin/python3}
ROOT=${M47_ARCHIVE_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
REF_ROOT=${M47_REF_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions}
REFS=${M47_CONDITION_REFS:-$REF_ROOT/case_references}
REF_AUDIT=${M47_REFERENCE_AUDIT:-$REF_ROOT/reference_audit.json}
M46_ROOT=${M47_M46_ROOT:-/scratch/coop/theret/GraphAttention_runs/m46_hit_similarity}
QUALIFIED=${M47_QUALIFIED_MANIFEST:-$M46_ROOT/qualified_manifest.json}
INVENTORY=${M47_INVENTORY:-$M46_ROOT/inventory.json}
POLICY=${M47_STATIONARITY_POLICY:-$REPO/configs/m46_stationary_selection_reviewed_legacy.json}
MODE=${M47_MODE:-smoke}
# In a Slurm array, task ID fixes the conditioning arm. Never let a stale
# exported M47_CONDITION silently produce three copies of the same model.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    if [ -n "${M47_CONDITION:-}" ]; then
        echo "Unset M47_CONDITION when using --array=0-2" >&2
        exit 2
    fi
    case "$SLURM_ARRAY_TASK_ID" in
        0) CONDITION=none ;;
        1) CONDITION=mach ;;
        2) CONDITION=mach_re ;;
        *) echo "M47 Slurm array indices must be 0, 1 or 2" >&2; exit 2 ;;
    esac
else
    CONDITION=${M47_CONDITION:-mach_re}
fi
EPOCHS=${M47_EPOCHS:-2}
RESUME=${M47_RESUME:-0}
CONFIRM=${M47_CONFIRM_LONG_TRAIN:-NO}
DISABLE_TENSORBOARD=${M47_DISABLE_TENSORBOARD:-NO}
case "$MODE" in smoke|train) ;; *) echo "M47_MODE must be smoke or train" >&2; exit 2 ;; esac
case "$CONDITION" in none|mach|mach_re) ;; *) echo "M47_CONDITION must be none, mach or mach_re" >&2; exit 2 ;; esac
case "$RESUME" in 0|1) ;; *) echo "M47_RESUME must be 0 or 1" >&2; exit 2 ;; esac
case "$DISABLE_TENSORBOARD" in NO|YES) ;; *) echo "M47_DISABLE_TENSORBOARD must be YES or NO" >&2; exit 2 ;; esac
if [ "$MODE" = train ] && [ "$CONFIRM" != YES ]; then
    echo "Refusing long training without M47_CONFIRM_LONG_TRAIN=YES" >&2
    exit 2
fi
if [ "$MODE" = smoke ] && [ "$RESUME" = 1 ]; then
    echo "Smoke jobs cannot resume" >&2; exit 2
fi
if [ "$EPOCHS" -lt 1 ]; then
    echo "M47_EPOCHS must be >=1" >&2; exit 2
fi
if [ "$MODE" = smoke ]; then
    DEFAULT_OUT=/scratch/coop/theret/GraphAttention_runs/m47_conditioned/${CONDITION}/smoke_${SLURM_JOB_ID}
else
    DEFAULT_OUT=/scratch/coop/theret/GraphAttention_runs/m47_conditioned/${CONDITION}/train_s42
fi
OUT=${M47_RUN_DIR:-$DEFAULT_OUT}
for path in "$REPO" "$IMAGE" "$ROOT" "$REFS" "$REF_AUDIT" "$QUALIFIED" "$INVENTORY" "$POLICY"; do
    test -e "$path" || { echo "Missing required path: $path" >&2; exit 2; }
done
test -x "$PYTHON" || { echo "Missing Python: $PYTHON" >&2; exit 2; }
for case_name in Re54_Ma0.05 Re54_Ma0.2 Re54_Ma0.3 Re94_Ma0.05 Re94_Ma0.2 Re94_Ma0.3; do
    test -s "$REFS/$case_name.yaml" || { echo "Missing M47 derived reference: $case_name" >&2; exit 2; }
done
cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
    echo "Refusing dirty code checkout" >&2
    git status --short >&2; exit 2
fi
if [ "$RESUME" = 0 ] && [ -e "$OUT" ]; then
    echo "Output already exists: $OUT" >&2; exit 2
fi
if [ "$RESUME" = 1 ] && [ ! -f "$OUT/last.pt" ]; then
    echo "Resume requires a preexisting last.pt in $OUT" >&2; exit 2
fi
if [ -f "$ENV_ROOT/bin/activate" ]; then source "$ENV_ROOT/bin/activate"; fi
BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done
echo "M47 conditional FM job=$SLURM_JOB_ID host=$(hostname) git=$(git rev-parse HEAD)"
echo "mode=$MODE condition=$CONDITION output=$OUT epochs=$EPOCHS resume=$RESUME"
echo "dataset=$ROOT refs=$REFS qualified=$QUALIFIED"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m pytest -q \
    tests/unit/test_m47_physics.py tests/unit/test_m43_b_multires.py \
    tests/unit/test_m43_b1b_checkpointing.py
EXTRA=()
if [ "$RESUME" = 1 ]; then EXTRA+=(--resume); fi
if [ "$DISABLE_TENSORBOARD" = YES ]; then EXTRA+=(--disable-tensorboard); fi
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -u -m scripts.train_m43_b_multires \
    --mode "$MODE" --root "$ROOT" --reference-dir "$REFS" \
    --qualified-manifest "$QUALIFIED" --source-inventory "$INVENTORY" \
    --selection-policy "$POLICY" --reference-audit "$REF_AUDIT" \
    --conditioning-mode "$CONDITION" --run-dir "$OUT" \
    --seed 42 --temporal-embargo 2 --accumulation 6 --max-epochs "$EPOCHS" \
    "${EXTRA[@]}"
echo "M47 result: $OUT/summary.json"