#!/bin/bash
#SBATCH --job-name=m44_sr05_extend
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# Paired 50 -> 150 epoch continuations, FULL optimizer/RNG state.
# ALL Python/pytest is executed in Singularity on the grace compute node.
# Output is separate from the original SR-0.3 / SR-0.4 training runs.
set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
ROOT=${M44_SR05_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
REFS=${M44_SR05_REFS:-/scratch/coop/theret/HIT_DNS_COMP_case_references}
MODE=${M44_SR05_MODE:-}
ORIGINAL_BASE=/scratch/coop/theret/GraphAttention_runs/m44_sr01/job_484619
ORIGINAL_DIVERSE=/scratch/coop/theret/GraphAttention_runs/m44_sr04/job_485096
ORIGINAL_BASE=${M44_SR05_ORIGINAL_BASE:-$ORIGINAL_BASE}
ORIGINAL_DIVERSE=${M44_SR05_ORIGINAL_DIVERSE:-$ORIGINAL_DIVERSE}
EPOCHS=${M44_SR05_TOTAL_EPOCHS:-150}
RESUME=${M44_SR05_RESUME:-0}
TENSORBOARD=${M44_SR05_TENSORBOARD:-1}
OUT=${M44_SR05_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m44_sr05/${MODE}_job_${SLURM_JOB_ID}}

if [ "$MODE" != baseline ] && [ "$MODE" != diverse ]; then
  echo "Required: M44_SR05_MODE=baseline or M44_SR05_MODE=diverse" >&2
  exit 2
fi
for flag in "$RESUME" "$TENSORBOARD"; do
  if [ "$flag" != 0 ] && [ "$flag" != 1 ]; then
    echo "M44_SR05_RESUME and M44_SR05_TENSORBOARD must be 0 or 1" >&2
    exit 2
  fi
done
if [ "$EPOCHS" -le 50 ]; then
  echo "M44_SR05_TOTAL_EPOCHS must exceed 50" >&2
  exit 2
fi
SOURCE=$ORIGINAL_BASE
if [ "$MODE" = diverse ]; then SOURCE=$ORIGINAL_DIVERSE; fi
for file in manifest.json scales.pt best.pt last.pt history.json; do
  if [ ! -s "$SOURCE/$file" ]; then
    echo "Source is missing $file: $SOURCE" >&2
    exit 2
  fi
done
if [ ! -s "$ORIGINAL_BASE/manifest.json" ] || [ ! -s "$ORIGINAL_BASE/scales.pt" ]; then
  echo "Missing original constrained baseline used for frozen SR-0.4 scaling" >&2
  exit 2
fi
if [ ! -d "$ROOT/Re54/Ma0.3" ] || [ ! -s "$REFS/Re54_Ma0.3.yaml" ]; then
  echo "Missing original DNS case or reference YAML" >&2
  exit 2
fi
if [ "$RESUME" = 0 ] && [ -e "$OUT" ]; then
  echo "Refusing to overwrite an existing continuation: $OUT" >&2
  exit 2
fi
if [ "$RESUME" = 1 ] && [ ! -s "$OUT/continuation_provenance.json" ]; then
  echo "Resume requires a previous SR-0.5 continuation directory" >&2
  exit 2
fi
cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
  echo "Refusing dirty GraphAttention checkout" >&2
  git status --short >&2
  exit 2
fi

echo "=== M44 SR-0.5 $MODE: full-state continuation to $EPOCHS epochs ==="
echo "Git=$(git rev-parse HEAD); host=$(hostname); job=${SLURM_JOB_ID}"
echo "Original=$SOURCE"
echo "Output=$OUT"
echo "Budget: 15 snapshots x 2 patches x $EPOCHS epochs"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "=== Correctness gates: Singularity on grace ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m pytest -q \
  tests/unit/test_m44_sr05_continuation.py \
  tests/unit/test_m44_sr05_diagnostics.py \
  tests/unit/test_m44_sr04_patch_diversity.py \
  tests/unit/test_m44_sr0_training_runner.py \
  tests/unit/test_m44_sr03_point_constraint.py \
  tests/unit/test_m44_tensorboard.py

EXTRA=()
if [ "$RESUME" = 1 ]; then EXTRA+=(--resume); fi

echo "=== Verify/copy initial 50-epoch optimizer checkpoint in Singularity ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -u -m scripts.prepare_m44_sr05_continuation \
  --source-run "$SOURCE" --output "$OUT" --mode "$MODE" \
  --initial-epochs 50 --target-epochs "$EPOCHS" \
  "${EXTRA[@]}"

TRAIN_EXTRA=(--resume --point-sample-constraint)
if [ "$MODE" = diverse ]; then
  TRAIN_EXTRA+=(--train-patch-pool-size 16 --frozen-scales-run "$ORIGINAL_BASE")
fi
if [ "$TENSORBOARD" = 1 ]; then TRAIN_EXTRA+=(--tensorboard); fi

echo "=== Continue training in Singularity (never restart weights/optimizer) ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -u -m scripts.train_m44_sr0 \
  --root "$ROOT" --reference-dir "$REFS" --case Re54/Ma0.3 \
  --side 65 --patch-cells 8 --refinement 2 \
  --patches-per-snapshot 2 --max-snapshots 24 --embargo-snapshots 1 \
  --epochs "$EPOCHS" --lr 0.001 --seed 42 \
  --condition-mode interpolated_only \
  --hidden-dim 64 --layers 4 --heads 4 --sparse-attention-backend scatter \
  --sample-steps 50 --device cuda --output "$OUT" \
  "${TRAIN_EXTRA[@]}"

test -s "$OUT/summary.json"
test -s "$OUT/best.pt"
test -s "$OUT/last.pt"
test -s "$OUT/continuation_provenance.json"
echo "M44 SR-0.5 $MODE summary: $OUT/summary.json"
echo "M44 SR-0.5 $MODE best checkpoint: $OUT/best.pt"
echo "M44 SR-0.5 $MODE continuation provenance: $OUT/continuation_provenance.json"
