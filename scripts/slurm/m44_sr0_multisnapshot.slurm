#!/bin/bash
#SBATCH --job-name=m44_sr01_train
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# M44 SR-0.1: conditional FM on *point-sampled DNS proxy*, NOT physical LES.
# Defaults to a bounded, 24-snapshot, two-patches-per-snapshot experiment.
set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
ROOT=${M44_SR01_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
REFS=${M44_SR01_REFS:-/scratch/coop/theret/HIT_DNS_COMP_case_references}
CASE=${M44_SR01_CASE:-Re54/Ma0.3}
OUT=${M44_SR01_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m44_sr01/job_${SLURM_JOB_ID}}
EPOCHS=${M44_SR01_EPOCHS:-5}
MAX_SNAPSHOTS=${M44_SR01_MAX_SNAPSHOTS:-24}
PATCHES=${M44_SR01_PATCHES:-2}
RESUME=${M44_SR01_RESUME:-0}

if [ "$RESUME" != 0 ] && [ "$RESUME" != 1 ]; then
  echo "M44_SR01_RESUME must be 0 or 1" >&2
  exit 2
fi
cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
  echo "Refusing dirty GraphAttention checkout" >&2
  git status --short >&2
  exit 2
fi
REF_NAME=${CASE//\//_}.yaml
test -s "$REFS/$REF_NAME"
test -d "$ROOT/$CASE"
if [ "$RESUME" = 0 ] && [ -e "$OUT" ]; then
  echo "Refusing to overwrite existing run: $OUT" >&2
  exit 2
fi
if [ "$RESUME" = 1 ] && [ ! -f "$OUT/last.pt" ]; then
  echo "Resume requires existing checkpoint: $OUT/last.pt" >&2
  exit 2
fi

echo "=== M44 SR-0.1 snapshot-held-out small-patch experiment ==="
echo "Git: $(git rev-parse HEAD); host: $(hostname); job: ${SLURM_JOB_ID}"
echo "Output: $OUT; epochs: $EPOCHS; max snapshots: $MAX_SNAPSHOTS; patches/snapshot: $PATCHES"
echo "WARNING: point-sampled DNS condition is not physical LES filtering"
nvidia-smi --query-gpu=name,memory.total --format=csv

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m pytest -q \
  tests/unit/test_m44_sr0.py \
  tests/unit/test_m44_sr0_training.py \
  tests/unit/test_m44_sr0_training_runner.py \
  tests/unit/test_m44_sr0_cartesian.py \
  tests/unit/test_m44_sr0_pilot_pair.py \
  tests/unit/test_flow_matching_task.py

EXTRA=()
if [ "$RESUME" = 1 ]; then EXTRA+=(--resume); fi
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -u -m scripts.train_m44_sr0 \
  --root "$ROOT" --reference-dir "$REFS" --case "$CASE" \
  --side 65 --patch-cells 8 --refinement 2 \
  --patches-per-snapshot "$PATCHES" --max-snapshots "$MAX_SNAPSHOTS" \
  --embargo-snapshots 1 --epochs "$EPOCHS" \
  --condition-mode interpolated_only \
  --hidden-dim 64 --layers 4 --heads 4 \
  --sample-steps 50 --device cuda --output "$OUT" \
  "${EXTRA[@]}"

echo "SR-0.1 summary: $OUT/summary.json"
echo "Best model: $OUT/best.pt"
