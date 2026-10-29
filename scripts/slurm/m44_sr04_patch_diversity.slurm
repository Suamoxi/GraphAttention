#!/bin/bash
#SBATCH --job-name=m44_sr04_diversity
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# M44 SR-0.4: 16 training origins, 2 per snapshot per epoch, 50 epochs.
# All Python (including pytest) must execute in Singularity under grace Slurm.
# THIS IS POINT-SAMPLED DNS, not a physically filtered LES coarse field.
set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
ROOT=${M44_SR04_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
REFS=${M44_SR04_REFS:-/scratch/coop/theret/HIT_DNS_COMP_case_references}
CASE=${M44_SR04_CASE:-Re54/Ma0.3}
SCALES_RUN=${M44_SR04_FROZEN_SCALES_RUN:-/scratch/coop/theret/GraphAttention_runs/m44_sr01/job_484619}
POOL=${M44_SR04_POOL_SIZE:-16}
EPOCHS=${M44_SR04_EPOCHS:-50}
RESUME=${M44_SR04_RESUME:-0}
TENSORBOARD=${M44_SR04_TENSORBOARD:-1}
OUT=${M44_SR04_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m44_sr04/job_${SLURM_JOB_ID}}

if [ "$TENSORBOARD" != 0 ] && [ "$TENSORBOARD" != 1 ]; then
  echo "M44_SR04_TENSORBOARD must be 0 or 1" >&2
  exit 2
fi
if [ "$RESUME" != 0 ] && [ "$RESUME" != 1 ]; then
  echo "M44_SR04_RESUME must be 0 or 1" >&2
  exit 2
fi
if [ ! -s "$SCALES_RUN/manifest.json" ] || [ ! -s "$SCALES_RUN/scales.pt" ]; then
  echo "Frozen SR-0.3 source missing manifest.json or scales.pt: $SCALES_RUN" >&2
  exit 2
fi
if [ ! -d "$ROOT/$CASE" ] || [ ! -s "$REFS/${CASE//\//_}.yaml" ]; then
  echo "SR-0.4 DNS case or M3.3 reference file is unavailable" >&2
  exit 2
fi
if [ "$RESUME" = 0 ] && [ -e "$OUT" ]; then
  echo "Refusing to overwrite existing run: $OUT" >&2
  exit 2
fi
if [ "$RESUME" = 1 ] && [ ! -s "$OUT/last.pt" ]; then
  echo "Resume requires existing last.pt in: $OUT" >&2
  exit 2
fi

cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
  echo "Refusing dirty GraphAttention checkout" >&2
  git status --short >&2
  exit 2
fi

echo "=== M44 SR-0.4 training patch-diversity ablation ==="
echo "Revision=$(git rev-parse HEAD); host=$(hostname); job=${SLURM_JOB_ID}"
echo "Train snapshot pool=$POOL origins, active=2 per epoch, epochs=$EPOCHS"
echo "Baseline frozen scales: $SCALES_RUN"
echo "Output: $OUT"
echo "IMPORTANT: point-sampled DNS proxy; no conservative LES restriction"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "=== Run SR-0.4 and legacy correctness gates in Singularity ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m pytest -q \
  tests/unit/test_m44_tensorboard.py \
  tests/unit/test_m44_sr04_patch_diversity.py \
  tests/unit/test_m44_sr0_training.py \
  tests/unit/test_m44_sr0_training_runner.py \
  tests/unit/test_m44_sr03_point_constraint.py \
  tests/unit/test_m44_sr0_diagnostics_runner.py \
  tests/unit/test_flow_matching_task.py

EXTRA=()
if [ "$RESUME" = 1 ]; then EXTRA+=(--resume); fi
if [ "$TENSORBOARD" = 1 ]; then EXTRA+=(--tensorboard); fi
echo "=== SR-0.4 training inside Singularity ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -u -m scripts.train_m44_sr0 \
  --root "$ROOT" --reference-dir "$REFS" --case "$CASE" \
  --side 65 --patch-cells 8 --refinement 2 \
  --patches-per-snapshot 2 --train-patch-pool-size "$POOL" \
  --frozen-scales-run "$SCALES_RUN" --max-snapshots 24 \
  --embargo-snapshots 1 --epochs "$EPOCHS" \
  --condition-mode interpolated_only --point-sample-constraint \
  --hidden-dim 64 --layers 4 --heads 4 --seed 42 \
  --sample-steps 50 --device cuda --output "$OUT" \
  "${EXTRA[@]}"

test -s "$OUT/summary.json"
test -s "$OUT/best.pt"
echo "SR-0.4 summary: $OUT/summary.json"
echo "SR-0.4 best checkpoint: $OUT/best.pt"
