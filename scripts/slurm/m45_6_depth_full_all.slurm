#!/bin/bash
#SBATCH --job-name=m45_6_depth
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err
set -euo pipefail
REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -lt 0 ] || [ "$IDX" -ge 12 ]; then echo 'Use indices 0..11' >&2; exit 2; fi
# Pair adjacent array indices on a common dataset, seed, and normalization.
if [ $((IDX % 2)) -eq 0 ]; then DEPTH=2; else DEPTH=4; fi
SEED=$((2026 + (IDX / 2) % 3))
if [ $((IDX / 6)) -eq 0 ]; then SAMPLES=24; else SAMPLES=256; fi
STEPS=${M45_6_STEPS:-2000}
GENERATED=${M45_6_GENERATED_SAMPLES:-128}
DRAWS=${M45_6_VELOCITY_DRAWS:-4}
OUT=${M45_6_OUT:-/scratch/coop/theret/GraphAttention_runs/m45_6_depth_full_all/job_${SLURM_JOB_ID}}
cd "$REPO"
[ -z "$(git status --porcelain)" ] || { echo "Dirty checkout" >&2; exit 2; }
[ -s "$IMAGE" ] && [ -x "$PYTHON" ] || { echo "Missing image or Python" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "Refusing existing output: $OUT" >&2; exit 2; }
echo "M45.6 depth study Git=$(git rev-parse HEAD) array=$IDX depth=$DEPTH seed=$SEED training=$SAMPLES steps=$STEPS"
nvidia-smi --query-gpu=name,memory.total --format=csv
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m py_compile scripts/m45_synthetic_pilot.py \
  scripts/m45_checkpoint_diagnostics.py scripts/m45_6_compare_depth.py \
  tests/unit/test_m45_6_diagnostics.py
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m pytest -q tests/unit/test_m45_6_diagnostics.py
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -u -m scripts.m45_synthetic_pilot \
  --output-dir "$OUT" --device cuda --operator block_average \
  --source-subspace nullspace --coarse-side 4 --refinement 2 --core-side 4 \
  --cases shared_mode --train-samples "$SAMPLES" --val-samples 8 \
  --seed "$SEED" --val-seed 7002026 --steps "$STEPS" \
  --heun-steps 8 --generated-samples 4 --training-scope full_all \
  --width 16 --heads 4 \
  --fine-pre-layers "$DEPTH" --fine-post-layers "$DEPTH" \
  --coarse-static-layers "$DEPTH" --coarse-dynamic-layers "$DEPTH"
test -s "$OUT/summary.json"
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -u -m scripts.m45_checkpoint_diagnostics \
  --checkpoint-root "$OUT" --output-dir "$OUT/diagnostics" --device cuda \
  --cases shared_mode --generated-samples "$GENERATED" \
  --velocity-noise-draws "$DRAWS" --heun-steps 25 --seed 53177
test -s "$OUT/diagnostics/generation_metrics.csv"
test -s "$OUT/diagnostics/time_summary.csv"
echo "M45.6 depth completed: $OUT"
