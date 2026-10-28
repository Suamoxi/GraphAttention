#!/bin/bash
#SBATCH --job-name=m44_sr02_diagnostics
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# READ-ONLY evaluation of an already trained SR-0.1 checkpoint.
# Only point-sampled DNS, not a filtered LES / physical SGS quality test.
set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
SOURCE=${M44_SR02_SOURCE:-}
OUT=${M44_SR02_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m44_sr02_diagnostics/job_${SLURM_JOB_ID}}
DRAWS=${M44_SR02_DRAWS:-4}
SAMPLE_STEPS=${M44_SR02_SAMPLE_STEPS:-50}
MAX_PATCHES=${M44_SR02_MAX_PATCHES:-2}
PATH_REPEATS=${M44_SR02_PATH_REPEATS:-16}

if [ -z "$SOURCE" ]; then
  echo 'Required: M44_SR02_SOURCE=/absolute/path/to/existing/m44_sr01/run' >&2
  exit 2
fi
if [ ! -s "$SOURCE/best.pt" ] || [ ! -s "$SOURCE/manifest.json" ] || [ ! -s "$SOURCE/scales.pt" ]; then
  echo "Source run missing best.pt, scales.pt or manifest.json: $SOURCE" >&2
  exit 2
fi
if [ -e "$OUT" ]; then
  echo "Refusing to overwrite: $OUT" >&2
  exit 2
fi
cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
  echo 'Refusing dirty repository for SR-0.2 diagnostics' >&2
  git status --short >&2
  exit 2
fi

echo "SR-0.2 revision=$(git rev-parse HEAD) original_run=$SOURCE output=$OUT"

echo '=== SR-0.2 point-sample and diagnostic correctness gates ==='
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m pytest -q \
  tests/unit/test_m44_sr0_evaluation.py \
  tests/unit/test_m44_sr03_point_constraint.py \
  tests/unit/test_m44_sr0_diagnostics_runner.py \
  tests/unit/test_m44_sr0_training_runner.py

echo '=== Re-evaluate saved best weights; zero training updates ==='
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -u -m scripts.diagnose_m44_sr0 \
  --run-dir "$SOURCE" --output "$OUT" --device cuda \
  --draws "$DRAWS" --sample-steps "$SAMPLE_STEPS" \
  --max-generation-patches "$MAX_PATCHES" \
  --path-repeats "$PATH_REPEATS"

echo "SR-0.2 report: $OUT/diagnostic_summary.json"
echo "SR-0.2 patch metrics: $OUT/flow_matching_per_patch.json"
