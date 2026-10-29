#!/bin/bash
#SBATCH --job-name=m44_tb_backfill
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# Convert existing M44 history.json to TensorBoard events via grace Slurm +
# Singularity. No training Python or pytest runs on the cluster login node.
# One GPU is reserved for the validated grace allocation pattern but not used.
set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
SOURCE=${M44_TB_SOURCE:-}
REFRESH=${M44_TB_REFRESH:-0}

if [ -z "$SOURCE" ] || [ "${SOURCE#/}" = "$SOURCE" ]; then
  echo 'Required: M44_TB_SOURCE=/absolute/path/to/existing/M44/run' >&2
  exit 2
fi
if [ "$REFRESH" != 0 ] && [ "$REFRESH" != 1 ]; then
  echo 'M44_TB_REFRESH must be 0 or 1' >&2
  exit 2
fi
if [ ! -s "$SOURCE/manifest.json" ] || [ ! -s "$SOURCE/history.json" ]; then
  echo "Run is missing manifest.json or history.json: $SOURCE" >&2
  exit 2
fi
cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
  echo "Refusing dirty GraphAttention checkout" >&2
  git status --short >&2
  exit 2
fi
echo "M44 TensorBoard backfill from $SOURCE; git=$(git rev-parse HEAD); refresh=$REFRESH"
echo "NOTE: history-only import is epoch-level and does not regenerate model histograms."
echo "NOTE: refresh is intended only for historical jobs not logging natively."

echo "=== Correctness tests inside Singularity ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m pytest -q tests/unit/test_m44_tensorboard.py

EXTRA=()
if [ "$REFRESH" = 1 ]; then EXTRA+=(--refresh); fi
echo "=== Backfill TensorBoard event scalars inside Singularity ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -u -m scripts.backfill_m44_tensorboard \
  --run-dir "$SOURCE" "${EXTRA[@]}"

echo "TensorBoard log directory: $SOURCE/tensorboard"
