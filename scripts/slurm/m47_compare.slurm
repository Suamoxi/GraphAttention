#!/bin/bash
#SBATCH --job-name=m47_cmp_s42
#SBATCH --partition=grace
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err

# Submit only after all three training arms completed successfully.
# sbatch --dependency=afterok:<TRAIN_ARRAY_JOB_ID> scripts/slurm/m47_compare.slurm
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M47_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
ROOT=${M47_COMPARE_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_conditioned}
BASENAME=${M47_COMPARE_BASENAME:-train_s42}

cd "$REPO"
test -f "$IMAGE"
test -x "$PYTHON"
test -d "$ROOT"
test -z "$(git status --porcelain)" || { echo "Dirty repository; comparison refused" >&2; exit 2; }
echo "M47 matched comparison: git=$(git rev-parse HEAD), root=$ROOT, basename=$BASENAME"
singularity exec -B /home -B /scratch "$IMAGE" "$PYTHON" -m pytest -q tests/unit/test_m47_compare.py
singularity exec -B /home -B /scratch "$IMAGE" "$PYTHON" -u -m scripts.m47_compare \
  --root "$ROOT" --basename "$BASENAME"