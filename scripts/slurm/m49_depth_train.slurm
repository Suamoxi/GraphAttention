#!/bin/bash
#SBATCH --job-name=m49_depth_train
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=1-4%2
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.err
set -euo pipefail
REPO=${M49_REPO:-/scratch/coop/theret/GraphAttention}
ROOT=${M49_ROOT:-/scratch/coop/theret/GraphAttention_runs/m49}
: "${M49_BENCH_ID:?Set the completed benchmark array job ID}"
VARIANTS=(10x12 9x123 8x123 20x12 15x123)
export M49_DEPTH_VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
export M49_DEPTH_REPORT="$ROOT/depth_benchmark/${M49_BENCH_ID}_${SLURM_ARRAY_TASK_ID}.json"
export M49_ARCH=dinat_depth
test -s "$M49_DEPTH_REPORT"
exec bash "$REPO/scripts/slurm/m49_full_dit_train.slurm"
