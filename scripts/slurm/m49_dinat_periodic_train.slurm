#!/bin/bash
#SBATCH --job-name=m49_periodic_train
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err
set -euo pipefail
export M49_ARCH=dinat_periodic
REPO=${M49_REPO:-/scratch/coop/theret/GraphAttention}
exec bash "$REPO/scripts/slurm/m49_full_dit_train.slurm"
