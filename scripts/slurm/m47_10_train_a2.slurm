#!/usr/bin/env bash
#SBATCH --job-name=m47_10_a2
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err

# Requires real-mesh qualification; does not depend on A3 GPU diagnostics.
set -euo pipefail
REPO=${M47_10_REPO:-/scratch/coop/theret/GraphAttention}
exec bash "$REPO/scripts/slurm/m47_10_train_common.sh" a2
