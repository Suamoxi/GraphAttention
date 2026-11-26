#!/bin/bash
# Submit mesh qualification -> full 128^3 A3 GPU test -> 100-epoch A1/A2/A3.
# Run this shell script on the Calypso login node. Do not execute Python there.
set -euo pipefail
REPO=${M47_10_REPO:-/scratch/coop/theret/GraphAttention}
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty repository: refusing submission" >&2; exit 2; }
for cmd in sbatch git; do command -v "$cmd" >/dev/null; done
rev=$(git rev-parse HEAD)
mesh=$(sbatch --parsable scripts/slurm/m47_10_mesh_preflight.slurm)
mesh=${mesh%%;*}
[[ "$mesh" =~ ^[0-9]+$ ]] || { echo "Unexpected mesh job ID $mesh" >&2; exit 2; }
gpu=$(sbatch --parsable --dependency="afterok:$mesh" scripts/slurm/m47_10_gpu_preflight.slurm)
gpu=${gpu%%;*}
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Unexpected GPU job ID $gpu" >&2; exit 2; }
train=$(sbatch --parsable --dependency="afterok:$gpu" --array=0-2%1 \
    scripts/slurm/m47_10_ablation_train.slurm)
train=${train%%;*}
[[ "$train" =~ ^[0-9]+$ ]] || { echo "Unexpected training array job ID $train" >&2; exit 2; }
printf 'Revision: %s\nMesh preflight: %s\nA3 full-size GPU gate: %s\nA1/A2/A3 training array: %s (0=A1, 1=A2, 2=A3)\n' \
    "$rev" "$mesh" "$gpu" "$train"
echo 'Training is contingent on both preflights passing. Check sacct/logs before resubmitting interrupted arms.'
