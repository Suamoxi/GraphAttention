#!/bin/bash
#SBATCH --job-name=m49_scale
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-20%2
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.err
set -euo pipefail
REPO=${M49_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M49_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M49_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
ROOT=${M49_ROOT:-/scratch/coop/theret/GraphAttention_runs/m49}
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty checkout"; exit 2; }
SIDES=(9 17 25 33 49 65 129)
I=${SLURM_ARRAY_TASK_ID}
SIDE=${SIDES[$((I%7))]}
case $((I/7)) in
  0) MODEL=full; BACKEND=sdpa; EXTRA=() ;;
  1) MODEL=dinat; BACKEND=torch_sparse_triton; EXTRA=() ;;
  2) MODEL=dinat; BACKEND=torch_sparse_triton; EXTRA=(--checkpointing) ;;
  *) exit 2 ;;
esac
OUT="$ROOT/scaling/${SLURM_ARRAY_JOB_ID}_${I}.json"
echo "ARTIFACT_REPORT=$OUT"
nvidia-smi
singularity exec --nv -B /home -B /scratch "$IMAGE" "$PYTHON" -u -m scripts.benchmark_m49_scaling --model "$MODEL" --side "$SIDE" --backend "$BACKEND" "${EXTRA[@]}" --output "$OUT"
