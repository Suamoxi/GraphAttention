#!/bin/bash
#SBATCH --job-name=m49_depth_eval
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=1-4%2
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.err
set -euo pipefail
REPO=${M49_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M49_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M49_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
ROOT=${M49_ROOT:-/scratch/coop/theret/GraphAttention_runs/m49}
: "${M49_BENCH_ID:?Set the benchmark array job ID}"
EPOCH=${M49_EVAL_EPOCH:-250}
VARIANTS=(10x12 9x123 8x123 20x12 15x123)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
REPORT="$ROOT/depth_benchmark/${M49_BENCH_ID}_${SLURM_ARRAY_TASK_ID}.json"
cd "$REPO"
test -d .git && test -z "$(git status --porcelain)" || { echo "Use clean standalone checkout"; exit 2; }
BIND=(-B /home -B /scratch)
test "$(singularity exec "${BIND[@]}" "$IMAGE" git -C "$REPO" rev-parse HEAD)" = "$(git rev-parse HEAD)"
SELECTION=$(singularity exec --nv "${BIND[@]}" --env "PYTHONPATH=$REPO/src:$REPO" "$IMAGE" "$PYTHON" -m scripts.benchmark_m49_depth --variant "$VARIANT" --select "$REPORT")
read -r LAYERS TOPO BATCH ACCUM AC <<< "$SELECTION"
RUN="$ROOT/training/m49_dinat_depth_${VARIANT}_fp32_h128_h4_b${BATCH}_a${ACCUM}_ac${AC}_seed42_blocked50"
singularity exec --nv "${BIND[@]}" --env "PYTHONPATH=$REPO/src:$REPO" "$IMAGE" "$PYTHON" -u -m scripts.evaluate_m49_generation --run-dir "$RUN" --epoch "$EPOCH" --steps 50 --batch-size "$BATCH" --boundary
