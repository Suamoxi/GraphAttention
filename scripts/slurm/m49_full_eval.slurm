#!/bin/bash
#SBATCH --job-name=m49_full_eval
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err
set -euo pipefail
REPO=${M49_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M49_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M49_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
ROOT=${M49_ROOT:-/scratch/coop/theret/GraphAttention_runs/m49}
RUN="$ROOT/training/m49_full_dit_fp32_h128_l10_h4_b8_a2_seed42_blocked50"
EPOCH=${M49_EVAL_EPOCH:-75}
BATCH=${M49_EVAL_BATCH_SIZE:-8}
[[ "$BATCH" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid inference batch size"; exit 2; }
[[ "$EPOCH" =~ ^[1-9][0-9]{0,3}$ ]] && (( EPOCH <= 1000 )) || { echo "Invalid epoch"; exit 2; }
printf -v CKPT "epoch_%04d.pt" "$EPOCH"
test -s "$RUN/$CKPT" || { echo "Missing immutable checkpoint: $RUN/$CKPT"; exit 2; }
cd "$REPO"
test -d .git && test -z "$(git status --porcelain)" || { echo "Use a clean standalone checkout"; exit 2; }
BIND=(-B /home -B /scratch -B /gpfs-calypso/scratch:/gpfs-calypso/scratch)
test "$(singularity exec "${BIND[@]}" "$IMAGE" git -C "$REPO" rev-parse HEAD)" = "$(git rev-parse HEAD)"
singularity exec "${BIND[@]}" --env "PYTHONPATH=$REPO/src:$REPO" "$IMAGE" "$PYTHON" -m pytest -q tests/unit/test_m49_evaluation_checkpoint.py
# Reuse the qualified training microbatch by default. Inference needs no
# backward activations or optimizer state; larger values remain opt-in.
echo "Full DiT inference epoch=$EPOCH batch_size=$BATCH"
singularity exec --nv "${BIND[@]}" --env "PYTHONPATH=$REPO/src:$REPO" "$IMAGE" "$PYTHON" -u -m scripts.evaluate_m49_generation --run-dir "$RUN" --epoch "$EPOCH" --steps 50 --batch-size "$BATCH" --boundary
