#!/bin/bash
#SBATCH --job-name=physics_token_gen
#SBATCH --time=12:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
RUN_ROOT=/scratch/coop/theret/GraphAttention_runs
RUN_DIR=${RUN_DIR:-${RUN_ROOT}/physics_token_3d_flow_h128_l10_h4_g64_b4_acc4}
CHECKPOINT=${CHECKPOINT:-best.pt}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
SAMPLING_STEPS=50
SAMPLING_SEED=5678

cd "$REPO"
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: modified tracked files; use a clean checkout" >&2
    exit 1
fi
for artifact in "$CHECKPOINT" resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
    test -s "${RUN_DIR}/${artifact}" || {
        echo "ERROR: missing training artifact ${RUN_DIR}/${artifact}" >&2
        exit 1
    }
done
# Freeze checkpoint while an ongoing resume job may update best.pt.
SNAPSHOT_TMP="${RUN_DIR}/.physics_eval_${CHECKPOINT%.pt}_${SLURM_JOB_ID}.tmp"
cp "${RUN_DIR}/${CHECKPOINT}" "$SNAPSHOT_TMP"
CHECKPOINT_EPOCH=$(singularity exec -B /home -B /scratch "$IMAGE" \
    "$PYTHON" -c "import torch; print(int(torch.load('$SNAPSHOT_TMP', map_location='cpu', weights_only=True)['epoch']))")
EVAL_CHECKPOINT="physics_eval_${CHECKPOINT%.pt}_epoch${CHECKPOINT_EPOCH}.pt"
if [ ! -s "${RUN_DIR}/${EVAL_CHECKPOINT}" ]; then
    mv "$SNAPSHOT_TMP" "${RUN_DIR}/${EVAL_CHECKPOINT}"
else
    rm -f "$SNAPSHOT_TMP"
fi
GEN_NAME="flow_heun_steps50_seed5678_${CHECKPOINT%.pt}_epoch${CHECKPOINT_EPOCH}"
GEN_DIR="${RUN_DIR}/generations/${GEN_NAME}"
echo "Physics-token DiT generation (own checkpoint, never M37)"
echo "Run: $RUN_DIR"
echo "Checkpoint: $EVAL_CHECKPOINT"
echo "Output: $GEN_DIR"
git rev-parse HEAD
nvidia-smi

if [ -s "$GEN_DIR/generated_test.pt" ] && [ -s "$GEN_DIR/summary.json" ]; then
    echo "Existing complete generation found; no overwrite"
    cat "$GEN_DIR/summary.json"
    exit 0
fi
OVERWRITE=false
if [ -e "$GEN_DIR" ]; then
    OVERWRITE=true
fi
singularity exec --nv -B /home -B /scratch "$IMAGE" \
    "$PYTHON" -u -m scripts.generate_generative \
    run_dir="$RUN_DIR" \
    checkpoint="$EVAL_CHECKPOINT" \
    device=cuda \
    batch_size="$BATCH_SIZE" \
    num_workers="$NUM_WORKERS" \
    overwrite="$OVERWRITE" \
    sampling.steps="$SAMPLING_STEPS" \
    sampling.method=flow_ode \
    sampling.solver=heun \
    sampling.sampling_eps=0.0 \
    sampling.final_denoise=false \
    sampling.seed="$SAMPLING_SEED" \
    output_name="$GEN_NAME"

for artifact in generated_test.pt summary.json resolved_config.yaml generation_config.yaml dataset_split_manifest.json; do
    test -s "$GEN_DIR/$artifact"
done
cat "$GEN_DIR/summary.json"
