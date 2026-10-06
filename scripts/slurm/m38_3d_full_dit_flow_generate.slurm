#!/bin/bash
#SBATCH --job-name=m38_3d_full_gen
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

RUN_DIR=${RUN_DIR:-${RUN_ROOT}/m38_3d_full_dit_flow_1000e_h128_l10_h4_b18}
CHECKPOINT=${CHECKPOINT:-last.pt}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-4}
SAMPLING_STEPS=50
SAMPLING_SEED=5678
CHECKPOINT_EPOCH=$(singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -c "import torch; p=torch.load('$RUN_DIR/$CHECKPOINT', map_location='cpu', weights_only=True); print(int(p['epoch']))")
GEN_NAME=flow_heun_steps50_seed5678_${CHECKPOINT%.pt}_epoch${CHECKPOINT_EPOCH}
GEN_DIR=${RUN_DIR}/generations/${GEN_NAME}

cd "$REPO"

echo "============================================================"
echo "M38 - full-volume 3-D Full-DiT Flow inference"
echo "============================================================"
echo "Source run:     $RUN_DIR"
echo "Checkpoint:     $CHECKPOINT (epoch $CHECKPOINT_EPOCH)"
echo "Batch size:     $BATCH_SIZE"
echo "Sampler:        Flow ODE / Heun"
echo "Steps:          $SAMPLING_STEPS (100 model evaluations)"
echo "Sampling seed:  $SAMPLING_SEED"
echo "Output:         $GEN_DIR"
echo "============================================================"

git rev-parse HEAD
git status --short
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

for artifact in "$CHECKPOINT" resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
    test -s "${RUN_DIR}/${artifact}" || {
        echo "ERROR: missing training artifact: ${RUN_DIR}/${artifact}"
        exit 1
    }
done

if [ -s "${GEN_DIR}/generated_test.pt" ] && [ -s "${GEN_DIR}/summary.json" ]; then
    echo "Generation is already complete; reusing:"
    echo "  $GEN_DIR"
    cat "${GEN_DIR}/summary.json"
    exit 0
fi

OVERWRITE=false
if [ -e "$GEN_DIR" ]; then
    OVERWRITE=true
fi

nvidia-smi

singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.generate_generative \
        run_dir="$RUN_DIR" \
        checkpoint="$CHECKPOINT" \
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
    test -s "${GEN_DIR}/${artifact}"
done

echo "===== M38 generation summary ====="
cat "${GEN_DIR}/summary.json"
