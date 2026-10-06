#!/bin/bash
#SBATCH --job-name=m37_3d_dinat_gen
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

RUN_DIR=${RUN_DIR:-${RUN_ROOT}/m37_3d_dinat_dit_flow_1000e_h128_l10_h4_b16}
CHECKPOINT=${CHECKPOINT:-best.pt}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-4}
SAMPLING_STEPS=50
SAMPLING_SEED=5678

cd "$REPO"

for artifact in "$CHECKPOINT" resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
    test -s "${RUN_DIR}/${artifact}" || {
        echo "ERROR: missing training artifact: ${RUN_DIR}/${artifact}"
        exit 1
    }
done

# Freeze the evolving training checkpoint before generation so a resumed training
# job cannot change the weights midway through this evaluation.
SNAPSHOT_TMP="${RUN_DIR}/.eval_${CHECKPOINT%.pt}_${SLURM_JOB_ID}.tmp"
cp "${RUN_DIR}/${CHECKPOINT}" "$SNAPSHOT_TMP"
CHECKPOINT_EPOCH=$(singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -c "import torch; p=torch.load('$SNAPSHOT_TMP', map_location='cpu', weights_only=True); print(int(p['epoch']))")
EVAL_CHECKPOINT="eval_${CHECKPOINT%.pt}_epoch${CHECKPOINT_EPOCH}.pt"
if [ ! -s "${RUN_DIR}/${EVAL_CHECKPOINT}" ]; then
    mv "$SNAPSHOT_TMP" "${RUN_DIR}/${EVAL_CHECKPOINT}"
else
    rm -f "$SNAPSHOT_TMP"
fi

GEN_NAME=flow_heun_steps50_seed5678_${CHECKPOINT%.pt}_epoch${CHECKPOINT_EPOCH}
GEN_DIR=${RUN_DIR}/generations/${GEN_NAME}

echo "============================================================"
echo "M37 - full-volume 3-D NAT-DiT Flow inference"
echo "============================================================"
echo "Source run:     $RUN_DIR"
echo "Checkpoint:     $EVAL_CHECKPOINT (snapshot of current $CHECKPOINT, epoch $CHECKPOINT_EPOCH)"
echo "Batch size:     $BATCH_SIZE"
echo "Attention:      alternating local / exact-two-hop"
echo "Backend:        saved run configuration (torch_sparse for M37)"
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

echo "===== Optimized NAT backend correctness gate ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -m pytest \
        tests/unit/test_dinat_dit.py::test_dinat_dit_torch_sparse_backend_matches_scatter \
        tests/unit/test_dinat_dit.py::test_dinat_dit_torch_sparse_backend_supports_inference_mode_tensors \
        tests/unit/test_dinat_dit.py::test_dinat_dit_torch_sparse_cache_survives_inference_then_training \
        -q

nvidia-smi

singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
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
    test -s "${GEN_DIR}/${artifact}"
done

echo "===== M37 generation summary ====="
cat "${GEN_DIR}/summary.json"
