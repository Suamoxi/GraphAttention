#!/bin/bash
#SBATCH --job-name=m21_linear_ddpm_train
#SBATCH --time=04:00:00
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
RUN_NAME=m21_m12_diffusion_linear_1000e_h128_l10_h4
RUN_DIR=${RUN_ROOT}/${RUN_NAME}

cd "$REPO"

printf '%s\n' "============================================================"
printf '%s\n' "M21 - Linear-beta DDPM schedule ablation"
printf '%s\n' "============================================================"
printf 'Run:          %s\n' "$RUN_NAME"
printf '%s\n' "T:            1000"
printf '%s\n' "beta_start:   1e-4"
printf '%s\n' "beta_end:     2e-2"
printf '%s\n' "Prediction:   epsilon"
printf '%s\n' "Backbone:     M12 L10 h128 h4"
printf '%s\n' "Epochs:       1000"
printf '%s\n' "============================================================"

echo
echo "===== Git ====="
git rev-parse HEAD
git status --short

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

if [ -e "$RUN_DIR" ]; then
    echo "ERROR: run directory already exists: $RUN_DIR"
    exit 1
fi

echo
echo "===== GPU ====="
nvidia-smi

echo
echo "===== Training ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.train_slice_linear_diffusion \
        data=hit_slice_pt \
        geometry=dinat_exact2 \
        model=alternating_dilated_geometric_sparse_transformer \
        task=hit_diffusion_linear \
        optimizer=adamw \
        +generative=hit_slice_diffusion \
        run_name="$RUN_NAME" \
        model.hidden_dim=128 \
        model.num_heads=4 \
        model.num_layers=10 \
        model.spatial_dim=2 \
        model.mlp_ratio=4 \
        generative.batch_size=128 \
        generative.num_workers=4 \
        generative.max_epochs=1000 \
        generative.device=cuda

echo
echo "===== Checking outputs ====="
test -f "${RUN_DIR}/best.pt"
test -f "${RUN_DIR}/last.pt"
test -f "${RUN_DIR}/history.csv"
test -f "${RUN_DIR}/summary.json"
test -f "${RUN_DIR}/resolved_config.yaml"
test -f "${RUN_DIR}/standardizers.pt"
test -f "${RUN_DIR}/dataset_split_manifest.json"

echo
echo "===== Summary ====="
cat "${RUN_DIR}/summary.json"

echo
echo "============================================================"
echo "M21 TRAINING COMPLETE"
echo "$RUN_DIR"
echo "============================================================"
