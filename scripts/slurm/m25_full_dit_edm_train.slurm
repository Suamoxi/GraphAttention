#!/bin/bash
#SBATCH --job-name=m25_full_dit_edm
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
RUN_NAME=m25_full_dit_edm_1000e_h128_l10_h4
RUN_DIR=${RUN_ROOT}/${RUN_NAME}

cd "$REPO"

echo "============================================================"
echo "M25 - Full DiT + EDM"
echo "============================================================"
echo "Run:            $RUN_NAME"
echo "Attention:      full"
echo "Backbone:       DiT adaLN-Zero, h128 L10 H4"
echo "Coordinates:    centered_bbox absolute embedding"
echo "EDM sigma_data: 1.0"
echo "EDM P_mean:     -0.5068528194400547"
echo "EDM P_std:      1.2"
echo "Epochs:         1000"
echo "Seed:           42"
echo "============================================================"

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

nvidia-smi

singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.train_generative \
        data=hit_slice_pt \
        geometry=local \
        model=full_dit_transformer \
        task=hit_edm_diffusion \
        optimizer=adamw \
        +generative=hit_slice_diffusion \
        seed=42 \
        run_name="$RUN_NAME" \
        model.hidden_dim=128 \
        model.num_heads=4 \
        model.num_layers=10 \
        model.spatial_dim=2 \
        model.mlp_ratio=4 \
        model.condition_embed_dim=128 \
        generative.batch_size=128 \
        generative.num_workers=4 \
        generative.max_epochs=1000 \
        generative.device=cuda

test -f "${RUN_DIR}/best.pt"
test -f "${RUN_DIR}/last.pt"
test -f "${RUN_DIR}/history.csv"
test -f "${RUN_DIR}/summary.json"
test -f "${RUN_DIR}/resolved_config.yaml"
test -f "${RUN_DIR}/standardizers.pt"
test -f "${RUN_DIR}/dataset_split_manifest.json"

cat "${RUN_DIR}/summary.json"
