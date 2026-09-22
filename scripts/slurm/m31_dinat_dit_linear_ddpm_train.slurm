#!/bin/bash
#SBATCH --job-name=m31_dinatdit_ddpm
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
RUN_NAME=m31_dinat_dit_linear_ddpm_1000e_h128_l10_h4
RUN_DIR=${RUN_ROOT}/${RUN_NAME}

cd "$REPO"

echo "============================================================"
echo "M31 - DiNAT-DiT + linear-beta DDPM"
echo "============================================================"
echo "Attention:    alternating local / exact-two-hop"
echo "Geometry:     learned relative-displacement bias"
echo "Backbone:     DiT adaLN-Zero h128 L10 H4"
echo "Coordinates:  centered_bbox absolute embedding"
echo "T:            1000"
echo "beta_start:   1e-4"
echo "beta_end:     2e-2"
echo "Prediction:   epsilon"
echo "Epochs:       1000"
echo "Seed:         42"
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
    "$PYTHON" -u -m scripts.train_slice_linear_diffusion \
        data=hit_slice_pt \
        geometry=dinat_exact2 \
        model=dinat_dit \
        task=hit_diffusion_linear \
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
        model.dropout=0.0 \
        model.qkv_bias=true \
        model.out_proj_bias=false \
        generative.batch_size=128 \
        generative.num_workers=4 \
        generative.max_epochs=1000 \
        generative.device=cuda

for artifact in best.pt last.pt history.csv summary.json resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
    test -f "${RUN_DIR}/${artifact}"
done

cat "${RUN_DIR}/summary.json"
