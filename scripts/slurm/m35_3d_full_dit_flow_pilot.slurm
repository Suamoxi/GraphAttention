#!/bin/bash
#SBATCH --job-name=m35_3d_full_dit
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
RUN_NAME=m35_3d_full_dit_flow_pilot_10e_h128_l10_h4_b1_acc8
RUN_DIR=${RUN_ROOT}/${RUN_NAME}

cd "$REPO"

echo "============================================================"
echo "M35 - 3D Full DiT Flow Matching pilot"
echo "============================================================"
echo "Attention:      global SDPA"
echo "Backbone:       DiT adaLN-Zero h128 L10 H4"
echo "Input:          full 3D AVBP HIT snapshots"
echo "Microbatch:     1 volume"
echo "Grad accum:     8"
echo "Nominal batch:  8 volumes / optimizer step"
echo "Epochs:         10"
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
        data=hit_3d_avbp \
        geometry=local \
        model=full_dit_transformer \
        task=hit_flow_matching \
        optimizer=adamw \
        +generative=hit_3d_flow_matching \
        seed=42 \
        run_name="$RUN_NAME" \
        task.time_embedding_scale=1000.0 \
        model.hidden_dim=128 \
        model.num_heads=4 \
        model.num_layers=10 \
        model.mlp_ratio=4 \
        model.condition_embed_dim=128 \
        model.dropout=0.0 \
        model.qkv_bias=true \
        model.out_proj_bias=false \
        model.use_sdpa=true \
        generative.batch_size=1 \
        generative.gradient_accumulation_steps=8 \
        generative.num_workers=4 \
        generative.max_epochs=10 \
        generative.device=cuda

for artifact in best.pt last.pt history.csv summary.json resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
    test -s "${RUN_DIR}/${artifact}"
done
cat "${RUN_DIR}/summary.json"
