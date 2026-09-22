#!/bin/bash
#SBATCH --job-name=m30_dit_fm_d4arch
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
RUN_NAME=m30_full_dit_flow_tscale1000_d4arch_1000e_h128_l6_h4_temb256
RUN_DIR=${RUN_ROOT}/${RUN_NAME}

cd "$REPO"

echo "============================================================"
echo "M30 - Flow Matching with diffusion4avbp-matched DiT architecture"
echo "============================================================"
echo "Hidden:              128"
echo "Heads:               4"
echo "Layers:              6"
echo "MLP ratio:           4"
echo "Condition embed dim: 256"
echo "Dropout:             0.1"
echo "QKV bias:            true"
echo "Output-proj bias:    true"
echo "Expected params:     1,881,733"
echo "Time scale:          1000"
echo
echo "Optimization intentionally remains M28-like:"
echo "Batch:               128"
echo "LR:                  1e-4"
echo "Epochs:              1000"
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

singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.train_generative \
        data=hit_slice_pt \
        geometry=local \
        model=full_dit_transformer \
        task=hit_flow_matching \
        optimizer=adamw \
        +generative=hit_slice_flow_matching \
        seed=42 \
        run_name="$RUN_NAME" \
        task.time_embedding_scale=1000.0 \
        model.hidden_dim=128 \
        model.num_heads=4 \
        model.num_layers=6 \
        model.spatial_dim=2 \
        model.mlp_ratio=4 \
        model.condition_embed_dim=256 \
        model.dropout=0.1 \
        model.qkv_bias=true \
        model.out_proj_bias=true \
        generative.batch_size=128 \
        generative.num_workers=4 \
        generative.max_epochs=1000 \
        generative.device=cuda

for artifact in best.pt last.pt history.csv summary.json resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
    test -f "${RUN_DIR}/${artifact}"
done

cat "${RUN_DIR}/summary.json"
