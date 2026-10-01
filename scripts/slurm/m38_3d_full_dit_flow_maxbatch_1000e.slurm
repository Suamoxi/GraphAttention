#!/bin/bash
#SBATCH --job-name=m38_3d_full_1000e
#SBATCH --time=20-00:00:00
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
PROBE_JSON=${RUN_ROOT}/m38_3d_full_max_batch_probe.json

cd "$REPO"

echo "============================================================"
echo "M38 - 3D Full DiT SDPA, maximum microbatch, 1000 epochs"
echo "============================================================"
echo "Attention:       full global self-attention"
echo "Backend:         PyTorch optimized SDPA"
echo "Backbone:        DiT adaLN-Zero h128 L10 H4"
echo "Batch policy:    largest passing microbatch <= 64"
echo "Grad accum:      1"
echo "Epochs:          1000"
echo "Seed:            42"
echo "============================================================"

git rev-parse HEAD
git status --short
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

nvidia-smi

echo
echo "===== Find maximum Full-DiT microbatch ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u scripts/probe_3d_max_batch.py \
        --model full \
        --max-batch 64 \
        --output "$PROBE_JSON"

BATCH=$(singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -c "import json; print(json.load(open('$PROBE_JSON'))['max_passing_batch'])")

RUN_NAME="m38_3d_full_dit_flow_1000e_h128_l10_h4_b${BATCH}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"

echo "Selected Full-DiT batch size: ${BATCH}"
if [ -e "$RUN_DIR" ]; then
    echo "ERROR: run directory already exists: $RUN_DIR"
    exit 1
fi

echo
echo "===== Train Full DiT for 1000 epochs ====="
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
        generative.batch_size="$BATCH" \
        generative.gradient_accumulation_steps=1 \
        generative.num_workers=4 \
        generative.max_epochs=1000 \
        generative.device=cuda

for artifact in best.pt last.pt history.csv summary.json resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
    test -s "${RUN_DIR}/${artifact}"
done

echo "===== Full-DiT max-batch probe ====="
cat "$PROBE_JSON"
echo "===== Full-DiT final summary ====="
cat "${RUN_DIR}/summary.json"
