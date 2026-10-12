#!/bin/bash
#SBATCH --job-name=physics_token_dit
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

# Phase 1: 10-epoch integration pilot on the exact same 33^3 HIT split/task
# as M37 and M38. Set MAX_EPOCHS=1000 to continue production training.
MAX_EPOCHS=${MAX_EPOCHS:-10}
MICROBATCH=${MICROBATCH:-4}
ACCUM=${ACCUM:-4}
TOKEN_COUNT=${TOKEN_COUNT:-64}
TOKEN_CHUNK_SIZE=${TOKEN_CHUNK_SIZE:-null}
RUN_NAME=${RUN_NAME:-physics_token_3d_flow_h128_l10_h4_g64_b4_acc4}
RUN_DIR=${RUN_ROOT}/${RUN_NAME}
EFFECTIVE_BATCH=$((MICROBATCH * ACCUM))

cd "$REPO"
git rev-parse HEAD
git status --short
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked changes present; benchmark requires a clean checkout" >&2
    exit 1
fi
nvidia-smi

echo "============================================================"
echo "Standalone global Physics-Token DiT (Transolver-style learned slices)"
echo "NO DiNAT/graph edges; original Flow Matching objective and data pipeline"
echo "M37/M38 architecture: H128, L10, 4 attention heads"
echo "Global tokens per head: $TOKEN_COUNT"
echo "Token chunk size: $TOKEN_CHUNK_SIZE"
echo "Physical microbatch: $MICROBATCH"
echo "Gradient accumulation: $ACCUM (effective batch $EFFECTIVE_BATCH)"
echo "Target epochs: $MAX_EPOCHS"
echo "Output: $RUN_DIR"
echo "============================================================"

echo "===== Focused architecture and regression tests ====="
singularity exec --nv -B /home -B /scratch "$IMAGE" \
    "$PYTHON" -m pytest \
    tests/unit/test_physics_token_dit.py \
    tests/unit/test_dit_configs.py \
    tests/unit/test_dit_transformer.py \
    tests/unit/test_flow_matching_task.py \
    tests/unit/test_flow_matching_generation.py \
    -q

echo "===== Train independent physics-token baseline ====="
singularity exec --nv -B /home -B /scratch "$IMAGE" \
    "$PYTHON" -u -m scripts.train_generative \
    data=hit_3d_avbp \
    geometry=local \
    model=physics_token_dit \
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
    model.num_tokens="$TOKEN_COUNT" \
    model.slice_temperature=0.5 \
    model.token_chunk_size="$TOKEN_CHUNK_SIZE" \
    model.use_sdpa=true \
    generative.batch_size="$MICROBATCH" \
    generative.gradient_accumulation_steps="$ACCUM" \
    generative.resume=true \
    generative.num_workers=4 \
    generative.max_epochs="$MAX_EPOCHS" \
    generative.device=cuda

for artifact in best.pt last.pt history.csv summary.json resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
    test -s "$RUN_DIR/$artifact" || {
        echo "ERROR: missing training artifact $RUN_DIR/$artifact" >&2
        exit 1
    }
done
echo "===== Physics-token DiT final summary ====="
cat "$RUN_DIR/summary.json"
