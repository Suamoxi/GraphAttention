#!/bin/bash
#SBATCH --job-name=m49_full_train
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err

# Provisional numbered-snapshot random split until the stationarity audit is reviewed.
# M49_MAX_EPOCHS is cumulative, not additional epochs.
set -euo pipefail
REPO=${M49_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M49_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M49_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
ROOT=${M49_ROOT:-/scratch/coop/theret/GraphAttention_runs/m49}
MAX_EPOCHS=${M49_MAX_EPOCHS:-25}
MODE=${M49_MODE:-start}
RUN_NAME=m49_full_dit_fp32_h128_l10_h4_b8_a2_seed42_numbered_random
RUN_DIR="$ROOT/training/$RUN_NAME"

cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty checkout"; exit 2; }
[[ "$MAX_EPOCHS" =~ ^[1-9][0-9]{0,3}$ ]] && (( MAX_EPOCHS <= 1000 )) || {
    echo "M49_MAX_EPOCHS must be an integer in [1,1000]"; exit 2;
}
mkdir -p "$ROOT/training"
# Prevent two allocations from writing the same run.
exec 9>"$ROOT/training/${RUN_NAME}.lock"
flock -n 9 || { echo "Another job holds the run lock"; exit 2; }
case "$MODE" in
    start)
        test ! -e "$RUN_DIR" || { echo "Run already exists; use resume explicitly"; exit 2; }
        RESUME=false
        ;;
    resume)
        test -s "$RUN_DIR/last.pt" || { echo "Missing last.pt"; exit 2; }
        RESUME=true
        ;;
    *) echo "M49_MODE must be start or resume"; exit 2;;
esac

git rev-parse HEAD
nvidia-smi
echo "RUN_DIR=$RUN_DIR cumulative_epochs=$MAX_EPOCHS resume=$RESUME"
# Explicitly use this checkout's source, including in a detached worktree.
singularity exec --nv -B /home -B /scratch --env "PYTHONPATH=$REPO/src:$REPO" \
    "$IMAGE" "$PYTHON" -u -m scripts.train_generative \
    data=hit_3d_avbp 'data.snapshot_pattern=solut_hit_*.h5' \
    geometry=local model=full_dit_transformer task=hit_flow_matching \
    optimizer=adamw +generative=hit_3d_flow_matching \
    seed=42 run_name="$RUN_NAME" \
    task.time_embedding_scale=1000.0 \
    model.hidden_dim=128 model.num_heads=4 model.num_layers=10 \
    model.mlp_ratio=4 model.condition_embed_dim=128 \
    model.dropout=0.0 model.qkv_bias=true model.out_proj_bias=false \
    model.use_sdpa=true model.activation_checkpointing=false \
    generative.output_root="$ROOT/training" \
    generative.batch_size=8 generative.gradient_accumulation_steps=2 \
    generative.num_workers=4 generative.device=cuda \
    generative.max_epochs="$MAX_EPOCHS" generative.resume="$RESUME" \
    generative.strict_resume=true generative.checkpoint_every=25 \
    generative.evaluate_test=false

for artifact in last.pt best.pt history.csv summary.json resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
    test -s "$RUN_DIR/$artifact"
done
