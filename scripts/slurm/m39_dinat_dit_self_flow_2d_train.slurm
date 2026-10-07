#!/bin/bash
#SBATCH --job-name=m39_dinat_self_2d
#SBATCH --time=02:00:00
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

RUN_NAME=m39_dinat_dit_self_flow_tscale1000_1000e_h128_l10_h4
RUN_DIR=${RUN_ROOT}/${RUN_NAME}

cd "$REPO"

echo "============================================================"
echo "M39 - 2-D DiNAT-DiT + self attention + Flow Matching"
echo "============================================================"
echo "Attention local:   self + one-hop"
echo "Attention dilated: self + exact-two-hop"
echo "Geometry:          learned relative-displacement bias"
echo "Backend:           torch_sparse"
echo "Backbone:          DiT adaLN-Zero h128 L10 H4"
echo "Coordinates:       centered_bbox absolute embedding"
echo "Path:              x_t=(1-t)*x_source+t*x_data"
echo "Target:            velocity=x_data-x_source"
echo "Time input:        1000*t"
echo "Batch size:        128"
echo "Epochs:            1000"
echo "Seed:              42"
echo "Control:           M33/M34 no-self 2-D DiNAT-DiT"
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

echo
echo "===== Self-attention correctness gate ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -m pytest \
        tests/unit/test_dinat_dit.py::test_dinat_dit_self_attention_augments_local_and_dilated_topologies \
        tests/unit/test_dinat_dit.py::test_dinat_dit_factory_matches_full_dit_shared_parameters \
        tests/unit/test_dinat_dit.py::test_dinat_dit_torch_sparse_backend_matches_scatter \
        tests/unit/test_dinat_dit.py::test_dinat_dit_torch_sparse_backend_supports_inference_mode_tensors \
        tests/unit/test_dinat_dit.py::test_dinat_dit_torch_sparse_cache_survives_inference_then_training \
        -q

nvidia-smi

singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.train_generative \
        data=hit_slice_pt \
        geometry=dinat_exact2 \
        model=dinat_dit \
        task=hit_flow_matching \
        optimizer=adamw \
        +generative=hit_slice_flow_matching \
        seed=42 \
        run_name="$RUN_NAME" \
        task.time_embedding_scale=1000.0 \
        model.hidden_dim=128 \
        model.num_heads=4 \
        model.num_layers=10 \
        model.spatial_dim=2 \
        model.mlp_ratio=4 \
        model.condition_embed_dim=128 \
        model.dropout=0.0 \
        model.qkv_bias=true \
        model.out_proj_bias=false \
        model.sparse_attention_backend=torch_sparse \
        model.include_self_attention=true \
        generative.batch_size=128 \
        generative.num_workers=4 \
        generative.max_epochs=1000 \
        generative.device=cuda

for artifact in best.pt last.pt history.csv summary.json resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
    test -s "${RUN_DIR}/${artifact}" || {
        echo "ERROR: missing training artifact: ${RUN_DIR}/${artifact}"
        exit 1
    }
done

echo
echo "===== M39 summary ====="
cat "${RUN_DIR}/summary.json"
