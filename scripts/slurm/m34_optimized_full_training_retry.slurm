#!/bin/bash
#SBATCH --job-name=m34_opt_retry
#SBATCH --time=05:00:00
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

SCATTER_RUN=${RUN_ROOT}/m34_dinat_dit_flow_scatter_control_1000e_h128_l10_h4
OPTIMIZED_NAME=m34_dinat_dit_flow_optimized_retry_1000e_h128_l10_h4
OPTIMIZED_RUN=${RUN_ROOT}/${OPTIMIZED_NAME}
COMPARE_DIR=${RUN_ROOT}/m34_dinat_dit_full_training_comparison_retry

cd "$REPO"

echo "============================================================"
echo "M34 optimized backend retry"
echo "============================================================"
echo "Control:        existing completed scatter run"
echo "Variant:        optimized CSR-SDDMM + chunked aggregation"
echo "Architecture:   DiNAT-DiT h128 L10 H4"
echo "Task:           Flow Matching, time scale 1000"
echo "Batch size:     128"
echo "Epochs:         1000"
echo "Seed:           42"
echo "============================================================"

git rev-parse HEAD
git status --short
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

for artifact in best.pt last.pt history.csv summary.json resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
    test -s "${SCATTER_RUN}/${artifact}" || {
        echo "ERROR: missing scatter control artifact: ${SCATTER_RUN}/${artifact}"
        exit 1
    }
done

for path in "$OPTIMIZED_RUN" "$COMPARE_DIR"; do
    if [ -e "$path" ]; then
        echo "ERROR: output path already exists: $path"
        exit 1
    fi
done

nvidia-smi

echo
echo "===== Optimized backend correctness + validation-mode gate ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -m pytest \
        tests/unit/test_dinat_dit.py::test_torch_sparse_sddmm_matches_edge_gather \
        tests/unit/test_dinat_dit.py::test_torch_sparse_attention_softmax_matches_scatter \
        tests/unit/test_dinat_dit.py::test_torch_sparse_spmm_matches_index_add \
        tests/unit/test_dinat_dit.py::test_dinat_dit_torch_sparse_backend_matches_scatter \
        tests/unit/test_dinat_dit.py::test_dinat_dit_torch_sparse_backend_supports_inference_mode_tensors \
        -q

echo
echo "===== Training optimized M34 retry ====="
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
        run_name="$OPTIMIZED_NAME" \
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
        model.include_self_attention=false \
        model.sparse_attention_backend=torch_sparse \
        generative.batch_size=128 \
        generative.num_workers=4 \
        generative.max_epochs=1000 \
        generative.device=cuda

for artifact in best.pt last.pt history.csv summary.json resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
    test -s "${OPTIMIZED_RUN}/${artifact}" || {
        echo "ERROR: missing optimized retry artifact: ${OPTIMIZED_RUN}/${artifact}"
        exit 1
    }
done

echo
echo "===== Compare against completed scatter control ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.compare_full_training \
        --scatter-run "$SCATTER_RUN" \
        --optimized-run "$OPTIMIZED_RUN" \
        --output-dir "$COMPARE_DIR"

cat "${OPTIMIZED_RUN}/summary.json"
cat "${COMPARE_DIR}/full_training_comparison.json"
