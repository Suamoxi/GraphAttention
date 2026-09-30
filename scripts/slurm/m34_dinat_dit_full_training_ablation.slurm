#!/bin/bash
#SBATCH --job-name=m34_dinat_fulltrain
#SBATCH --time=10:00:00
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

SCATTER_NAME=m34_dinat_dit_flow_scatter_control_1000e_h128_l10_h4
OPTIMIZED_NAME=m34_dinat_dit_flow_optimized_1000e_h128_l10_h4
SCATTER_RUN=${RUN_ROOT}/${SCATTER_NAME}
OPTIMIZED_RUN=${RUN_ROOT}/${OPTIMIZED_NAME}
COMPARE_DIR=${RUN_ROOT}/m34_dinat_dit_full_training_comparison

cd "$REPO"

echo "============================================================"
echo "M34 - full-training attention-backend ablation"
echo "============================================================"
echo "Architecture:   DiNAT-DiT h128 L10 H4"
echo "Task:           Flow Matching, time scale 1000"
echo "Topology:       alternating local / exact-two-hop"
echo "Batch size:     128"
echo "Epochs:         1000"
echo "Seed:           42"
echo "Variant A:      scatter"
echo "Variant B:      optimized CSR-SDDMM + chunked aggregation"
echo "GPU request:    1"
echo "============================================================"

echo
echo "===== Git ====="
git rev-parse HEAD
git status --short
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

for path in "$SCATTER_RUN" "$OPTIMIZED_RUN" "$COMPARE_DIR"; do
    if [ -e "$path" ]; then
        echo "ERROR: output path already exists: $path"
        exit 1
    fi
done

echo
echo "===== GPU ====="
nvidia-smi
nvidia-smi \
    --query-gpu=name,uuid,memory.total,driver_version,pstate,clocks.sm,clocks.mem \
    --format=csv

echo
echo "===== Optimized backend correctness gate ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -m pytest \
        tests/unit/test_dinat_dit.py::test_torch_sparse_sddmm_matches_edge_gather \
        tests/unit/test_dinat_dit.py::test_torch_sparse_attention_softmax_matches_scatter \
        tests/unit/test_dinat_dit.py::test_torch_sparse_spmm_matches_index_add \
        tests/unit/test_dinat_dit.py::test_dinat_dit_torch_sparse_backend_matches_scatter \
        -q

train_variant() {
    run_name="$1"
    backend="$2"

    echo
    echo "============================================================"
    echo "TRAINING: $run_name"
    echo "attention backend: $backend"
    echo "============================================================"

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
            run_name="$run_name" \
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
            model.sparse_attention_backend="$backend" \
            generative.batch_size=128 \
            generative.num_workers=4 \
            generative.max_epochs=1000 \
            generative.device=cuda

    run_dir=${RUN_ROOT}/${run_name}
    for artifact in \
        best.pt \
        last.pt \
        history.csv \
        summary.json \
        resolved_config.yaml \
        standardizers.pt \
        dataset_split_manifest.json
    do
        test -s "${run_dir}/${artifact}" || {
            echo "ERROR: missing training artifact: ${run_dir}/${artifact}"
            exit 1
        }
    done

    echo
    echo "===== Summary: $run_name ====="
    cat "${run_dir}/summary.json"
}

train_variant "$SCATTER_NAME" scatter
train_variant "$OPTIMIZED_NAME" torch_sparse

echo
echo "===== Controlled full-training comparison ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.compare_full_training \
        --scatter-run "$SCATTER_RUN" \
        --optimized-run "$OPTIMIZED_RUN" \
        --output-dir "$COMPARE_DIR"

for artifact in full_training_comparison.json epoch_comparison.csv; do
    test -s "${COMPARE_DIR}/${artifact}" || {
        echo "ERROR: missing comparison artifact: ${COMPARE_DIR}/${artifact}"
        exit 1
    }
done

echo
echo "============================================================"
echo "M34 FULL-TRAINING ABLATION COMPLETE"
echo "============================================================"
echo "Scatter:"
echo "  $SCATTER_RUN"
echo "Optimized:"
echo "  $OPTIMIZED_RUN"
echo "Comparison:"
echo "  $COMPARE_DIR/full_training_comparison.json"
echo "  $COMPARE_DIR/epoch_comparison.csv"
echo "============================================================"
