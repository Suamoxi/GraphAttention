#!/bin/bash
#SBATCH --job-name=m40_3d_nat_self
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

# Keep the same 3-D training microbatch as M37 for a controlled topology-only
# ablation. Reject this configuration before training if M40 does not fit.
BATCH=16
PROBE_JSON=${RUN_ROOT}/m40_3d_dinat_local_self_batch16_probe.json
RUN_NAME=m40_3d_dinat_local_self_flow_1000e_h128_l10_h4_b16
RUN_DIR=${RUN_ROOT}/${RUN_NAME}

cd "$REPO"

echo "============================================================"
echo "M40 - 3D NAT self-attention only vs M37, 1000 epochs"
echo "============================================================"
echo "Local/even layers: self + graph one-hop"
echo "Dilated/odd layers: exact-two-hop, NO self"
echo "Backend:          torch_sparse (CSR SDDMM + chunked aggregation)"
echo "Architecture:     DiT adaLN-Zero, h128 L10 H4"
echo "Batch:            16 physical volumes (same as M37)"
echo "Gradient accum:   1"
echo "Task:             3D HIT Flow Matching, time scale 1000"
echo "Epochs:           1000"
echo "Seed:             42"
echo "Control:          M37 3-D DiNAT-DiT without self"
echo "Run:              $RUN_DIR"
echo "============================================================"

git rev-parse HEAD
git status --short
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files have uncommitted changes."
    exit 1
fi

echo
echo "===== Topology, initialization, and optimized-backend correctness ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -m pytest \
        tests/unit/test_dinat_dit.py::test_dinat_dit_3d_self_attention_local_only \
        tests/unit/test_dinat_dit.py::test_dinat_dit_3d_local_only_factory_metadata \
        tests/unit/test_dinat_dit.py::test_dinat_dit_self_attention_augments_local_and_dilated_topologies \
        tests/unit/test_dinat_dit.py::test_dinat_dit_torch_sparse_backend_matches_scatter \
        tests/unit/test_dinat_dit.py::test_dinat_dit_torch_sparse_backend_supports_inference_mode_tensors \
        tests/unit/test_dinat_dit.py::test_dinat_dit_torch_sparse_cache_survives_inference_then_training \
        -q

nvidia-smi

echo
echo "===== M40 CUDA safety probe at M37's batch size ====="
if [ ! -s "$PROBE_JSON" ]; then
    singularity exec --nv \
        -B /home \
        -B /scratch \
        "$IMAGE" \
        "$PYTHON" -u scripts/probe_3d_max_batch.py \
            --model dinat \
            --include-local-self-attention \
            --max-batch "$BATCH" \
            --output "$PROBE_JSON"
fi

singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -c "import json,sys; p=json.load(open('$PROBE_JSON')); assert p['model']=='dinat' and p['include_local_self_attention'] is True and p['max_passing_batch'] >= $BATCH, 'M40 batch 16 failed; do not change the batch silently'"

echo
echo "===== M40 training (fresh or resume M40 only) ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.train_generative \
        data=hit_3d_avbp \
        geometry=dinat_exact2 \
        model=dinat_dit \
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
        model.include_self_attention=false \
        +model.include_local_self_attention=true \
        model.sparse_attention_backend=torch_sparse \
        generative.batch_size="$BATCH" \
        generative.gradient_accumulation_steps=1 \
        generative.resume=true \
        generative.num_workers=4 \
        generative.max_epochs=1000 \
        generative.device=cuda

for artifact in best.pt last.pt history.csv summary.json resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
    test -s "${RUN_DIR}/${artifact}"
done

echo
echo "===== M40 final summary ====="
cat "${RUN_DIR}/summary.json"
