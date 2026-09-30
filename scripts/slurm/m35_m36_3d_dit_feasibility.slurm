#!/bin/bash
#SBATCH --job-name=m35_m36_3d_feas
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
OUT=/scratch/coop/theret/GraphAttention_runs/hit_3d_dit_feasibility/summary.json

cd "$REPO"

echo "============================================================"
echo "3D HIT feasibility - Full DiT vs optimized NAT-DiT"
echo "============================================================"
git rev-parse HEAD
git status --short
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

nvidia-smi

echo
echo "===== Focused implementation tests ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -m pytest \
        tests/unit/test_avbp_hdf5.py::test_avbp_directory_reader_discovers_fixed_mesh_snapshots \
        tests/unit/test_generative_data_pipeline.py::test_collator_reuses_fixed_mesh_connectivity_and_attention_topology \
        tests/unit/test_generative_training_adapters.py::test_gradient_accumulation_matches_one_equal_sample_large_batch_step \
        tests/unit/test_dinat_dit.py::test_dinat_dit_torch_sparse_backend_matches_scatter \
        -q

echo
echo "===== Real 3D volume benchmark ====="
rm -rf "$(dirname "$OUT")"
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.benchmark_3d_dit_feasibility \
        --output "$OUT"

test -s "$OUT"
cat "$OUT"
