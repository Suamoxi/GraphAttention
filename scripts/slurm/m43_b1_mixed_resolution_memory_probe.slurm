#!/bin/bash
#SBATCH --job-name=m43_b1_train_probe
#SBATCH --time=04:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# M43-B1: one real 65^3 and one real 129^3 M37 training-memory probe.
# Two independent subprocesses, two full AdamW steps each. NOT a train run.
# No --mem cap and no changes to DNS data, graph or M37 math.
set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
DATA_ROOT=${M43_B1_DATA_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
OUT=${M43_B1_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m43_b1_training_memory_gate}
STEPS=${M43_B1_STEPS:-2}

cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: checkout must be clean for reproducibility" >&2
    git status --short >&2
    exit 1
fi
test -d "$DATA_ROOT" || { echo "Dataset root not available: $DATA_ROOT" >&2; exit 1; }
test -s "$DATA_ROOT/Re54/Ma0.3/MESH/mesh.mesh.h5"
test -s "$DATA_ROOT/Re94/Ma0.3/MESH/mesh.mesh.h5"

echo "==== M43-B1 actual full-graph training memory feasibility ===="
echo "Host: $(hostname)"
echo "Code: $(git rev-parse HEAD)"
echo "Cases: Re54/Ma0.3 (65^3), Re94/Ma0.3 (129^3)"
echo "Training steps per case: $STEPS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "==== Syntax preflight ===="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
    scripts/m43_b1_memory_gate.py \
    scripts/m43_b1_training_step.py \
    tests/unit/test_m43_b1_training_memory.py

echo "==== Small HDF5 and M37 forward/backward regression ===="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q \
    tests/unit/test_m43_b1_training_memory.py \
    tests/unit/test_avbp_hdf5.py::test_avbp_reader_supports_multiple_explicit_meshes \
    tests/unit/test_generative_data_pipeline.py::test_collator_reuses_fixed_mesh_connectivity_and_attention_topology \
    tests/unit/test_dinat_dit.py::test_dinat_dit_torch_sparse_backend_matches_scatter

echo "==== Run separate real HDF5 65^3 and 129^3 training-step probes ===="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m43_b1_memory_gate \
        --root "$DATA_ROOT" \
        --output-dir "$OUT" \
        --case Re54/Ma0.3:65 \
        --case Re94/Ma0.3:129 \
        --steps "$STEPS" \
        --seed 42 \
        --hidden-dim 128 \
        --layers 10 \
        --heads 4

echo "==== M43-B1 success: $OUT/summary.json ===="
