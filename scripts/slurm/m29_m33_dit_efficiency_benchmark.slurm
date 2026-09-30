#!/bin/bash
#SBATCH --job-name=m29_m33_eff
#SBATCH --time=01:30:00
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

M29_RUN=${RUN_ROOT}/m29_full_dit_flow_tscale1000_1000e_h128_l10_h4
M33_RUN=${RUN_ROOT}/m33_dinat_dit_flow_tscale1000_1000e_h128_l10_h4
OUTPUT_DIR=${RUN_ROOT}/efficiency_m29_vs_m33_scatter_vs_torch_sparse_flow

cd "$REPO"

echo "============================================================"
echo "M29/M33 controlled Flow-Matching efficiency benchmark"
echo "============================================================"
echo "M29: Full DiT"
echo "M33-A: Local/exact-two-hop DiT, scatter attention"
echo "M33-B: Local/exact-two-hop DiT, native PyTorch sparse attention"
echo "GPU request: 1"
echo "Batch size: 128"
echo "Flow sampler: Heun, 50 steps"
echo "Memory: absolute + incremental CUDA peak allocated"
echo "============================================================"

echo
echo "===== Git ====="
git rev-parse HEAD
git status --short
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

echo
echo "===== Run artifacts ====="
for run_dir in "$M29_RUN" "$M33_RUN"; do
    for artifact in \
        best.pt \
        resolved_config.yaml \
        summary.json \
        standardizers.pt \
        dataset_split_manifest.json
    do
        test -f "${run_dir}/${artifact}" || {
            echo "ERROR: missing artifact: ${run_dir}/${artifact}"
            exit 1
        }
    done
done

echo
echo "===== Native sparse correctness test on allocated GPU ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -m pytest \
        tests/unit/test_dinat_dit.py::test_dinat_dit_torch_sparse_backend_matches_scatter \
        -q

echo
echo "===== GPU ====="
nvidia-smi
nvidia-smi \
    --query-gpu=name,uuid,memory.total,driver_version,pstate,clocks.sm,clocks.mem \
    --format=csv

echo
echo "===== Three-way efficiency benchmark ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.benchmark_dit_efficiency \
        --run "full_dit=$M29_RUN" \
        --run "local_scatter=$M33_RUN" \
        --run "local_torch_sparse=$M33_RUN" \
        --attention-backend "local_scatter=scatter" \
        --attention-backend "local_torch_sparse=torch_sparse" \
        --output-dir "$OUTPUT_DIR" \
        --checkpoint best.pt \
        --device cuda \
        --batch-size 128 \
        --num-workers 0 \
        --forward-warmup 10 \
        --forward-steps 50 \
        --train-warmup 5 \
        --train-steps 20 \
        --generation-warmup 1 \
        --generation-repeats 3 \
        --sampling-steps 50 \
        --sampling-seed 5678 \
        --overwrite

echo
echo "===== Output checks ====="
for artifact in \
    efficiency_summary.json \
    efficiency_results.csv \
    efficiency_comparison.csv
do
    test -s "${OUTPUT_DIR}/${artifact}"
done

echo
echo "===== Compact results ====="
cat "${OUTPUT_DIR}/efficiency_results.csv"

echo
echo "===== Relative to Full DiT ====="
cat "${OUTPUT_DIR}/efficiency_comparison.csv"

echo
echo "============================================================"
echo "M29 / M33 SCATTER / M33 TORCH-SPARSE BENCHMARK COMPLETE"
echo "Output directory:"
echo "  $OUTPUT_DIR"
echo "============================================================"
