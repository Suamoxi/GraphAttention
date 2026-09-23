#!/bin/bash
#SBATCH --job-name=m33_dgl_eff
#SBATCH --time=01:00:00
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

M33_RUN=${RUN_ROOT}/m33_dinat_dit_flow_tscale1000_1000e_h128_l10_h4
OUTPUT_DIR=${RUN_ROOT}/efficiency_m33_scatter_vs_dgl_flow

cd "$REPO"

echo "============================================================"
echo "M33 sparse attention backend benchmark"
echo "============================================================"
echo "Same checkpoint: scatter vs DGL sparse SDDMM/softmax/SPMM"
echo "GPU request: 1"
echo "Batch size: 128"
echo "Flow sampler: Heun, 50 steps"
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
echo "===== DGL sparse compatibility ====="
if ! singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -c 'import torch; import dgl; import dgl.sparse as dglsp; print("torch", torch.__version__); print("cuda", torch.version.cuda); print("dgl", dgl.__version__); print("dgl.sparse", dglsp.__name__)'
then
    echo
    echo "ERROR: DGL sparse is not importable in the current CERFACS environment."
    echo "The model code is unchanged for the default scatter backend."
    echo "Send this job error/output back before installing or changing the environment."
    exit 2
fi

echo
echo "===== Run artifacts ====="
for artifact in \
    best.pt \
    resolved_config.yaml \
    summary.json \
    standardizers.pt \
    dataset_split_manifest.json
do
    test -f "${M33_RUN}/${artifact}" || {
        echo "ERROR: missing artifact: ${M33_RUN}/${artifact}"
        exit 1
    }
done

echo
echo "===== GPU ====="
nvidia-smi
nvidia-smi \
    --query-gpu=name,uuid,memory.total,driver_version,pstate,clocks.sm,clocks.mem \
    --format=csv

echo
echo "===== Scatter vs DGL efficiency benchmark ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.benchmark_dit_efficiency \
        --run "local_scatter=$M33_RUN" \
        --run "local_dgl=$M33_RUN" \
        --attention-backend "local_scatter=scatter" \
        --attention-backend "local_dgl=dgl" \
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
echo "===== Compact results ====="
cat "${OUTPUT_DIR}/efficiency_results.csv"

echo
echo "===== DGL relative to scatter ====="
cat "${OUTPUT_DIR}/efficiency_comparison.csv"

echo
echo "============================================================"
echo "M33 SCATTER VS DGL BENCHMARK COMPLETE"
echo "============================================================"
echo "Output directory:"
echo "  $OUTPUT_DIR"
echo "============================================================"
