#!/bin/bash
#SBATCH --job-name=m41_1_halo
#SBATCH --time=03:00:00
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
M37_RUN=${M37_RUN:-${RUN_ROOT}/m37_3d_dinat_dit_flow_1000e_h128_l10_h4_b16}
CHECKPOINT=${CHECKPOINT:-best.pt}
OUT=${OUT:-${RUN_ROOT}/m41_1_halo_partitioned_m37/${SLURM_JOB_ID:-manual}}
LAYOUTS=${M41_BLOCKS:-"1 1 1;2 1 1;2 2 1;2 2 2;4 2 2"}

cd "$REPO"
mkdir -p "$OUT"
for artifact in "$CHECKPOINT" resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
    test -s "$M37_RUN/$artifact" || {
        echo "ERROR: missing M37 artifact: $M37_RUN/$artifact" >&2
        exit 1
    }
done
git rev-parse HEAD > "$OUT/git_commit.txt"
echo "M41.1 exact single-GPU halo inference"
echo "Repository commit: $(git rev-parse HEAD)"
echo "M37 run: $M37_RUN"
echo "Checkpoint: $CHECKPOINT"
echo "Partition layouts: $LAYOUTS"
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpu_info.csv"

echo "==== Partitioned DiNAT test gate ===="
GATE_FAILED=0
if singularity exec --nv -B /home -B /scratch "$IMAGE" \
    "$PYTHON" -m pytest tests/unit/test_partitioned_dinat.py tests/unit/test_m41_summary.py \
    tests/unit/test_dinat_dit.py tests/unit/test_flow_matching_task.py -q; then
    echo "M41.1 test gate passed"
else
    GATE_FAILED=1
    if [ "${M41_CONTINUE_AFTER_FAILURE:-0}" != "1" ]; then
        echo "M41.1 test gate failed. To collect diagnostic checkpoint reports, set M41_CONTINUE_AFTER_FAILURE=1." >&2
        exit 1
    fi
    echo "DIAGNOSTIC MODE: proceeding despite test failure; job will exit nonzero" >&2
fi

echo "Real M37 checkpoint: per-layer / 50-step Heun parity and performance"
IFS=';' read -ra PARTITION_LAYOUTS <<< "$LAYOUTS"
BENCH_FAILED=0
for layout in "${PARTITION_LAYOUTS[@]}"; do
    read -r BX BY BZ <<< "$layout"
    if singularity exec --nv -B /home -B /scratch "$IMAGE" \
        "$PYTHON" -u -m scripts.benchmark_m41_partitioned \
        --run-dir "$M37_RUN" --checkpoint "$CHECKPOINT" \
        --blocks "$BX" "$BY" "$BZ" \
        --warmup "${M41_WARMUP:-2}" --repeats "${M41_REPEATS:-5}" --sampling-steps 50 \
        --output "$OUT/blocks_${BX}x${BY}x${BZ}.json"; then
        echo "M41.1 layout ${BX}x${BY}x${BZ} passed"
    else
        BENCH_FAILED=1
        if [ "${M41_CONTINUE_AFTER_FAILURE:-0}" != "1" ]; then
            echo "M41.1 layout failed: ${BX}x${BY}x${BZ}" >&2
            exit 1
        fi
        echo "DIAGNOSTIC MODE: layout ${BX}x${BY}x${BZ} failed" >&2
    fi
done

SUMMARY_FAILED=0
if singularity exec -B /home -B /scratch "$IMAGE" \
    "$PYTHON" -u -m scripts.summarize_m41_partitioned \
    --input-dir "$OUT"; then
    echo "M41.1 summary passed"
else
    SUMMARY_FAILED=1
    echo "M41.1 comparison is incomplete or failed its numerical gate" >&2
fi
echo "M41.1 validation reports and comparison.csv (if available): $OUT"
if [ "$GATE_FAILED" -ne 0 ] || [ "$BENCH_FAILED" -ne 0 ] || [ "$SUMMARY_FAILED" -ne 0 ]; then
    exit 1
fi
