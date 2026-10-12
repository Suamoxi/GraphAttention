#!/bin/bash
#SBATCH --job-name=m41_2_owned_query
#SBATCH --time=04:00:00
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
OUT=${OUT:-${RUN_ROOT}/m41_2_owned_query_m37/${SLURM_JOB_ID:-manual}}
LAYOUTS=${M41_2_BLOCKS:-"1 1 1;2 1 1;2 2 1;2 2 2;4 2 2"}

cd "$REPO"
mkdir -p "$OUT"
for artifact in "$CHECKPOINT" resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
    test -s "$M37_RUN/$artifact" || {
        echo "ERROR: M37 artifact missing: $M37_RUN/$artifact" >&2
        exit 1
    }
done
git rev-parse HEAD > "$OUT/git_commit.txt"
git status --short > "$OUT/git_status.txt"
if [ -s "$OUT/git_status.txt" ]; then
    echo "ERROR: tracked/untracked Git changes; benchmark requires a clean checkout" >&2
    cat "$OUT/git_status.txt"
    exit 1
fi
echo "M41.2: full-block M41.1 vs owned-query on one GH200"
echo "Commit: $(git rev-parse HEAD)"
echo "M37 run: $M37_RUN, checkpoint: $CHECKPOINT"
echo "Layouts: $LAYOUTS"
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpu_info.csv"

echo "==== Unit-test gate (owned-query, M41.1, M37 and Flow Matching) ===="
FAILED=0
if singularity exec --nv -B /home -B /scratch "$IMAGE" \
    "$PYTHON" -m pytest tests/unit/test_owned_query_dinat.py \
    tests/unit/test_m41_2_summary.py \
    tests/unit/test_partitioned_dinat.py tests/unit/test_m41_summary.py \
    tests/unit/test_dinat_dit.py tests/unit/test_flow_matching_task.py -q; then
    echo "M41.2 test gate passed"
else
    FAILED=1
    if [ "${M41_2_CONTINUE_AFTER_FAILURE:-0}" != "1" ]; then
        echo "M41.2 test gate failed; set M41_2_CONTINUE_AFTER_FAILURE=1 for diagnostics" >&2
        exit 1
    fi
    echo "DIAGNOSTIC MODE: test gate failed; job will exit nonzero" >&2
fi

IFS=';' read -ra PARTITION_LAYOUTS <<< "$LAYOUTS"
for layout in "${PARTITION_LAYOUTS[@]}"; do
    read -r BX BY BZ <<< "$layout"
    for strategy in layerwise_full owned_query; do
        if [ "$strategy" = "layerwise_full" ]; then
            label=full
        else
            label=owned
        fi
        report="$OUT/${label}_blocks_${BX}x${BY}x${BZ}.json"
        log="$OUT/${label}_blocks_${BX}x${BY}x${BZ}.log"
        profile_args=()
        if [ "${M41_2_PROFILE:-0}" = "1" ]; then
            profile_args=(--profile-dir "$OUT/profile_${label}_${BX}x${BY}x${BZ}")
            if [ "${M41_2_TRACE:-0}" = "1" ]; then
                profile_args+=(--profile-chrome-trace)
            fi
        fi
        echo "==== $label, blocks ${BX}x${BY}x${BZ} ===="
        if singularity exec --nv -B /home -B /scratch "$IMAGE" \
            "$PYTHON" -u -m scripts.benchmark_m41_partitioned \
            --executor "$strategy" --run-dir "$M37_RUN" --checkpoint "$CHECKPOINT" \
            --blocks "$BX" "$BY" "$BZ" --sampling-steps 50 \
            --warmup "${M41_2_WARMUP:-2}" --repeats "${M41_2_REPEATS:-5}" \
            "${profile_args[@]}" --output "$report" > "$log" 2>&1; then
            echo "$label layout ${BX}x${BY}x${BZ}: completed"
        else
            FAILED=1
            echo "ERROR: $label layout ${BX}x${BY}x${BZ} failed; see $log" >&2
            tail -n 35 "$log" || true
            if [ "${M41_2_CONTINUE_AFTER_FAILURE:-0}" != "1" ]; then
                exit 1
            fi
        fi
    done
done

if singularity exec -B /home -B /scratch "$IMAGE" \
    "$PYTHON" -m scripts.summarize_m41_2 --input-dir "$OUT"; then
    echo "Comparison generated"
else
    FAILED=1
    echo "M41.2 comparison incomplete or numerical gate failed" >&2
fi
echo "Results: $OUT"
exit "$FAILED"
