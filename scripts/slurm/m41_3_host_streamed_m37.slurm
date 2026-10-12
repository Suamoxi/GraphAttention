#!/bin/bash
#SBATCH --job-name=m41_3_streamed
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
OUT=${OUT:-${RUN_ROOT}/m41_3_host_streamed_m37/${SLURM_JOB_ID:-manual}}
LAYOUTS=${M41_3_BLOCKS:-"1 1 1;2 1 1;2 2 1;2 2 2;4 2 2"}

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
    echo "ERROR: checkout must be clean for a traceable M41.3 benchmark" >&2
    cat "$OUT/git_status.txt"
    exit 1
fi
echo "M41.3: single-GPU CPU-streamed M37 versus M41.2 owned-query"
echo "Commit: $(git rev-parse HEAD)"
echo "M37 run: $M37_RUN"
echo "Layouts: $LAYOUTS"
echo "Pin host hidden buffers: ${M41_3_PIN_HOST:-0}"
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpu_info.csv"

FAILED=0
echo "==== M41.3/M41.2/M41.1/M37 regression gate ===="
if singularity exec --nv -B /home -B /scratch "$IMAGE" \
    "$PYTHON" -m pytest \
    tests/unit/test_streamed_dinat.py tests/unit/test_m41_3_summary.py \
    tests/unit/test_owned_query_dinat.py tests/unit/test_m41_2_summary.py \
    tests/unit/test_partitioned_dinat.py tests/unit/test_m41_summary.py \
    tests/unit/test_dinat_dit.py tests/unit/test_flow_matching_task.py -q; then
    echo "Selected regression gate passed"
else
    FAILED=1
    if [ "${M41_3_CONTINUE_AFTER_FAILURE:-0}" != "1" ]; then
        echo "ERROR: test gate failed, benchmark not started" >&2
        exit 1
    fi
    echo "Diagnostic continuation enabled; overall exit will remain nonzero"
fi

pin_args=()
if [ "${M41_3_PIN_HOST:-0}" = "1" ]; then
    pin_args=(--pin-host-hidden)
fi
IFS=';' read -ra PARTITION_LAYOUTS <<< "$LAYOUTS"
for layout in "${PARTITION_LAYOUTS[@]}"; do
    read -r BX BY BZ <<< "$layout"
    for strategy in owned_query streamed; do
        if [ "$strategy" = "streamed" ]; then
            label=streamed
            extra_args=("${pin_args[@]}")
        else
            label=owned
            extra_args=()
        fi
        report="$OUT/${label}_blocks_${BX}x${BY}x${BZ}.json"
        log="$OUT/${label}_blocks_${BX}x${BY}x${BZ}.log"
        echo "==== $strategy: blocks ${BX}x${BY}x${BZ} ===="
        if singularity exec --nv -B /home -B /scratch "$IMAGE" \
            "$PYTHON" -u -m scripts.benchmark_m41_partitioned \
            --executor "$strategy" \
            --run-dir "$M37_RUN" --checkpoint "$CHECKPOINT" \
            --blocks "$BX" "$BY" "$BZ" --sampling-steps 50 \
            --warmup "${M41_3_WARMUP:-1}" --repeats "${M41_3_REPEATS:-2}" \
            "${extra_args[@]}" --output "$report" > "$log" 2>&1; then
            echo "$strategy ${BX}x${BY}x${BZ}: completed"
        else
            FAILED=1
            echo "ERROR: $strategy ${BX}x${BY}x${BZ} failed; see $log" >&2
            tail -n 40 "$log" || true
            if [ "${M41_3_CONTINUE_AFTER_FAILURE:-0}" != "1" ]; then
                exit 1
            fi
        fi
    done
done

if singularity exec -B /home -B /scratch "$IMAGE" \
    "$PYTHON" -m scripts.summarize_m41_3 --input-dir "$OUT"; then
    echo "Paired comparison generated"
else
    FAILED=1
    echo "ERROR: incomplete comparison or failed M41.3 numerical gate" >&2
fi
echo "Results: $OUT"
exit "$FAILED"
