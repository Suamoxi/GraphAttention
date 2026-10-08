#!/bin/bash
#SBATCH --job-name=m37_m38_3d_volume
#SBATCH --time=02:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
RUN_ROOT=/scratch/coop/theret/GraphAttention_runs

M37_RUN=${M37_RUN:-${RUN_ROOT}/m37_3d_dinat_dit_flow_1000e_h128_l10_h4_b16}
M38_RUN=${M38_RUN:-${RUN_ROOT}/m38_3d_full_dit_flow_1000e_h128_l10_h4_b18}
CHECKPOINT_KIND=${CHECKPOINT_KIND:-best}
OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/m37_m38_3d_volume_comparison}
NUM_EXAMPLES=${NUM_EXAMPLES:-3}
RESOLUTION=${RESOLUTION:-220}
RAY_STEPS=${RAY_STEPS:-84}
OPTICAL_DEPTH=${OPTICAL_DEPTH:-2.0}

cd "$REPO"

latest_generation () {
    local run_dir="$1"
    local found
    local legacy
    found=$(find "${run_dir}/generations" -maxdepth 1 -type d \
        -name "flow_heun_steps50_seed5678_${CHECKPOINT_KIND}_epoch*" \
        -print 2>/dev/null | sort -V | tail -n 1)
    if [ -n "$found" ]; then
        printf '%s\n' "$found"
        return 0
    fi

    legacy="${run_dir}/generations/flow_heun_steps50_seed5678"
    if [ -s "${legacy}/generated_test.pt" ] && [ -s "${legacy}/summary.json" ]; then
        if singularity exec -B /home -B /scratch "$IMAGE" "$PYTHON" -c \
            "import json,pathlib,sys; s=json.loads(pathlib.Path('${legacy}/summary.json').read_text()); sys.exit(0 if s.get('source_checkpoint') == '${CHECKPOINT_KIND}.pt' else 1)"
        then
            printf '%s\n' "$legacy"
            return 0
        fi
    fi
    echo "ERROR: no completed ${CHECKPOINT_KIND} generation found in ${run_dir}" >&2
    return 1
}

M37_GEN=$(latest_generation "$M37_RUN")
M38_GEN=$(latest_generation "$M38_RUN")

echo "============================================================"
echo "M37 DiNAT-DiT versus M38 Full DiT - true 3D volume rendering"
echo "============================================================"
echo "M37: $M37_GEN"
echo "M38: $M38_GEN"
echo "Full scalar ray casting, not slicing; colormap RdBu_r"
echo "Shared scalar limits and transfer function for DNS/M37/M38"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

git rev-parse HEAD
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked Git files are modified"
    exit 1
fi

for gen_dir in "$M37_GEN" "$M38_GEN"; do
    for artifact in generated_test.pt summary.json resolved_config.yaml; do
        test -s "${gen_dir}/${artifact}" || {
            echo "ERROR: missing generation artifact ${gen_dir}/${artifact}" >&2
            exit 1
        }
    done
done

echo "===== True-volume rendering regression tests ====="
singularity exec -B /home -B /scratch "$IMAGE" \
    "$PYTHON" -m pytest tests/unit/test_volume_rendering_3d.py -q

echo "===== Render five fields, three unpaired examples ====="
singularity exec -B /home -B /scratch "$IMAGE" \
    "$PYTHON" -u -m scripts.plot_m37_m38_3d_volumes \
        --m37-generation "$M37_GEN" \
        --m38-generation "$M38_GEN" \
        --output-dir "$OUTPUT_DIR" \
        --num-examples "$NUM_EXAMPLES" \
        --resolution "$RESOLUTION" \
        --steps "$RAY_STEPS" \
        --optical-depth "$OPTICAL_DEPTH" \
        --cmap RdBu_r

for name in rho_value rhou_x rhov_y rhow_z rhoE_value; do
    test -s "${OUTPUT_DIR}/example_000/${name}.png" || {
        echo "ERROR: missing rendered volume $name" >&2
        exit 1
    }
done

echo "M37/M38 VOLUME RENDERING COMPLETE: $OUTPUT_DIR"
