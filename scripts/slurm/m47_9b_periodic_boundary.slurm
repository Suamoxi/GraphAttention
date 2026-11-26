#!/bin/bash
#SBATCH --job-name=m47_9b_boundary
#SBATCH --partition=grace
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err

# Read-only CPU postprocessing of previously generated 3-D HIT fields.
# No Python execution on the Calypso login nodes.
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M47_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
RUNS=${M47_RUNS_ROOT:-/scratch/coop/theret/GraphAttention_runs}
M37=${M47_9B_M37_GEN:-$RUNS/m37_3d_dinat_dit_flow_1000e_h128_l10_h4_b16/generations/flow_heun_steps50_seed5678}
M38=${M47_9B_M38_GEN:-$RUNS/m38_3d_full_dit_flow_1000e_h128_l10_h4_b18/generations/flow_heun_steps50_seed5678_best_epoch108}
OUT_ROOT=${M47_9B_OUT_ROOT:-$RUNS/m47_9b_periodic_boundary}
OUT=$OUT_ROOT/job_${SLURM_JOB_ID}

cd "$REPO"
test -z "$(git status --porcelain)" || { echo 'ERROR: dirty checkout' >&2; exit 2; }
test -s "$IMAGE" && test -x "$PYTHON"
test ! -e "$OUT" || { echo "ERROR: existing output $OUT" >&2; exit 2; }

for path in "$M37" "$M38"; do
    for f in generated_test.pt summary.json resolved_config.yaml; do
        test -s "$path/$f" || { echo "ERROR: missing $path/$f" >&2; exit 2; }
    done
done
BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done
run() { singularity exec "${BIND[@]}" "$IMAGE" "$PYTHON" "$@"; }
ARGS=(--generation "M37=$M37" --generation "M38=$M38")
if [ -n "${M47_9B_M40_GEN:-}" ]; then
    for f in generated_test.pt summary.json resolved_config.yaml; do
        test -s "$M47_9B_M40_GEN/$f" || { echo "ERROR: missing M40 $f" >&2; exit 2; }
    done
    ARGS+=(--generation "M40=$M47_9B_M40_GEN")
fi

echo "[M47.9B] job=$SLURM_JOB_ID host=$(hostname) git=$(git rev-parse HEAD)"
echo "[M47.9B] M37=$M37"
echo "[M47.9B] M38=$M38"
echo "[M47.9B] output=$OUT"
run -m py_compile scripts/m47_9b_boundary_metrics.py \
    scripts/m47_9b_periodic_boundary_audit.py \
    tests/unit/test_m47_9b_boundary_metrics.py \
    tests/unit/test_m47_9b_boundary_audit.py
run -m pytest -q \
    tests/unit/test_m47_9b_boundary_metrics.py \
    tests/unit/test_m47_9b_boundary_audit.py \
    tests/unit/test_m47_9_hit_les_forced.py
run -u -m scripts.m47_9b_periodic_boundary_audit \
    "${ARGS[@]}" --core-trim 4 --output-dir "$OUT"
for f in audit.json per_field.csv per_axis.csv per_shell.csv \
    plots/face_mismatch.png plots/divergence_by_depth.png \
    plots/divergence_slice_profiles.png; do
    test -s "$OUT/$f" || { echo "ERROR: absent result $OUT/$f" >&2; exit 2; }
done
echo "[M47.9B] COMPLETE $OUT/audit.json"
