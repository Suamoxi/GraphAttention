#!/bin/bash
#SBATCH --job-name=m47_9_hit_LT
#SBATCH --partition=grace
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err

# CPU-only archival analysis on Grace. Never run Python on the login node.
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M47_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
RUNS=${M47_RUNS_ROOT:-/scratch/coop/theret/GraphAttention_runs}
OUT_ROOT=${M47_9_OUT_ROOT:-$RUNS/m47_9_hit_les_forced_helmholtz}
M37_RUN=${M37_RUN:-$RUNS/m37_3d_dinat_dit_flow_1000e_h128_l10_h4_b16}
M38_RUN=${M38_RUN:-$RUNS/m38_3d_full_dit_flow_1000e_h128_l10_h4_b18}
M40_RUN=${M40_RUN:-$RUNS/m40_3d_dinat_local_self_flow_1000e_h128_l10_h4_b16}
OUT="$OUT_ROOT/job_${SLURM_JOB_ID}"
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty checkout" >&2; exit 2; }
test -s "$IMAGE" && test -x "$PYTHON"
test ! -e "$OUT" || { echo "Refusing existing output: $OUT" >&2; exit 2; }

BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done
run() { singularity exec "${BIND[@]}" "$IMAGE" "$PYTHON" "$@"; }
latest_generation() {
    local dir="$1"
    local match
    match=$(find "$dir/generations" -maxdepth 1 -type d \
        -name 'flow_heun_steps50_seed5678_best_epoch*' -print 2>/dev/null \
        | sort -V | tail -n 1)
    if [ -n "$match" ]; then
        echo "$match"
        return 0
    fi
    local legacy="$dir/generations/flow_heun_steps50_seed5678"
    if [ -s "$legacy/generated_test.pt" ] && [ -s "$legacy/summary.json" ]; then
        if run -c "import json,pathlib,sys; s=json.loads(pathlib.Path('$legacy/summary.json').read_text()); sys.exit(0 if s.get('source_checkpoint')=='best.pt' else 1)"; then
            echo "$legacy"
            return 0
        fi
    fi
    echo "No best-checkpoint Heun 50 generation under $dir" >&2
    return 1
}
M37=$(latest_generation "$M37_RUN")
M38=$(latest_generation "$M38_RUN")
for dir in "$M37" "$M38"; do
    for file in generated_test.pt summary.json resolved_config.yaml; do
        test -s "$dir/$file" || { echo "Missing $dir/$file" >&2; exit 2; }
    done
done
MODELS=(--generation "M37=$M37" --generation "M38=$M38")
if [ "${M47_9_INCLUDE_M40:-0}" = "1" ]; then
    test -d "$M40_RUN/generations" || { echo "Requested M40 generations unavailable" >&2; exit 2; }
    M40=$(latest_generation "$M40_RUN")
    for file in generated_test.pt summary.json resolved_config.yaml; do
        test -s "$M40/$file" || { echo "Missing M40 $M40/$file" >&2; exit 2; }
    done
    MODELS+=(--generation "M40=$M40")
fi

echo "[M47.9] job=$SLURM_JOB_ID host=$(hostname) git=$(git rev-parse HEAD)"
echo "[M47.9] M37=$M37"
echo "[M47.9] M38=$M38"
echo "[M47.9] optional M40=${M40:-not_available}"
echo "[M47.9] output=$OUT"
run -m py_compile scripts/m47_9_hit_les_forced_helmholtz.py \
    tests/unit/test_m47_9_hit_les_forced.py
run -m pytest -q tests/unit/test_m47_9_hit_les_forced.py \
    tests/unit/test_m47_7_covariance.py
run -u -m scripts.m47_9_hit_les_forced_helmholtz \
    "${MODELS[@]}" --output-dir "$OUT"
test -s "$OUT/audit.json"
echo "[M47.9] COMPLETE $OUT/audit.json"
