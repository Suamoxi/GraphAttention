#!/bin/bash
#SBATCH --job-name=m49_global_integrals
#SBATCH --partition=grace
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err
set -euo pipefail
REPO=${M49_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M49_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M49_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
ROOT=${M49_ROOT:-/scratch/coop/theret/GraphAttention_runs/m49}
BASE="$ROOT/training/m49_dinat_dit_fp32_h128_l10_h4_b8_a2_seed42_blocked50"
OUT="$ROOT/global_integrals/${SLURM_JOB_ID}/audit.json"
cd "$REPO"
test -d .git && test -z "$(git status --porcelain)" || { echo "Use a clean standalone checkout"; exit 2; }
BIND=(-B /home -B /scratch -B /gpfs-calypso/scratch:/gpfs-calypso/scratch)
test "$(singularity exec "${BIND[@]}" "$IMAGE" git -C "$REPO" rev-parse HEAD)" = "$(git rev-parse HEAD)"
run() { singularity exec "${BIND[@]}" --env "PYTHONPATH=$REPO/src:$REPO" --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 "$IMAGE" "$PYTHON" "$@"; }
run -m pytest -q tests/unit/test_m49_global_integrals.py
# Optional positional generation directories; default to existing baseline e1000.
if [[ $# == 0 ]]; then
    set -- "$BASE/generations/m49_validation_epoch_1000_heun50_seed5678"
fi
ARGS=()
for DEST in "$@"; do
    test -s "$DEST/generated_validation.pt" && test -s "$DEST/summary.json" || { echo "Missing completed generation: $DEST"; exit 2; }
    ARGS+=(--generation-dir "$DEST")
done
run -u -m scripts.audit_m49_global_integrals --baseline-run "$BASE" --output "$OUT" "${ARGS[@]}"
test -s "$OUT"
