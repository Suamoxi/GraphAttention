#!/bin/bash
#SBATCH --job-name=m49_heun
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err
set -euo pipefail
REPO=${M49_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M49_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M49_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
ROOT=${M49_ROOT:-/scratch/coop/theret/GraphAttention_runs/m49}
RUN="$ROOT/training/m49_dinat_dit_fp32_h128_l10_h4_b8_a2_seed42_blocked50"
cd "$REPO"
test -d .git && test -z "$(git status --porcelain)" || { echo "Use a clean standalone checkout"; exit 2; }
BIND=(-B /home -B /scratch)
test "$(singularity exec "${BIND[@]}" "$IMAGE" git -C "$REPO" rev-parse HEAD)" = "$(git rev-parse HEAD)"
exec 9>"$RUN/.m49_sampling_study.lock"
flock -n 9 || { echo "Sampling study already running"; exit 2; }
run() { singularity exec --nv "${BIND[@]}" --env "PYTHONPATH=$REPO/src:$REPO" "$IMAGE" "$PYTHON" "$@"; }
run -m pytest -q tests/unit/test_m49_sampling_convergence.py
for steps in 50 100 200; do
    run -u -m scripts.evaluate_m49_generation --run-dir "$RUN" --epoch 1000 --steps "$steps" --sampling-study
done
run -u -m scripts.compare_m49_sampling --run-dir "$RUN"
