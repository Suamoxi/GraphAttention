#!/bin/bash
#SBATCH --job-name=m45_6_repro
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err
set -euo pipefail
REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
TASK=${M45_6_TASK:-trace}
STEPS=${M45_6_STEPS:-128}
SEED=${M45_6_SEED:-2026}
DETERMINISTIC=${M45_6_DETERMINISTIC:-off}
OUT=${M45_6_OUT:-/scratch/coop/theret/GraphAttention_runs/m45_6_repro/job_${SLURM_JOB_ID}}
# Required by CuBLAS when PyTorch deterministic algorithms are enabled.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$REPO"
[ -z "$(git status --porcelain)" ] || { echo "Dirty checkout" >&2; exit 2; }
[ -s "$IMAGE" ] && [ -x "$PYTHON" ] || { echo "Missing image or Python" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "Refusing existing output: $OUT" >&2; exit 2; }
echo "M45.6 reproducibility Git=$(git rev-parse HEAD) task=$TASK steps=$STEPS seed=$SEED deterministic=$DETERMINISTIC"
nvidia-smi --query-gpu=name,memory.total --format=csv
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m py_compile scripts/m45_6_reproducibility.py scripts/m45_6_oracle_transport.py \
  scripts/m45_6_compare_depth.py tests/unit/test_m45_6_diagnostics.py
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m pytest -q tests/unit/test_m45_6_diagnostics.py
mkdir -p "$OUT"
case "$TASK" in
  trace)
    # Separate Python processes, identical parameters/data/noise paths.
    for replica in 0 1; do
      singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
        "$PYTHON" -u -m scripts.m45_6_reproducibility \
        --task trace --output-dir "$OUT/replica_${replica}" --device cuda \
        --cases local shared_mode --modes static dynamic --depth 2 \
        --seed "$SEED" --steps "$STEPS" --deterministic "$DETERMINISTIC"
    done
    singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
      "$PYTHON" -u -m scripts.m45_6_reproducibility \
      --task compare --left "$OUT/replica_0" --right "$OUT/replica_1" \
      --output-dir "$OUT/comparison"
    test -s "$OUT/comparison/summary.json"
    ;;
  parity)
    # Identical sampled FM paths and owned-node objectives, two model depths.
    for depth in 2 4; do
      singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
        "$PYTHON" -u -m scripts.m45_6_reproducibility \
        --task parity --output-dir "$OUT/depth_${depth}" --device cuda \
        --cases local shared_mode --modes static dynamic --depth "$depth" \
        --seed "$SEED" --steps "$STEPS" --deterministic "$DETERMINISTIC"
      test -s "$OUT/depth_${depth}/summary.json"
    done
    ;;
  *) echo "M45_6_TASK must be trace or parity" >&2; exit 2 ;;
esac
echo "M45.6 reproducibility completed: $OUT"
