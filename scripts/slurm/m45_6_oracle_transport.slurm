#!/bin/bash
#SBATCH --job-name=m45_6_oracle
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err
set -euo pipefail
REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
OUT=${M45_6_OUT:-/scratch/coop/theret/GraphAttention_runs/m45_6_oracle/job_${SLURM_JOB_ID}}
DRAWS=${M45_6_DRAWS:-32}
DIRECT_DRAWS=${M45_6_DIRECT_DRAWS:-512}
cd "$REPO"
[ -z "$(git status --porcelain)" ] || { echo "Dirty checkout" >&2; exit 2; }
[ -s "$IMAGE" ] && [ -x "$PYTHON" ] || { echo "Missing image or Python" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "Refusing existing output: $OUT" >&2; exit 2; }
echo "M45.6 analytical transport Git=$(git rev-parse HEAD) Heun=25,50,100,200,400,800 draws=$DRAWS direct=$DIRECT_DRAWS"
nvidia-smi --query-gpu=name,memory.total --format=csv
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m py_compile scripts/m45_6_oracle_transport.py tests/unit/test_m45_6_diagnostics.py
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m pytest -q tests/unit/test_m45_6_diagnostics.py
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -u -m scripts.m45_6_oracle_transport \
  --output-dir "$OUT" --device cuda --train-samples 24 \
  --seed 2026 --validation-seed 7002026 --cases local shared_mode \
  --draws "$DRAWS" --direct-draws "$DIRECT_DRAWS" \
  --heun-steps 25 50 100 200 400 800
test -s "$OUT/summary.json"
test -s "$OUT/oracle_transport.csv"
echo "M45.6 oracle completed: $OUT"
