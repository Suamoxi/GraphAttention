#!/bin/bash
#SBATCH --job-name=m44_sr0_pilot
#SBATCH --time=01:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# One REAL DNS snapshot, point-sampled mock LES (NOT physical LES filtering).
# This smoke test contains no independent scientific validation.
set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
ROOT=${M44_SR0_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
CASE=${M44_SR0_CASE:-Re54/Ma0.3}
REFS=${M44_SR0_REFS:-/scratch/coop/theret/HIT_DNS_COMP_case_references}
OUT=${M44_SR0_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m44_sr0_pilot/job_${SLURM_JOB_ID}}

cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
    echo "Refusing a dirty repository for M44 SR-0 pilot" >&2
    exit 1
fi
if [ -e "$OUT" ]; then
    echo "Output already exists: $OUT" >&2
    exit 1
fi
REF_NAME=${CASE//\//_}.yaml
if [ ! -s "$REFS/$REF_NAME" ]; then
    echo "Missing audited case reference: $REFS/$REF_NAME" >&2
    exit 1
fi

echo "M44 SR-0 pilot: Git $(git rev-parse HEAD), host $(hostname), job $SLURM_JOB_ID"
echo "One-snapshot diagnostic only. No physical LES / Framework-B claims."
nvidia-smi --query-gpu=name,memory.total --format=csv

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q \
    tests/unit/test_m44_sr0_cartesian.py \
    tests/unit/test_m44_sr0_pilot_pair.py \
    tests/unit/test_m44_sr0.py \
    tests/unit/test_flow_matching_task.py

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m44_sr0_pilot \
    --root "$ROOT" --case "$CASE" --reference-dir "$REFS" \
    --side 65 --patch-cells 8 --refinement 2 \
    --steps 2 --sample-steps 4 --device cuda \
    --output "$OUT"

echo "M44 SR-0 numerical pilot results: $OUT/summary.json"
