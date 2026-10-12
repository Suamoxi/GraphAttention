#!/bin/bash
#SBATCH --job-name=m41_1_halo
#SBATCH --time=01:30:00
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
OUT=${OUT:-${RUN_ROOT}/m41_1_halo_partitioned_m37}
read -r BX BY BZ <<< "${M41_BLOCKS:-2 2 2}"

cd "$REPO"
mkdir -p "$OUT"
echo "M41.1 exact single-GPU halo inference"
echo "Repository commit: $(git rev-parse HEAD)"
echo "M37 run: $M37_RUN"
echo "Checkpoint: $CHECKPOINT"
echo "Partition blocks: $BX $BY $BZ"
nvidia-smi

echo "==== Partitioned DiNAT test gate ===="
singularity exec --nv -B /home -B /scratch "$IMAGE" \
    "$PYTHON" -m pytest tests/unit/test_partitioned_dinat.py -q

echo "==== Real M37 checkpoint parity / forward-time comparison ===="
singularity exec --nv -B /home -B /scratch "$IMAGE" \
    "$PYTHON" -u -m scripts.benchmark_m41_partitioned \
    --run-dir "$M37_RUN" \
    --checkpoint "$CHECKPOINT" \
    --blocks "$BX" "$BY" "$BZ" \
    --warmup 2 --repeats 5 \
    --output "$OUT/m41_1_benchmark.json"

test -s "$OUT/m41_1_benchmark.json"
echo "M41.1 completed: $OUT/m41_1_benchmark.json"
