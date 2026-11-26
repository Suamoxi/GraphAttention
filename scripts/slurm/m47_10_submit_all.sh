#!/usr/bin/env bash
# Submit five independently visible Slurm jobs:
# mesh preflight -> GPU preflight -> A1, A2, A3 (each depending on GPU only).
set -euo pipefail
REPO=${M47_10_REPO:-/scratch/coop/theret/GraphAttention}
cd "$REPO"
test -z "$(git status --porcelain)" || {
    echo "Dirty repository: refusing submission" >&2; exit 2;
}
command -v sbatch >/dev/null
command -v git >/dev/null
submit_id() {
    local id
    id=$(sbatch --parsable "$@")
    id=${id%%;*}
    [[ "$id" =~ ^[0-9]+$ ]] || {
        echo "Unexpected Slurm job ID: $id" >&2; return 2;
    }
    printf '%s\n' "$id"
}
rev=$(git rev-parse HEAD)
mesh=$(submit_id scripts/slurm/m47_10_mesh_preflight.slurm)
gpu=$(submit_id "--dependency=afterok:$mesh" scripts/slurm/m47_10_gpu_preflight.slurm)
a1=$(submit_id "--dependency=afterok:$gpu" scripts/slurm/m47_10_train_a1.slurm)
a2=$(submit_id "--dependency=afterok:$gpu" scripts/slurm/m47_10_train_a2.slurm)
a3=$(submit_id "--dependency=afterok:$gpu" scripts/slurm/m47_10_train_a3.slurm)
printf 'Revision: %s\nMesh preflight: %s\nGPU preflight: %s\nA1 training: %s\nA2 training: %s\nA3 training: %s\n' \
    "$rev" "$mesh" "$gpu" "$a1" "$a2" "$a3"
echo 'A1, A2 and A3 are independent Slurm jobs; each requires the GPU preflight to succeed.'
echo 'Monitor: sacct -j <jobID> -o JobID,JobName,State,ExitCode,Reason'
