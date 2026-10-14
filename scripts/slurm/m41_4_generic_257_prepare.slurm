#!/bin/bash
#SBATCH --job-name=m41_4_generic_prepare
#SBATCH --time=04:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# Grace resources explicitly reserved; no --mem cap.
# Stage 1: numerical gates on genuinely irregular meshes.
# Stage 2: stream real 257^3 connectivity into generic disk-backed CSR
# and extract exact-two-hop owned-query neighborhoods per partition.
set -euo pipefail
REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
BASE=${M41_4_DATA_BASE:-/archivenfs/phlusim/HIT_DNS_COMP/Re150/Ma0.3}
MESH_FILE=${MESH_FILE:-${BASE}/MESH/mesh.mesh.h5}
SNAPSHOT_FILE=${SNAPSHOT_FILE:-${BASE}/SOLUT/solut_hit_00000001.h5}
GRAPH_ROOT=${GRAPH_ROOT:-/scratch/coop/theret/GraphAttention_runs/m41_4_257_generic_graph}
BLOCKS=${M41_4_BLOCKS:-"4 4 4"}
COORD_DIVISOR=${M41_4_COORD_DIVISOR:-0.0006227710672453675}
MAX_OWNED=${M41_4_MAX_OWNED:-400000}
TESTS=${M41_4_TESTS:-"tests/unit/test_m41_4_generic_csr.py tests/unit/test_streamed_dinat.py tests/unit/test_owned_query_dinat.py tests/unit/test_geometry_connectivity.py"}
cd "$REPO"
test -s "$MESH_FILE" || { echo "Missing mesh $MESH_FILE" >&2; exit 1; }
test -s "$SNAPSHOT_FILE" || { echo "Missing snapshot $SNAPSHOT_FILE" >&2; exit 1; }
echo "===== M41.4 unstructured numerical gate ====="
echo "Host=$(hostname), git=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest $TESTS -q
echo "===== M41.4 real 257^3 generic CSR/partition construction ====="
read -r BX BY BZ <<< "$BLOCKS"
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m41_4_generic_large prepare \
    --mesh-file "$MESH_FILE" --snapshot-file "$SNAPSHOT_FILE" \
    --output-dir "$GRAPH_ROOT" --expected-nodes 16974593 \
    --element-type hex --connectivity-path "Connectivity/hex->node" \
    --indexing one --blocks "$BX" "$BY" "$BZ" \
    --coordinate-divisor "$COORD_DIVISOR" --max-owned-nodes "$MAX_OWNED"
echo "Prepared graph: $GRAPH_ROOT/prepare.json"
