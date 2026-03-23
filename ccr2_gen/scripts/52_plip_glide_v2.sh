#!/bin/bash
#PBS -N plip_gl_v2
#PBS -q cpu
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=2:00:00
#PBS -J 1-52
#PBS -o /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/full_pipeline/
#PBS -e /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/full_pipeline/
#
# V2 PLIP on Glide bulk SDF poses. 52 batches (1 per SDF file).
# Each batch: ~1000 molecules, ~18 min, ncpus=2.

set -eo pipefail

DRUGEX_DIR="/home/fulopj/drx-run/DrugEx"
CONDA_ROOT="/home/fulopj/miniconda3"
CONDA_ENV="drugex"

GLIDE_BULK="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/full_pipeline/glide_bulk"
RECEPTOR_PDB="${DRUGEX_DIR}/ccr2_gen/docking/receptor/5T1A_clean_mutations_reversed_withHs.pdb"
OUT_DIR="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/full_pipeline/glide_profiles_v2"

set +eu
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -eu

cd "${DRUGEX_DIR}"
export PYTHONPATH="${DRUGEX_DIR}:${PYTHONPATH:-}"

BATCH_IDX=$(printf "%03d" ${PBS_ARRAY_INDEX})
POSES_SDF="${GLIDE_BULK}/glide_batch_${BATCH_IDX}_poses.sdf"
OUT_JSONL="${OUT_DIR}/glide_batch_${BATCH_IDX}_profiles.jsonl"

mkdir -p "${OUT_DIR}"

if [ ! -f "${POSES_SDF}" ]; then
    echo "ERROR: ${POSES_SDF} not found"
    exit 1
fi

echo "=== PLIP Glide v2 batch ${BATCH_IDX} ($(date)) on $(hostname) ==="

python3 ccr2_gen/docking/plip_analysis/run_plip_bulk.py \
    --poses-sdf "${POSES_SDF}" \
    --receptor-pdb "${RECEPTOR_PDB}" \
    --out-jsonl "${OUT_JSONL}"

echo "=== Batch ${BATCH_IDX} done ($(date)) ==="
