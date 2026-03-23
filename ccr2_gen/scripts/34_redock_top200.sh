#!/bin/bash
#PBS -N redock_top200
#PBS -q cpu
#PBS -l select=1:ncpus=46:mem=60gb
#PBS -l walltime=8:00:00
#PBS -J 1-20
#PBS -o /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/redock_top200/
#PBS -e /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/redock_top200/
#
# Re-dock top 200 ECR candidates at exhaustiveness=32 for PLIP analysis.
# PBS array: 20 chunks x 10 molecules each.
#
# Pre-requisite: run prepare_redock_top200.py first to generate chunk files.
#
# Usage:
#   mkdir -p ccr2_gen/logs/redock_top200
#   qsub ccr2_gen/scripts/34_redock_top200.sh

set -euo pipefail

# --- Config ---
DRUGEX_DIR="/home/fulopj/drx-run/DrugEx"
CONDA_ROOT="/home/fulopj/miniconda3"
CONDA_ENV="drugex"

CHUNK_DIR="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/redock_top200/chunks"
OUT_BASE="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/redock_top200/results"
RECEPTOR_PDBQT="${DRUGEX_DIR}/ccr2_gen/docking/receptor/5T1A_clean_mutations_reversed_withHs.pdbqt"
RECEPTOR_PDB="${DRUGEX_DIR}/ccr2_gen/docking/receptor/5T1A_clean_mutations_reversed_withHs.pdb"

EXHAUSTIVENESS=32
N_POSES=20

# --- Setup ---
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${DRUGEX_DIR}"
export PYTHONPATH="${DRUGEX_DIR}:${PYTHONPATH:-}"

# --- Chunk for this array index ---
CHUNK_IDX=$(printf "%02d" ${PBS_ARRAY_INDEX})
CHUNK_FILE="${CHUNK_DIR}/chunk_${CHUNK_IDX}.tsv"
OUT_DIR="${OUT_BASE}/chunk_${CHUNK_IDX}"

if [ ! -f "${CHUNK_FILE}" ]; then
    echo "ERROR: Chunk file not found: ${CHUNK_FILE}"
    exit 1
fi

echo "=== Re-docking chunk ${CHUNK_IDX} ==="
echo "  Chunk: ${CHUNK_FILE}"
echo "  Output: ${OUT_DIR}"
echo "  Exhaustiveness: ${EXHAUSTIVENESS}"
echo "  Poses: ${N_POSES}"

python3 ccr2_gen/docking/plip_analysis/redock_top10.py \
    --consensus-tsv "${CHUNK_FILE}" \
    --receptor-pdbqt "${RECEPTOR_PDBQT}" \
    --receptor-pdb "${RECEPTOR_PDB}" \
    --out-dir "${OUT_DIR}" \
    --exhaustiveness ${EXHAUSTIVENESS} \
    --n-poses ${N_POSES} \
    --max-molecules 10

echo "=== Chunk ${CHUNK_IDX} complete ==="
