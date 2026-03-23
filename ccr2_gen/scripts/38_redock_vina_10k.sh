#!/bin/bash
#PBS -N vina_10k
#PBS -q cpu
#PBS -l select=1:ncpus=46:mem=60gb
#PBS -l walltime=4:00:00
#PBS -J 1-1000%10
#PBS -o /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/redock_top10k/
#PBS -e /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/redock_top10k/
#
# Track 1: Re-dock top 10K ECR candidates with Vina (exhaust=32).
# PBS array: 1000 chunks x 10 molecules. %10 concurrent.

set -eo pipefail

DRUGEX_DIR="/home/fulopj/drx-run/DrugEx"
CONDA_ROOT="/home/fulopj/miniconda3"
CONDA_ENV="drugex"

CHUNK_DIR="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/redock_top10k/chunks"
FINAL_OUT="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/redock_top10k/vina_results"
RECEPTOR_PDBQT="${DRUGEX_DIR}/ccr2_gen/docking/receptor/5T1A_clean_mutations_reversed_withHs.pdbqt"
RECEPTOR_PDB="${DRUGEX_DIR}/ccr2_gen/docking/receptor/5T1A_clean_mutations_reversed_withHs.pdb"

EXHAUSTIVENESS=32
N_POSES=20

# Scratch
SCRATCHDIR="/scratch/$USER/${PBS_JOBID}"
mkdir -p "${SCRATCHDIR}"
cleanup() {
    if [ -d "${SCRATCHDIR}/out" ]; then
        CHUNK_IDX_C=$(printf "%03d" ${PBS_ARRAY_INDEX})
        mkdir -p "${FINAL_OUT}/chunk_${CHUNK_IDX_C}"
        cp -r "${SCRATCHDIR}/out/"* "${FINAL_OUT}/chunk_${CHUNK_IDX_C}/" \
            || echo "WARNING: copy failed! Files at $(hostname):${SCRATCHDIR}/out"
    fi
    rm -rf "${SCRATCHDIR}"
}
trap cleanup EXIT

# Conda
set +eu
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -eu

cd "${DRUGEX_DIR}"
export PYTHONPATH="${DRUGEX_DIR}:${PYTHONPATH:-}"

CHUNK_IDX=$(printf "%03d" ${PBS_ARRAY_INDEX})
CHUNK_FILE="${CHUNK_DIR}/chunk_${CHUNK_IDX}.tsv"

if [ ! -f "${CHUNK_FILE}" ]; then
    echo "ERROR: ${CHUNK_FILE} not found"
    exit 1
fi

cp "${CHUNK_FILE}" "${SCRATCHDIR}/"
cp "${RECEPTOR_PDBQT}" "${RECEPTOR_PDB}" "${SCRATCHDIR}/"

echo "=== Vina re-dock chunk ${CHUNK_IDX} ($(date)) on $(hostname) ==="

python3 ccr2_gen/docking/plip_analysis/redock_top10.py \
    --consensus-tsv "${SCRATCHDIR}/chunk_${CHUNK_IDX}.tsv" \
    --receptor-pdbqt "${SCRATCHDIR}/$(basename ${RECEPTOR_PDBQT})" \
    --receptor-pdb "${SCRATCHDIR}/$(basename ${RECEPTOR_PDB})" \
    --out-dir "${SCRATCHDIR}/out" \
    --exhaustiveness ${EXHAUSTIVENESS} \
    --n-poses ${N_POSES} \
    --max-molecules 10

echo "=== Chunk ${CHUNK_IDX} complete ($(date)) ==="
