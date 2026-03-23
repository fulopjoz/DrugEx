#!/bin/bash
#PBS -N vina_all
#PBS -q cpu
#PBS -l select=1:ncpus=46:mem=60gb
#PBS -l walltime=4:00:00
#PBS -J 1-5249%40
#PBS -o /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/full_pipeline/
#PBS -e /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/full_pipeline/
#
# Vina docking of ALL 52,488 filtered molecules at exhaust=32.
# 5249 chunks × 10 mols. %40 concurrent (fits ~42 slots across all nodes).
# Creates PDB complexes for PLIP interaction analysis.
#
# Literature justification: Agarwal & Smith (2023) Mol. Informatics 42(2)
# show pose RMSD plateaus at exhaust~25. exhaust=32 is the sweet spot for
# PLIP-quality poses without computational waste.

set -eo pipefail

DRUGEX_DIR="/home/fulopj/drx-run/DrugEx"
CONDA_ROOT="/home/fulopj/miniconda3"
CONDA_ENV="drugex"

CHUNK_DIR="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/full_pipeline/vina_chunks"
FINAL_OUT="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/full_pipeline/vina_results"
RECEPTOR_PDBQT="${DRUGEX_DIR}/ccr2_gen/docking/receptor/5T1A_clean_mutations_reversed_withHs.pdbqt"
RECEPTOR_PDB="${DRUGEX_DIR}/ccr2_gen/docking/receptor/5T1A_clean_mutations_reversed_withHs.pdb"

EXHAUSTIVENESS=32
N_POSES=20

# --- Scratch ---
SCRATCHDIR="/scratch/$USER/${PBS_JOBID}"
mkdir -p "${SCRATCHDIR}"
cleanup() {
    if [ -d "${SCRATCHDIR}/out" ]; then
        CIDX=$(printf "%04d" ${PBS_ARRAY_INDEX})
        mkdir -p "${FINAL_OUT}/chunk_${CIDX}"
        cp -r "${SCRATCHDIR}/out/"* "${FINAL_OUT}/chunk_${CIDX}/" \
            || echo "WARNING: copy failed! Files at $(hostname):${SCRATCHDIR}/out"
    fi
    rm -rf "${SCRATCHDIR}"
}
trap cleanup EXIT

# --- Conda ---
set +eu
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -eu

cd "${DRUGEX_DIR}"
export PYTHONPATH="${DRUGEX_DIR}:${PYTHONPATH:-}"

CIDX=$(printf "%04d" ${PBS_ARRAY_INDEX})
CHUNK_FILE="${CHUNK_DIR}/chunk_${CIDX}.tsv"

if [ ! -f "${CHUNK_FILE}" ]; then
    echo "ERROR: ${CHUNK_FILE} not found"
    exit 1
fi

cp "${CHUNK_FILE}" "${RECEPTOR_PDBQT}" "${RECEPTOR_PDB}" "${SCRATCHDIR}/"

python3 ccr2_gen/docking/plip_analysis/redock_top10.py \
    --consensus-tsv "${SCRATCHDIR}/chunk_${CIDX}.tsv" \
    --receptor-pdbqt "${SCRATCHDIR}/$(basename ${RECEPTOR_PDBQT})" \
    --receptor-pdb "${SCRATCHDIR}/$(basename ${RECEPTOR_PDB})" \
    --out-dir "${SCRATCHDIR}/out" \
    --exhaustiveness ${EXHAUSTIVENESS} \
    --n-poses ${N_POSES} \
    --max-molecules 10

echo "=== Chunk ${CIDX} done ($(date)) on $(hostname) ==="
