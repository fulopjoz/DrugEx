#!/bin/bash
#PBS -N redock_2k
#PBS -q cpu
#PBS -l select=1:ncpus=46:mem=60gb
#PBS -l walltime=4:00:00
#PBS -J 1-200%10
#PBS -o /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/redock_top2000/
#PBS -e /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/redock_top2000/
#
# Re-dock top 2000 ECR candidates at exhaustiveness=32 for PLIP analysis.
# PBS array: 200 chunks x 10 molecules each.
# %10 = max 10 concurrent sub-jobs (leaves 2 CPU nodes free).
# Adjust live: qalter -W max_run_subjobs=N <jobid>[]
#
# Uses /scratch/$USER/ on worker node (NFS home is not for computation).

set -eo pipefail

# --- Paths ---
DRUGEX_DIR="/home/fulopj/drx-run/DrugEx"
CONDA_ROOT="/home/fulopj/miniconda3"
CONDA_ENV="drugex"
CHUNK_IDX=$(printf "%03d" ${PBS_ARRAY_INDEX})

CHUNK_DIR="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/redock_top2000/chunks"
FINAL_OUT="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/redock_top2000/results/chunk_${CHUNK_IDX}"
RECEPTOR_PDBQT="${DRUGEX_DIR}/ccr2_gen/docking/receptor/5T1A_clean_mutations_reversed_withHs.pdbqt"
RECEPTOR_PDB="${DRUGEX_DIR}/ccr2_gen/docking/receptor/5T1A_clean_mutations_reversed_withHs.pdb"

EXHAUSTIVENESS=32
N_POSES=20

# --- Scratch setup ---
SCRATCHDIR="/scratch/$USER/${PBS_JOBID}"
mkdir -p "${SCRATCHDIR}"

# Trap: copy results on exit (success or failure), then clean scratch
cleanup() {
    echo "Copying results from scratch to home..."
    if [ -d "${SCRATCHDIR}/out" ]; then
        mkdir -p "${FINAL_OUT}"
        cp -r "${SCRATCHDIR}/out/"* "${FINAL_OUT}/" || echo "WARNING: result copy failed! Files at $(hostname):${SCRATCHDIR}/out"
    fi
    rm -rf "${SCRATCHDIR}"
    echo "Scratch cleaned."
}
trap cleanup EXIT

# --- Conda activation (relax -eu for conda internals) ---
set +eu
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -eu

# Verify environment
python3 -c "import vina; import meeko; import rdkit" || { echo "ERROR: missing packages"; exit 2; }

# --- Copy inputs to scratch ---
CHUNK_FILE="${CHUNK_DIR}/chunk_${CHUNK_IDX}.tsv"
if [ ! -f "${CHUNK_FILE}" ]; then
    echo "ERROR: Chunk file not found: ${CHUNK_FILE}"
    exit 1
fi

cp "${CHUNK_FILE}" "${SCRATCHDIR}/"
cp "${RECEPTOR_PDBQT}" "${SCRATCHDIR}/"
cp "${RECEPTOR_PDB}" "${SCRATCHDIR}/"

echo "=== Re-docking chunk ${CHUNK_IDX} ($(date)) ==="
echo "  Node: $(hostname)"
echo "  Scratch: ${SCRATCHDIR}"
echo "  Chunk: ${CHUNK_FILE}"

# --- Run docking in scratch ---
cd "${DRUGEX_DIR}"
export PYTHONPATH="${DRUGEX_DIR}:${PYTHONPATH:-}"

python3 ccr2_gen/docking/plip_analysis/redock_top10.py \
    --consensus-tsv "${SCRATCHDIR}/chunk_${CHUNK_IDX}.tsv" \
    --receptor-pdbqt "${SCRATCHDIR}/$(basename ${RECEPTOR_PDBQT})" \
    --receptor-pdb "${SCRATCHDIR}/$(basename ${RECEPTOR_PDB})" \
    --out-dir "${SCRATCHDIR}/out" \
    --exhaustiveness ${EXHAUSTIVENESS} \
    --n-poses ${N_POSES} \
    --max-molecules 10

echo "=== Chunk ${CHUNK_IDX} complete ($(date)) ==="
# cleanup trap fires on EXIT, copies results to FINAL_OUT
