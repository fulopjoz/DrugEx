#!/bin/bash
#PBS -N merge_extract
#PBS -q cpu
#PBS -l select=1:ncpus=46:mem=60gb
#PBS -l walltime=4:00:00
#PBS -o /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/full_pipeline/
#PBS -e /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/full_pipeline/
#
# Step 1: Merge Vina results + Extract Glide poses.
# Run AFTER 40_vina_dock_all.sh completes.

set -eo pipefail

DRUGEX_DIR="/home/fulopj/drx-run/DrugEx"
CONDA_ROOT="/home/fulopj/miniconda3"
CONDA_ENV="drugex"

BASE="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/full_pipeline"
VINA_RESULTS="${BASE}/vina_results"
GLIDE_RAW="${DRUGEX_DIR}/ccr2_gen/docking/results/glide_sp"
RECEPTOR_PDB="${DRUGEX_DIR}/ccr2_gen/docking/receptor/5T1A_clean_mutations_reversed_withHs.pdb"
FILTERED_TSV="${BASE}/filtered_molecules.tsv"

SCRATCHDIR="/scratch/$USER/${PBS_JOBID}"
mkdir -p "${SCRATCHDIR}"

cleanup() {
    # Copy merged and glide_poses to home
    for subdir in vina_merged glide_poses; do
        if [ -d "${SCRATCHDIR}/${subdir}" ]; then
            echo "Copying ${subdir} to home..."
            mkdir -p "${BASE}/${subdir}"
            rsync -a "${SCRATCHDIR}/${subdir}/" "${BASE}/${subdir}/" \
                || echo "WARNING: ${subdir} copy failed! Files at $(hostname):${SCRATCHDIR}/${subdir}"
        fi
    done
    rm -rf "${SCRATCHDIR}"
    echo "Scratch cleaned."
}
trap cleanup EXIT

set +eu
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -eu

cd "${DRUGEX_DIR}"
export PYTHONPATH="${DRUGEX_DIR}:${PYTHONPATH:-}"

echo "=== Merge + Extract ($(date)) on $(hostname) ==="

# ---- Merge Vina ----
echo ""
echo "=== Merging Vina re-dock results ==="
VINA_MERGED="${SCRATCHDIR}/vina_merged"
mkdir -p "${VINA_MERGED}"

for chunk_dir in "${VINA_RESULTS}"/chunk_*/; do
    for mol_dir in "${chunk_dir}"/CCR2_GEN_*/; do
        [ -d "${mol_dir}" ] && cp -r "${mol_dir}" "${VINA_MERGED}/$(basename ${mol_dir})"
    done
done

first_tsv=""
for chunk_dir in "${VINA_RESULTS}"/chunk_*/; do
    tsv="${chunk_dir}/redock_results.tsv"
    if [ -f "${tsv}" ]; then
        if [ -z "${first_tsv}" ]; then
            first_tsv="${tsv}"; head -1 "${tsv}" > "${VINA_MERGED}/redock_results.tsv"
        fi
        tail -n +2 "${tsv}" >> "${VINA_MERGED}/redock_results.tsv"
    fi
done

n_vina=$(find "${VINA_MERGED}" -maxdepth 1 -type d -name "CCR2_GEN_*" | wc -l)
echo "  Vina merged: ${n_vina} molecules"

# ---- Extract Glide ----
echo ""
echo "=== Extracting Glide poses ==="
python3 ccr2_gen/docking/plip_analysis/extract_glide_poses.py \
    --glide-results-dir "${GLIDE_RAW}" \
    --ecr-tsv "${FILTERED_TSV}" \
    --receptor-pdb "${RECEPTOR_PDB}" \
    --out-dir "${SCRATCHDIR}/glide_poses"

n_glide=$(find "${SCRATCHDIR}/glide_poses" -maxdepth 1 -type d -name "CCR2_GEN_*" | wc -l)
echo "  Glide complexes: ${n_glide}"

echo ""
echo "=== Merge + Extract complete ($(date)) ==="
echo "  Vina: ${n_vina}, Glide: ${n_glide}"
