#!/bin/bash
#PBS -N plip_3track
#PBS -q cpu
#PBS -l select=1:ncpus=190:mem=2000gb
#PBS -l walltime=10:00:00
#PBS -o /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/full_pipeline/
#PBS -e /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/full_pipeline/
#
# Full post-docking pipeline:
#   1. Merge Vina results (52K molecules)
#   2. Extract Glide poses from raw CSVs (52K molecules)
#   3. PLIP on Vina poses (52K)
#   4. PLIP on Glide poses (52K)
#   5. Three-track scoring (Vina, Glide, Consensus)
#
# Runs on a single emcpu node (190 CPUs, 2TB RAM) to avoid OOM.
# PLIP is serial but we have enough RAM for the summary data.
# Run AFTER 40_vina_dock_all.sh completes.
#
# Usage:
#   qsub -W depend=afterok:7022[] ccr2_gen/scripts/41_plip_three_track.sh

set -eo pipefail

DRUGEX_DIR="/home/fulopj/drx-run/DrugEx"
CONDA_ROOT="/home/fulopj/miniconda3"
CONDA_ENV="drugex"

BASE="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/full_pipeline"
VINA_RESULTS="${BASE}/vina_results"
GLIDE_RAW="${DRUGEX_DIR}/ccr2_gen/docking/results/glide_sp"
RECEPTOR_PDB="${DRUGEX_DIR}/ccr2_gen/docking/receptor/5T1A_clean_mutations_reversed_withHs.pdb"
FILTERED_TSV="${BASE}/filtered_molecules.tsv"

# Use scratch on the emcpu node (22.8 TB NVMe!)
SCRATCHDIR="/scratch/$USER/${PBS_JOBID}"
mkdir -p "${SCRATCHDIR}"

cleanup() {
    echo "=== Copying results to home ($(date)) ==="
    for subdir in vina_merged glide_poses vina_profiles glide_profiles three_track_results; do
        if [ -d "${SCRATCHDIR}/${subdir}" ]; then
            mkdir -p "${BASE}/${subdir}"
            rsync -a "${SCRATCHDIR}/${subdir}/" "${BASE}/${subdir}/" \
                || echo "WARNING: ${subdir} rsync failed! Files at $(hostname):${SCRATCHDIR}/${subdir}"
            echo "  ${subdir}: $(ls "${SCRATCHDIR}/${subdir}" | wc -l) files copied"
        fi
    done
    rm -rf "${SCRATCHDIR}"
    echo "Scratch cleaned."
}
trap cleanup EXIT

# --- Conda ---
set +eu
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -eu

cd "${DRUGEX_DIR}"
export PYTHONPATH="${DRUGEX_DIR}:${PYTHONPATH:-}"

echo "============================================================"
echo "THREE-TRACK PLIP PIPELINE — ALL FILTERED MOLECULES"
echo "$(date) — $(hostname)"
echo "Scratch: ${SCRATCHDIR}"
echo "============================================================"

# ======== Step 1: Merge Vina re-dock results ========
echo ""
echo "=== Step 1: Merging Vina re-dock results ==="
VINA_MERGED="${SCRATCHDIR}/vina_merged"
mkdir -p "${VINA_MERGED}"

# Copy all per-molecule dirs from chunks to merged
for chunk_dir in "${VINA_RESULTS}"/chunk_*/; do
    for mol_dir in "${chunk_dir}"/CCR2_GEN_*/; do
        [ -d "${mol_dir}" ] && cp -r "${mol_dir}" "${VINA_MERGED}/$(basename ${mol_dir})"
    done
done

# Merge redock_results.tsv
first_tsv=""
for chunk_dir in "${VINA_RESULTS}"/chunk_*/; do
    tsv="${chunk_dir}/redock_results.tsv"
    if [ -f "${tsv}" ]; then
        if [ -z "${first_tsv}" ]; then
            first_tsv="${tsv}"
            head -1 "${tsv}" > "${VINA_MERGED}/redock_results.tsv"
        fi
        tail -n +2 "${tsv}" >> "${VINA_MERGED}/redock_results.tsv"
    fi
done

n_vina=$(find "${VINA_MERGED}" -maxdepth 1 -type d -name "CCR2_GEN_*" | wc -l)
echo "  Vina merged: ${n_vina} molecules"

# ======== Step 2: Extract Glide poses ========
echo ""
echo "=== Step 2: Extracting Glide poses from raw CSVs ==="
python3 ccr2_gen/docking/plip_analysis/extract_glide_poses.py \
    --glide-results-dir "${GLIDE_RAW}" \
    --ecr-tsv "${FILTERED_TSV}" \
    --receptor-pdb "${RECEPTOR_PDB}" \
    --out-dir "${SCRATCHDIR}/glide_poses"

n_glide=$(find "${SCRATCHDIR}/glide_poses" -maxdepth 1 -type d -name "CCR2_GEN_*" | wc -l)
echo "  Glide complexes: ${n_glide}"

# ======== Step 3: PLIP on Vina poses ========
echo ""
echo "=== Step 3: PLIP on Vina poses (${n_vina} molecules) ==="
python3 ccr2_gen/docking/plip_analysis/run_plip.py \
    --redock-dir "${VINA_MERGED}" \
    --out-dir "${SCRATCHDIR}/vina_profiles"

n_vina_prof=$(ls "${SCRATCHDIR}/vina_profiles/"*_interactions.json 2>/dev/null | wc -l)
echo "  Vina profiles: ${n_vina_prof}"

# ======== Step 4: PLIP on Glide poses ========
echo ""
echo "=== Step 4: PLIP on Glide poses (${n_glide} molecules) ==="
python3 ccr2_gen/docking/plip_analysis/run_plip.py \
    --redock-dir "${SCRATCHDIR}/glide_poses" \
    --out-dir "${SCRATCHDIR}/glide_profiles"

n_glide_prof=$(ls "${SCRATCHDIR}/glide_profiles/"*_interactions.json 2>/dev/null | wc -l)
echo "  Glide profiles: ${n_glide_prof}"

# ======== Step 5: Three-track scoring ========
echo ""
echo "=== Step 5: Three-track scoring ==="

# First compute ECR for all filtered molecules
python3 ccr2_gen/docking/plip_analysis/prepare_redock_top200.py \
    --vina-tsv ccr2_gen/docking/analysis_vina/docking_results_merged.tsv \
    --glide-tsv ccr2_gen/docking/analysis_glide_sp/docking_results_merged.tsv \
    --out-dir "${SCRATCHDIR}/ecr_all" \
    --top-n 99999 --mw-cutoff 546 --max-rings 5 --chunk-size 99999

cd ccr2_gen/docking/plip_analysis
python3 score_three_tracks.py \
    --vina-profiles "${SCRATCHDIR}/vina_profiles" \
    --glide-profiles "${SCRATCHDIR}/glide_profiles" \
    --vina-redock "${VINA_MERGED}" \
    --glide-redock "${SCRATCHDIR}/glide_poses" \
    --ecr-tsv "${SCRATCHDIR}/ecr_all/ecr_top99999.tsv" \
    --out-dir "${SCRATCHDIR}/three_track_results"
cd "${DRUGEX_DIR}"

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE ($(date))"
echo "  Vina profiled: ${n_vina_prof}"
echo "  Glide profiled: ${n_glide_prof}"
echo "  Results: ${BASE}/three_track_results/"
echo "============================================================"
