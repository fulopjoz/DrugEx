#!/bin/bash
#PBS -N plip_score_top200
#PBS -q cpu
#PBS -l select=1:ncpus=46:mem=60gb
#PBS -l walltime=4:00:00
#PBS -o /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/redock_top200/
#PBS -e /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/redock_top200/
#
# Post-merge: collect all 200 re-docked molecules, run PLIP + tiered scoring.
# Run AFTER 34_redock_top200.sh array completes.
#
# Usage:
#   qsub -W depend=afterokarray:<JOB_ID> ccr2_gen/scripts/35_plip_score_top200.sh

set -euo pipefail

DRUGEX_DIR="/home/fulopj/drx-run/DrugEx"
CONDA_ROOT="/home/fulopj/miniconda3"
CONDA_ENV="drugex"

RESULTS_BASE="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/redock_top200/results"
MERGED_DIR="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/redock_top200/merged"
PROFILES_DIR="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/redock_top200/interaction_profiles"
SCORED_DIR="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/redock_top200/scored_candidates"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${DRUGEX_DIR}"
export PYTHONPATH="${DRUGEX_DIR}:${PYTHONPATH:-}"

# --- Step 1: Merge chunk results into single directory ---
echo "=== Merging re-docking results ==="
mkdir -p "${MERGED_DIR}"

# Copy all per-molecule directories from all chunks into merged
for chunk_dir in "${RESULTS_BASE}"/chunk_*/; do
    for mol_dir in "${chunk_dir}"/CCR2_GEN_*/; do
        if [ -d "${mol_dir}" ]; then
            mol_id=$(basename "${mol_dir}")
            cp -r "${mol_dir}" "${MERGED_DIR}/${mol_id}"
        fi
    done
done

# Merge redock_results.tsv from all chunks (use header from first chunk)
first_tsv=""
for chunk_dir in "${RESULTS_BASE}"/chunk_*/; do
    tsv="${chunk_dir}/redock_results.tsv"
    if [ -f "${tsv}" ]; then
        if [ -z "${first_tsv}" ]; then
            first_tsv="${tsv}"
            head -1 "${tsv}" > "${MERGED_DIR}/redock_results.tsv"
        fi
        tail -n +2 "${tsv}" >> "${MERGED_DIR}/redock_results.tsv"
    fi
done

n_merged=$(find "${MERGED_DIR}" -maxdepth 1 -type d -name "CCR2_GEN_*" | wc -l)
echo "  Merged: ${n_merged} molecules"

# --- Step 2: Run PLIP on all merged molecules ---
echo ""
echo "=== Running PLIP interaction profiling ==="
python3 ccr2_gen/docking/plip_analysis/run_plip.py \
    --redock-dir "${MERGED_DIR}" \
    --out-dir "${PROFILES_DIR}"

# --- Step 3: Score all candidates ---
echo ""
echo "=== Scoring candidates (tiered IFP + docking composite) ==="
cd ccr2_gen/docking/plip_analysis
python3 score_candidates.py \
    --redock-dir "${MERGED_DIR}" \
    --profiles-dir "${PROFILES_DIR}" \
    --out-dir "${SCORED_DIR}"
cd "${DRUGEX_DIR}"

echo ""
echo "=== Pipeline complete ==="
echo "  Re-docked: ${n_merged} molecules"
echo "  Profiles:  ${PROFILES_DIR}"
echo "  Scored:    ${SCORED_DIR}/scored_candidates.tsv"
