#!/bin/bash
#PBS -N plip_2k
#PBS -q cpu
#PBS -l select=1:ncpus=46:mem=60gb
#PBS -l walltime=4:00:00
#PBS -o /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/redock_top2000/
#PBS -e /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/redock_top2000/
#
# Post-merge: collect all 2000 re-docked molecules, run PLIP + scoring.
# Run AFTER 36_redock_top2000.sh array completes:
#   qsub -W depend=afterokarray:<JOB_ID> ccr2_gen/scripts/37_plip_score_top2000.sh
#
# Uses /scratch/$USER/ on worker node for PLIP computation (memory-intensive).

set -eo pipefail

DRUGEX_DIR="/home/fulopj/drx-run/DrugEx"
CONDA_ROOT="/home/fulopj/miniconda3"
CONDA_ENV="drugex"

RESULTS_BASE="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/redock_top2000/results"
FINAL_MERGED="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/redock_top2000/merged"
FINAL_PROFILES="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/redock_top2000/interaction_profiles"
FINAL_SCORED="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/redock_top2000/scored_candidates"

# --- Scratch setup ---
SCRATCHDIR="/scratch/$USER/${PBS_JOBID}"
mkdir -p "${SCRATCHDIR}"

cleanup() {
    echo "Copying final results from scratch to home..."
    # Copy profiles
    if [ -d "${SCRATCHDIR}/interaction_profiles" ]; then
        mkdir -p "${FINAL_PROFILES}"
        cp -r "${SCRATCHDIR}/interaction_profiles/"* "${FINAL_PROFILES}/" \
            || echo "WARNING: profiles copy failed! Files at $(hostname):${SCRATCHDIR}"
    fi
    # Copy scored candidates
    if [ -d "${SCRATCHDIR}/scored_candidates" ]; then
        mkdir -p "${FINAL_SCORED}"
        cp -r "${SCRATCHDIR}/scored_candidates/"* "${FINAL_SCORED}/" \
            || echo "WARNING: scored copy failed! Files at $(hostname):${SCRATCHDIR}"
    fi
    rm -rf "${SCRATCHDIR}"
    echo "Scratch cleaned."
}
trap cleanup EXIT

# --- Conda activation ---
set +eu
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -eu

cd "${DRUGEX_DIR}"
export PYTHONPATH="${DRUGEX_DIR}:${PYTHONPATH:-}"

echo "=== Post-merge pipeline ($(date)) ==="
echo "  Node: $(hostname)"
echo "  Scratch: ${SCRATCHDIR}"

# --- Step 1: Merge chunk results into scratch ---
echo ""
echo "=== Merging re-docking results to scratch ==="
MERGED="${SCRATCHDIR}/merged"
mkdir -p "${MERGED}"

for chunk_dir in "${RESULTS_BASE}"/chunk_*/; do
    for mol_dir in "${chunk_dir}"/CCR2_GEN_*/; do
        if [ -d "${mol_dir}" ]; then
            mol_id=$(basename "${mol_dir}")
            cp -r "${mol_dir}" "${MERGED}/${mol_id}"
        fi
    done
done

# Merge redock_results.tsv (header from first chunk)
first_tsv=""
for chunk_dir in "${RESULTS_BASE}"/chunk_*/; do
    tsv="${chunk_dir}/redock_results.tsv"
    if [ -f "${tsv}" ]; then
        if [ -z "${first_tsv}" ]; then
            first_tsv="${tsv}"
            head -1 "${tsv}" > "${MERGED}/redock_results.tsv"
        fi
        tail -n +2 "${tsv}" >> "${MERGED}/redock_results.tsv"
    fi
done

# Also save merged to home for persistence
mkdir -p "${FINAL_MERGED}"
cp "${MERGED}/redock_results.tsv" "${FINAL_MERGED}/"

n_merged=$(find "${MERGED}" -maxdepth 1 -type d -name "CCR2_GEN_*" | wc -l)
n_success=$(tail -n +2 "${MERGED}/redock_results.tsv" | grep -c "success" || true)
n_failed=$(tail -n +2 "${MERGED}/redock_results.tsv" | grep -c "failed" || true)
echo "  Merged: ${n_merged} molecules (${n_success} success, ${n_failed} failed)"

# --- Step 2: Run PLIP in scratch (avoids OOM from NFS buffering) ---
echo ""
echo "=== Running PLIP interaction profiling (in scratch) ==="
python3 ccr2_gen/docking/plip_analysis/run_plip.py \
    --redock-dir "${MERGED}" \
    --out-dir "${SCRATCHDIR}/interaction_profiles"

n_profiles=$(ls "${SCRATCHDIR}/interaction_profiles/"*_interactions.json 2>/dev/null | wc -l)
echo "  Generated ${n_profiles} interaction profiles"

# --- Step 3: Score in scratch ---
echo ""
echo "=== Scoring candidates ==="
cd ccr2_gen/docking/plip_analysis
python3 score_candidates.py \
    --redock-dir "${MERGED}" \
    --profiles-dir "${SCRATCHDIR}/interaction_profiles" \
    --out-dir "${SCRATCHDIR}/scored_candidates"
cd "${DRUGEX_DIR}"

echo ""
echo "=== Pipeline complete ($(date)) ==="
echo "  Re-docked: ${n_merged} molecules"
echo "  Profiles:  ${n_profiles}"
echo "  Scored:    ${SCRATCHDIR}/scored_candidates/scored_candidates.tsv"
# cleanup trap fires on EXIT, copies profiles + scored to home
