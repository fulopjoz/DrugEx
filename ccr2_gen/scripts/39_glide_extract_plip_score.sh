#!/bin/bash
#PBS -N glide_plip_10k
#PBS -q cpu
#PBS -l select=1:ncpus=46:mem=60gb
#PBS -l walltime=8:00:00
#PBS -o /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/redock_top10k/
#PBS -e /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/redock_top10k/
#
# Track 2: Extract Glide poses, run PLIP, then score all three tracks.
# Run AFTER 38_redock_vina_10k.sh completes:
#   qsub -W depend=afterokarray:<VINA_JOB_ID> ccr2_gen/scripts/39_glide_extract_plip_score.sh

set -eo pipefail

DRUGEX_DIR="/home/fulopj/drx-run/DrugEx"
CONDA_ROOT="/home/fulopj/miniconda3"
CONDA_ENV="drugex"

BASE="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/redock_top10k"
VINA_RESULTS="${BASE}/vina_results"
GLIDE_RAW="${DRUGEX_DIR}/ccr2_gen/docking/results/glide_sp"
RECEPTOR_PDB="${DRUGEX_DIR}/ccr2_gen/docking/receptor/5T1A_clean_mutations_reversed_withHs.pdb"
ECR_TSV="${BASE}/ecr_top10000.tsv"

SCRATCHDIR="/scratch/$USER/${PBS_JOBID}"
mkdir -p "${SCRATCHDIR}"
cleanup() {
    # Copy all outputs to home
    for subdir in vina_merged glide_poses vina_profiles glide_profiles three_track_results; do
        if [ -d "${SCRATCHDIR}/${subdir}" ]; then
            mkdir -p "${BASE}/${subdir}"
            cp -r "${SCRATCHDIR}/${subdir}/"* "${BASE}/${subdir}/" \
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

echo "============================================================"
echo "THREE-TRACK PLIP PIPELINE ($(date))"
echo "Node: $(hostname), Scratch: ${SCRATCHDIR}"
echo "============================================================"

# ---- Step 1: Merge Vina re-dock results ----
echo ""
echo "=== Step 1: Merging Vina re-dock results ==="
VINA_MERGED="${SCRATCHDIR}/vina_merged"
mkdir -p "${VINA_MERGED}"

for chunk_dir in "${VINA_RESULTS}"/chunk_*/; do
    for mol_dir in "${chunk_dir}"/CCR2_GEN_*/; do
        if [ -d "${mol_dir}" ]; then
            mol_id=$(basename "${mol_dir}")
            cp -r "${mol_dir}" "${VINA_MERGED}/${mol_id}"
        fi
    done
done

# Merge TSVs
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

# Also save merged to home for persistence
mkdir -p "${BASE}/vina_merged"
cp "${VINA_MERGED}/redock_results.tsv" "${BASE}/vina_merged/"

# ---- Step 2: Extract Glide poses ----
echo ""
echo "=== Step 2: Extracting Glide poses ==="
python3 ccr2_gen/docking/plip_analysis/extract_glide_poses.py \
    --glide-results-dir "${GLIDE_RAW}" \
    --ecr-tsv "${ECR_TSV}" \
    --receptor-pdb "${RECEPTOR_PDB}" \
    --out-dir "${SCRATCHDIR}/glide_poses"

n_glide=$(find "${SCRATCHDIR}/glide_poses" -maxdepth 1 -type d -name "CCR2_GEN_*" | wc -l)
echo "  Glide complexes: ${n_glide}"

# ---- Step 3: PLIP on Vina poses ----
echo ""
echo "=== Step 3: PLIP on Vina poses ==="
python3 ccr2_gen/docking/plip_analysis/run_plip.py \
    --redock-dir "${VINA_MERGED}" \
    --out-dir "${SCRATCHDIR}/vina_profiles"

# ---- Step 4: PLIP on Glide poses ----
echo ""
echo "=== Step 4: PLIP on Glide poses ==="
python3 ccr2_gen/docking/plip_analysis/run_plip.py \
    --redock-dir "${SCRATCHDIR}/glide_poses" \
    --out-dir "${SCRATCHDIR}/glide_profiles"

# ---- Step 5: Three-track scoring ----
echo ""
echo "=== Step 5: Three-track scoring ==="
cd ccr2_gen/docking/plip_analysis
python3 score_three_tracks.py \
    --vina-profiles "${SCRATCHDIR}/vina_profiles" \
    --glide-profiles "${SCRATCHDIR}/glide_profiles" \
    --vina-redock "${VINA_MERGED}" \
    --glide-redock "${SCRATCHDIR}/glide_poses" \
    --ecr-tsv "${ECR_TSV}" \
    --out-dir "${SCRATCHDIR}/three_track_results"
cd "${DRUGEX_DIR}"

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE ($(date))"
echo "  Vina molecules:  ${n_vina}"
echo "  Glide molecules: ${n_glide}"
echo "  Results: ${BASE}/three_track_results/"
echo "============================================================"
