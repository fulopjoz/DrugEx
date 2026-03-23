#!/bin/bash
#PBS -N score_3track
#PBS -q cpu
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -l walltime=1:00:00
#PBS -o /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/full_pipeline/
#PBS -e /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/full_pipeline/
#
# Final scoring: combine PLIP profiles from both engines + ECR → three rankings.
# Run AFTER both 42_plip_vina_array.sh AND 43_plip_glide_array.sh complete.

set -eo pipefail

DRUGEX_DIR="/home/fulopj/drx-run/DrugEx"
CONDA_ROOT="/home/fulopj/miniconda3"
CONDA_ENV="drugex"

BASE="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/full_pipeline"

set +eu
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -eu

cd "${DRUGEX_DIR}"
export PYTHONPATH="${DRUGEX_DIR}:${PYTHONPATH:-}"

echo "=== Three-Track Scoring ($(date)) ==="

# Compute ECR for all filtered molecules
echo "Computing ECR for all filtered molecules..."
python3 ccr2_gen/docking/plip_analysis/prepare_redock_top200.py \
    --vina-tsv ccr2_gen/docking/analysis_vina/docking_results_merged.tsv \
    --glide-tsv ccr2_gen/docking/analysis_glide_sp/docking_results_merged.tsv \
    --out-dir "${BASE}/ecr_all" \
    --top-n 99999 --mw-cutoff 546 --max-rings 5 --chunk-size 99999

echo ""
echo "Scoring three tracks..."
cd ccr2_gen/docking/plip_analysis
python3 score_three_tracks.py \
    --vina-profiles "${BASE}/vina_profiles_v2" \
    --glide-profiles "${BASE}/glide_profiles_v2" \
    --vina-redock "${BASE}/vina_merged" \
    --glide-redock "${BASE}/glide_bulk" \
    --ecr-tsv "${BASE}/ecr_all/ecr_top99999.tsv" \
    --out-dir "${BASE}/three_track_results"

echo ""
echo "=== Scoring complete ($(date)) ==="
echo "  Results: ${BASE}/three_track_results/"
ls -lh "${BASE}/three_track_results/"
