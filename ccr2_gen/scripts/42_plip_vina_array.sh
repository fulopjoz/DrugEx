#!/bin/bash
#PBS -N plip_vina
#PBS -q cpu
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=2:00:00
#PBS -J 1-53
#PBS -o /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/full_pipeline/
#PBS -e /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/full_pipeline/
#
# PLIP on Vina poses: 53 batches × 1000 molecules. %10 concurrent.
# Each batch takes ~18 min (1000 × 1.1 sec). Total: ~6 waves × 18 min = ~2h.
# Run AFTER 41_merge_extract.sh completes.

set -eo pipefail

DRUGEX_DIR="/home/fulopj/drx-run/DrugEx"
CONDA_ROOT="/home/fulopj/miniconda3"
CONDA_ENV="drugex"

VINA_MERGED="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/full_pipeline/vina_merged"
FINAL_PROFILES="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/full_pipeline/vina_profiles"

SCRATCHDIR="/scratch/$USER/${PBS_JOBID}"
mkdir -p "${SCRATCHDIR}/profiles"

cleanup() {
    if [ -d "${SCRATCHDIR}/profiles" ]; then
        mkdir -p "${FINAL_PROFILES}"
        cp "${SCRATCHDIR}/profiles/"*_interactions.json "${FINAL_PROFILES}/" 2>/dev/null \
            || echo "WARNING: profile copy failed! Files at $(hostname):${SCRATCHDIR}/profiles"
    fi
    rm -rf "${SCRATCHDIR}"
}
trap cleanup EXIT

set +eu
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -eu

cd "${DRUGEX_DIR}"
export PYTHONPATH="${DRUGEX_DIR}:${PYTHONPATH:-}"

echo "=== PLIP Vina batch ${PBS_ARRAY_INDEX} ($(date)) on $(hostname) ==="

python3 ccr2_gen/docking/plip_analysis/run_plip_batch.py \
    --redock-dir "${VINA_MERGED}" \
    --out-dir "${SCRATCHDIR}/profiles" \
    --batch-index ${PBS_ARRAY_INDEX} \
    --batch-size 1000

echo "=== Batch ${PBS_ARRAY_INDEX} done ($(date)) ==="
