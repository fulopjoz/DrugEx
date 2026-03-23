#!/bin/bash
#PBS -N plip_vi_v2
#PBS -q cpu
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=2:00:00
#PBS -J 1-53
#PBS -o /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/full_pipeline/
#PBS -e /home/fulopj/drx-run/DrugEx/ccr2_gen/logs/full_pipeline/
#
# V2 PLIP on Vina poses — reads from per-molecule dirs but writes JSONL.
# Avoids the glob copy problem of v1 (37/53 batches failed to copy 1000 JSON files).
# Each batch scans vina_merged/ for complex PDB files, takes a slice, writes 1 JSONL.

set -eo pipefail

DRUGEX_DIR="/home/fulopj/drx-run/DrugEx"
CONDA_ROOT="/home/fulopj/miniconda3"
CONDA_ENV="drugex"

VINA_MERGED="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/full_pipeline/vina_merged"
OUT_DIR="${DRUGEX_DIR}/ccr2_gen/docking/plip_analysis/full_pipeline/vina_profiles_v2"

set +eu
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -eu

cd "${DRUGEX_DIR}"
export PYTHONPATH="${DRUGEX_DIR}:${PYTHONPATH:-}"

mkdir -p "${OUT_DIR}"

echo "=== PLIP Vina v2 batch ${PBS_ARRAY_INDEX} ($(date)) on $(hostname) ==="

# run_plip_batch.py but output to JSONL instead of per-molecule JSON
python3 -c "
import json, sys, tempfile
from pathlib import Path

sys.path.insert(0, 'ccr2_gen/docking/plip_analysis')
from run_plip import run_plip_on_complex, score_interactions

merged = Path('${VINA_MERGED}')
out_jsonl = Path('${OUT_DIR}/vina_batch_$(printf '%03d' ${PBS_ARRAY_INDEX})_profiles.jsonl')

all_complexes = sorted(merged.rglob('*_complex.pdb'))
total = len(all_complexes)
batch_size = 1000
start = (${PBS_ARRAY_INDEX} - 1) * batch_size
end = min(start + batch_size, total)
batch = all_complexes[start:end]

print(f'Batch ${PBS_ARRAY_INDEX}: {start+1}-{end} of {total} ({len(batch)} molecules)')

n_ok = n_fail = 0
with open(out_jsonl, 'w') as out:
    for pdb_path in batch:
        mol_id = pdb_path.stem.replace('_complex', '')
        try:
            interactions = run_plip_on_complex(pdb_path)
            scores = score_interactions(interactions)
            record = {'mol_id': mol_id, 'interactions': interactions, 'scores': scores}
            out.write(json.dumps(record, default=str) + '\n')
            n_ok += 1
            if n_ok % 200 == 0:
                print(f'  [{n_ok + n_fail}] {n_ok} ok, {n_fail} failed')
        except Exception as e:
            n_fail += 1
            if n_fail <= 3:
                print(f'  {mol_id}: FAILED - {e}')

print(f'Done: {n_ok} ok, {n_fail} failed -> {out_jsonl.name}')
"

echo "=== Batch ${PBS_ARRAY_INDEX} done ($(date)) ==="
