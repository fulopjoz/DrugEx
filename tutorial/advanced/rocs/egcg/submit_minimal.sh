#!/bin/bash
#PBS -N egcg_minimal
#PBS -q cpu
#PBS -l select=1:ncpus=32:mem=48gb:feature=cpu
#PBS -l walltime=48:00:00
#PBS -o /home/fulopj/drx-run/DrugEx/.worktrees/egcg/tutorial/advanced/rocs/egcg/logs/
#PBS -e /home/fulopj/drx-run/DrugEx/.worktrees/egcg/tutorial/advanced/rocs/egcg/logs/

# Minimal EGCG adaptation of rocs_rl_tutorial.ipynb.
#
# 32 cores, NOT 190: benchmark job 14059 measured 64 molecules at n_jobs
# 16/64/190 in 172.5/170.6/171.7 s -- no gain beyond ~16 workers, because
# conformer generation is serial across molecules in the main process and is
# ~86% of the runtime. Only shape/colour alignment uses the scorer pool.

set -euo pipefail
WORKTREE=/home/fulopj/drx-run/DrugEx/.worktrees/egcg
EXPDIR="$WORKTREE/tutorial/advanced/rocs/egcg"
mkdir -p "$EXPDIR/logs"

export PYTHONPATH="$WORKTREE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="/scratch/$USER/egcg_minimal.${PBS_JOBID:-local}"
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT

cd "$EXPDIR"
echo "host $(hostname) | commit $(git -C "$WORKTREE" rev-parse --short HEAD)"
STAMP=$(date +%Y%m%d_%H%M%S)

/home/fulopj/miniconda3/envs/drugex/bin/python -m jupyter nbconvert \
    --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=egcg-drugex \
    --output "executed_minimal_${STAMP}.ipynb" \
    --output-dir "$EXPDIR/minimal_out" \
    egcg_rocs_minimal.ipynb

echo "done: $EXPDIR/minimal_out/executed_minimal_${STAMP}.ipynb"
