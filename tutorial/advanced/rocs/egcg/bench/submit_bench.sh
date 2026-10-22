#!/bin/bash
#PBS -N egcg_bench
#PBS -q cpu
#PBS -l select=1:ncpus=190:mem=120gb:feature=cpu
#PBS -l walltime=03:00:00
#PBS -o /home/fulopj/drx-run/DrugEx/.worktrees/egcg/tutorial/advanced/rocs/egcg/bench/
#PBS -e /home/fulopj/drx-run/DrugEx/.worktrees/egcg/tutorial/advanced/rocs/egcg/bench/
set -euo pipefail
WORKTREE=/home/fulopj/drx-run/DrugEx/.worktrees/egcg
export PYTHONPATH="$WORKTREE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="/scratch/$USER/egcg_bench.${PBS_JOBID:-local}"
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
cd "$WORKTREE/tutorial/advanced/rocs/egcg"
echo "host $(hostname) | ncpus $(nproc)"
/home/fulopj/miniconda3/envs/drugex/bin/python bench/bench_njobs.py 64
