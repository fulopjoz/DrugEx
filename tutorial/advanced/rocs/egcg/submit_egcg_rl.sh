#!/bin/bash
#PBS -N egcg_rl
#PBS -q cpu
#PBS -l select=1:ncpus=190:mem=120gb:feature=cpu
#PBS -l walltime=48:00:00
#PBS -o /home/fulopj/drx-run/DrugEx/.worktrees/egcg/tutorial/advanced/rocs/egcg/logs/
#PBS -e /home/fulopj/drx-run/DrugEx/.worktrees/egcg/tutorial/advanced/rocs/egcg/logs/

# ============================================================================
# EGCG ligand-based shape-matching RL pilot
#
# Single sequential job (replication across seeds is deferred -- spec 2).
#
# WHY CPU, NOT GPU (decided 2026-10-20 against the live cluster):
#   The bottleneck is RDKit conformer embedding + shape/colour alignment --
#   pure CPU, embarrassingly parallel across molecules via the scorer's
#   process pool. The GPU-eligible part is RNN sampling (RL_N_SAMPLES
#   sequences per epoch), which is a negligible fraction of runtime. A GPU
#   would accelerate ~1% of the work.
#   The emcpu-2..9 nodes carry 192 cores each and were all free; the GPU nodes
#   (in-6..18) carry only 48 (in-18: 32), are saturated by other users, and
#   several are offline. lich-compute-emgpu has 192 cores AND GPUs, but it is
#   the only such node, so a CPU-only job should not occupy it.
#   ncpus=190 of 192 leaves 2 CPUs free for interactive SSH (standing rule).
#
# PBS WALLTIME CANNOT BE EXTENDED. It is requested here, up front. The
# notebook's cost projection can only decide whether to proceed WITHIN this
# allocation -- if the projection exceeds EGCG_BUDGET_HOURS it raises and the
# job stops early rather than being killed mid-RL with nothing written.
# ============================================================================

set -euo pipefail

WORKTREE=/home/fulopj/drx-run/DrugEx/.worktrees/egcg
EXPDIR="$WORKTREE/tutorial/advanced/rocs/egcg"
mkdir -p "$EXPDIR/logs"

# --- resolve the CPU allocation --------------------------------------------
# $PBS_NODEFILE has one line per NODE on this cluster, not per CPU, so query
# PBS directly and fall back to nproc (which is cgroup-limited to this job).
if [ -z "${PBS_NCPUS:-}" ] || [ "${PBS_NCPUS:-0}" -le 1 ]; then
    PBS_NCPUS=""
    if [ -n "${PBS_RESOURCE_LIST_ncpus:-}" ]; then
        PBS_NCPUS="$PBS_RESOURCE_LIST_ncpus"
    elif [ -n "${PBS_RESOURCE_LIST_select:-}" ]; then
        PBS_NCPUS=$(printf '%s' "$PBS_RESOURCE_LIST_select" \
            | awk -F: '{for (i=1;i<=NF;i++) if ($i ~ /^ncpus=/){split($i,a,"="); print a[2]; exit}}')
    elif command -v qstat >/dev/null 2>&1 && [ -n "${PBS_JOBID:-}" ]; then
        PBS_NCPUS=$(qstat -f "$PBS_JOBID" 2>/dev/null \
            | awk -F' = ' '/Resource_List.ncpus/{v=$2} END{if (v!="") print v}') || PBS_NCPUS=""
    fi
    [ -z "${PBS_NCPUS:-}" ] && PBS_NCPUS=$(nproc 2>/dev/null || echo 1)
    export PBS_NCPUS
fi

# --- thread caps (oversubscription guard, spec 15) -------------------------
# Conformer generation is single-threaded in config_egcg.py; parallelism comes
# from the scorer's process pool only. Without these caps each of the N scorer
# workers would also spawn N BLAS threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- Gate 0 depends on this: pin the worktree ahead of the editable install --
# The conda env has drugex installed editable against the MAIN CHECKOUT, which
# is on a different branch. PYTHONPATH wins over the editable finder (verified),
# and the notebook asserts the resolved path regardless.
export PYTHONPATH="$WORKTREE"

# --- scratch for working files ---------------------------------------------
export SCRATCH_DIR="/scratch/$USER/egcg_rl.${PBS_JOBID:-local}"
mkdir -p "$SCRATCH_DIR"
export TMPDIR="$SCRATCH_DIR"
trap 'rm -rf "$SCRATCH_DIR"' EXIT

# Stop early rather than overrun the allocation (leave headroom for eval+plots).
export EGCG_BUDGET_HOURS="${EGCG_BUDGET_HOURS:-40}"

PY=/home/fulopj/miniconda3/envs/drugex/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)

cd "$EXPDIR"

echo "=========================================================="
echo "job        : ${PBS_JOBID:-local}"
echo "host       : $(hostname)"
echo "ncpus      : $PBS_NCPUS"
echo "worktree   : $WORKTREE"
echo "commit     : $(git -C "$WORKTREE" rev-parse --short HEAD)"
echo "scratch    : $SCRATCH_DIR"
echo "budget     : ${EGCG_BUDGET_HOURS} h"
echo "=========================================================="

# --- reference must exist and pass Gate 1 before the notebook runs ----------
"$PY" build_reference.py --verify

# --- execute the notebook in place, keeping outputs ------------------------
# papermill is not installed in this env, so parameters come from
# config_egcg.py and the environment, not from injected parameter cells.
"$PY" -m jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=egcg-drugex \
    --output "executed_egcg_rl_${STAMP}.ipynb" \
    --output-dir "$EXPDIR/runs" \
    egcg_rl.ipynb

echo "=========================================================="
echo "done: $EXPDIR/runs/executed_egcg_rl_${STAMP}.ipynb"
ls -la "$EXPDIR/runs" "$EXPDIR/panels" 2>/dev/null || true
