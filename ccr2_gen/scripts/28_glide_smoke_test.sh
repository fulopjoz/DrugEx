#!/bin/bash
#PBS -N glide_smoke_test
#PBS -l select=1:ncpus=10:ngpus=0:mem=64gb
#PBS -l walltime=01:00:00
#PBS -m ae
#PBS -q cpu
#
# Glide smoke test: dock cocaine + paracetamol with bundled test_grid.zip.
# Follows the same approach as qsub_chunk_WT.sh.
#
# Submit from ccr2_gen/schrodtainer/:
#   cd ccr2_gen/schrodtainer
#   qsub ../scripts/28_glide_smoke_test.sh

set -euo pipefail

cd $PBS_O_WORKDIR

# --- Setup scratch ---
SCRATCHDIR=/scratch/$USER/$PBS_JOBID
mkdir -p "$SCRATCHDIR"

# Copy glide.py and run_glide.py to scratch
cp images/glide/src/glide.py "$SCRATCHDIR/glide.py"
cp images/glide/src/run_glide.py "$SCRATCHDIR/run_glide.py"

# Apptainer image — same location as qsub_chunk_WT.sh
APPTAINER_IMAGE_PATH=$(pwd)/images/glide.sif

# Test grid bundled with schrodtainer
GLIDE_GRID_FILE=$(pwd)/images/glide/src/test_grid.zip

# Output
DATADIR=$(pwd)
OUTDIR=$DATADIR/outputs
mkdir -p "$OUTDIR"

echo "=== Glide Smoke Test ==="
echo "PBS Job ID:  $PBS_JOBID"
echo "Node:        $(hostname -f)"
echo "Image:       $APPTAINER_IMAGE_PATH"
echo "Grid:        $GLIDE_GRID_FILE"
echo "Scratch:     $SCRATCHDIR"
echo "========================"

# --- Validate ---
if [[ ! -f "$APPTAINER_IMAGE_PATH" ]]; then
    echo "ERROR: glide.sif not found at $APPTAINER_IMAGE_PATH" >&2
    echo "Listing images/:" >&2
    ls -la images/ >&2
    ls -la images/glide/ >&2
    exit 1
fi

if [[ ! -f "$GLIDE_GRID_FILE" ]]; then
    echo "ERROR: test_grid.zip not found at $GLIDE_GRID_FILE" >&2
    exit 1
fi

# Copy grid to scratch
cp -v "$GLIDE_GRID_FILE" "$SCRATCHDIR/" || { echo >&2 "Error copying grid!"; exit 2; }

cd "$SCRATCHDIR"

echo "$PBS_JOBID smoke_test $(hostname -f) $SCRATCHDIR" >> "$OUTDIR/jobs_info.txt"

# --- Load apptainer ---
module load apptainer

GRID_BASENAME=$(basename "$GLIDE_GRID_FILE")
GRID_IN_SCRATCH="$SCRATCHDIR/$GRID_BASENAME"

# --- Run Glide smoke test: cocaine + paracetamol via --smiles/--ids ---
echo "Running Glide smoke test (cocaine + paracetamol)..."

apptainer run --writable-tmpfs \
  --bind "$SCRATCHDIR":/home/$USER/:rw \
  --bind "$GRID_IN_SCRATCH":/tmp/"$GRID_BASENAME":ro \
  --bind "$SCRATCHDIR/glide.py":/opt/glide/src/glide.py:ro \
  --bind "$SCRATCHDIR/run_glide.py":/opt/glide/src/run_glide.py:ro \
  --pwd /home/$USER \
  --network=none \
  --cleanenv \
  --contain \
  --env USER=$USER \
  "$APPTAINER_IMAGE_PATH" \
  python /opt/glide/src/run_glide.py \
    --smiles "CN1[C@H]2CC[C@@H]1[C@H]([C@H](C2)OC(=O)C3=CC=CC=C3)C(=O)OC" "CC(=O)NC1=CC=C(C=C1)O" \
    --ids cocaine paracetamol \
    --workers 2 \
    --chunk-size 1 \
    --precision SP \
    --run-name glide_smoke_test \
    --grid-file /tmp/"$GRID_BASENAME"

# --- Copy results ---
mkdir -p "$OUTDIR/$PBS_JOBID"
cp -v "$SCRATCHDIR/glide_smoke_test.csv" "$OUTDIR/$PBS_JOBID/" || {
    echo >&2 "Result CSV not found!"
    ls -la "$SCRATCHDIR/"
    exit 4
}

echo ""
echo "=== Results ==="
cat "$OUTDIR/$PBS_JOBID/glide_smoke_test.csv"
echo ""
echo "=== Glide smoke test complete ==="

# Cleanup
rm -rf "$SCRATCHDIR"/*
