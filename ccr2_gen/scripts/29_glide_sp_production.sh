#!/bin/bash
#PBS -N ccr2_glide_sp
#PBS -l select=1:ncpus=10:ngpus=0:mem=64gb
#PBS -l walltime=24:00:00
#PBS -m ae
#PBS -q cpu
#
# Glide SP docking of CCR2 generated ligands via Apptainer.
# Runs as a PBS array job: one chunk per array element.
#
# Prerequisites:
#   1. Apptainer image: glide.sif (schrodtainer)
#   2. Glide grid: glide-grid_5T1A_allosteric.zip
#   3. Chunk .smi files from prepare_docking_library.py
#
# Usage:
#   N_CHUNKS=$(ls "$CHUNK_DIR"/chunk_*.smi 2>/dev/null | wc -l)
#   qsub -J 1-${N_CHUNKS} \
#     -v GRID_FILE=/path/to/grid.zip,CHUNK_DIR=/path/to/chunks,RESULTS_DIR=/path/to/results \
#     ccr2_gen/scripts/29_glide_sp_production.sh
#
# Required environment variables (pass via qsub -v):
#   GRID_FILE       -- Absolute path to Glide grid .zip
#   CHUNK_DIR       -- Absolute path to directory with chunk_NNN.smi files
#   RESULTS_DIR     -- Absolute path for output results
#
# Optional environment variables:
#   APPTAINER_IMAGE -- Path to glide.sif (default: ccr2_gen/schrodtainer/data/images/glide/glide.sif)
#   N_WORKERS       -- Number of parallel Glide workers (default: 10)

set -euo pipefail

# --- Configuration ---
PRECISION="SP"
N_WORKERS="${N_WORKERS:-10}"
GLIDE_CHUNK_SIZE=10
RUN_NAME="ccr2_glide"

# Default Apptainer image location
if [[ -z "${APPTAINER_IMAGE:-}" ]]; then
    APPTAINER_IMAGE="${PBS_O_WORKDIR}/ccr2_gen/schrodtainer/data/images/glide/glide.sif"
fi

# --- Validate inputs ---
if [[ -z "${GRID_FILE:-}" ]]; then
    echo "ERROR: GRID_FILE not set. Pass via qsub -v GRID_FILE=/path/to/grid.zip" >&2
    exit 1
fi
if [[ -z "${CHUNK_DIR:-}" ]]; then
    echo "ERROR: CHUNK_DIR not set. Pass via qsub -v CHUNK_DIR=/path/to/chunks" >&2
    exit 1
fi
if [[ -z "${RESULTS_DIR:-}" ]]; then
    echo "ERROR: RESULTS_DIR not set. Pass via qsub -v RESULTS_DIR=/path/to/results" >&2
    exit 1
fi

# --- Derive chunk file from PBS array index ---
CHUNK_IDX=$(printf "%03d" "$PBS_ARRAY_INDEX")
CHUNK_FILE="${CHUNK_DIR}/chunk_${CHUNK_IDX}.smi"

if [[ ! -f "$CHUNK_FILE" ]]; then
    echo "ERROR: Chunk file not found: $CHUNK_FILE" >&2
    exit 1
fi
if [[ ! -f "$GRID_FILE" ]]; then
    echo "ERROR: Grid file not found: $GRID_FILE" >&2
    exit 1
fi
if [[ ! -f "$APPTAINER_IMAGE" ]]; then
    echo "ERROR: Apptainer image not found: $APPTAINER_IMAGE" >&2
    exit 1
fi

echo "=== CCR2 Glide SP Docking ==="
echo "PBS Job ID:    $PBS_JOBID"
echo "Array Index:   $PBS_ARRAY_INDEX"
echo "Chunk File:    $CHUNK_FILE"
echo "Grid File:     $GRID_FILE"
echo "Precision:     $PRECISION"
echo "Results Dir:   $RESULTS_DIR"
echo "Image:         $APPTAINER_IMAGE"
echo "Node:          $(hostname -f)"
echo "Workers:       $N_WORKERS"
echo "======================================="

# --- Setup scratch ---
SCRATCHDIR="/scratch/$USER/$PBS_JOBID"
mkdir -p "$SCRATCHDIR"

# Copy inputs to scratch
cp "$CHUNK_FILE" "$SCRATCHDIR/current_chunk.smi"
cp "$GRID_FILE" "$SCRATCHDIR/"

# Copy glide.py (precision is now a CLI parameter, no sed patching needed)
GLIDE_SRC="${PBS_O_WORKDIR}/ccr2_gen/schrodtainer/images/glide/src/glide.py"
if [[ ! -f "$GLIDE_SRC" ]]; then
    GLIDE_SRC="$(dirname "$PBS_O_WORKDIR")/ccr2_gen/schrodtainer/images/glide/src/glide.py"
fi
cp "$GLIDE_SRC" "$SCRATCHDIR/glide.py"

# Copy run_glide.py (entry point with .smi format support)
RUN_GLIDE_SRC="${PBS_O_WORKDIR}/ccr2_gen/schrodtainer/images/glide/src/run_glide.py"
if [[ ! -f "$RUN_GLIDE_SRC" ]]; then
    RUN_GLIDE_SRC="$(dirname "$PBS_O_WORKDIR")/ccr2_gen/schrodtainer/images/glide/src/run_glide.py"
fi
cp "$RUN_GLIDE_SRC" "$SCRATCHDIR/run_glide.py"

GRID_BASENAME=$(basename "$GRID_FILE")

cd "$SCRATCHDIR"

# Log job info
mkdir -p "$RESULTS_DIR"
echo "$PBS_JOBID chunk_${CHUNK_IDX} ${PRECISION} $(hostname -f) $SCRATCHDIR" >> "$RESULTS_DIR/jobs_info.txt"

# --- Run Glide via Apptainer ---
module load apptainer 2>/dev/null || true

apptainer run --writable-tmpfs \
    --bind "$SCRATCHDIR":/home/$USER/:rw \
    --bind "$SCRATCHDIR/$GRID_BASENAME":/tmp/"$GRID_BASENAME":ro \
    --bind "$SCRATCHDIR/glide.py":/opt/glide/src/glide.py:ro \
    --bind "$SCRATCHDIR/run_glide.py":/opt/glide/src/run_glide.py:ro \
    --pwd /home/$USER \
    --network=none \
    --cleanenv \
    --contain \
    --env USER=$USER \
    "$APPTAINER_IMAGE" \
    python /opt/glide/src/run_glide.py \
        --ligand-csv current_chunk.smi \
        --workers "$N_WORKERS" \
        --chunk-size "$GLIDE_CHUNK_SIZE" \
        --run-name "$RUN_NAME" \
        --precision "$PRECISION" \
        --grid-file /tmp/"$GRID_BASENAME"

# --- Copy results ---
RESULT_FILE="$RESULTS_DIR/glide_chunk_${CHUNK_IDX}.csv"
cp "$SCRATCHDIR/${RUN_NAME}.csv" "$RESULT_FILE" || {
    echo "ERROR: Result file not found at $SCRATCHDIR/${RUN_NAME}.csv" >&2
    ls -la "$SCRATCHDIR/"
    exit 4
}

# Also copy skip file if it exists
if [[ -f "$SCRATCHDIR/${RUN_NAME}_skip.csv" ]]; then
    cp "$SCRATCHDIR/${RUN_NAME}_skip.csv" "$RESULTS_DIR/glide_skip_${CHUNK_IDX}.csv"
fi

echo "Results copied to: $RESULT_FILE"
echo "Molecules in chunk: $(wc -l < "$CHUNK_FILE")"

# --- Cleanup scratch ---
# Note: rm may fail if jobserverd holds file handles; this is cosmetic
rm -rf "$SCRATCHDIR" 2>/dev/null || true

echo "=== Done: chunk_${CHUNK_IDX} (SP) ==="
