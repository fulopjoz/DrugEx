#!/usr/bin/env python3
"""Run PLIP on a batch of molecules (subset of a larger set).

Designed for PBS array parallelization. Each batch processes molecules
from a specific range of the alphabetically sorted complex PDB files.

Usage:
    python run_plip_batch.py \
        --redock-dir merged_results \
        --out-dir profiles_batch_01 \
        --batch-index 1 \
        --batch-size 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_plip import run_plip_on_complex, score_interactions


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--redock-dir", type=Path, required=True,
                   help="Directory with *_complex.pdb files")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for this batch's profiles")
    p.add_argument("--batch-index", type=int, required=True,
                   help="1-based batch index")
    p.add_argument("--batch-size", type=int, default=1000,
                   help="Molecules per batch (default: 1000)")
    args = p.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find ALL complex files, sort, take this batch's slice
    all_complexes = sorted(args.redock_dir.resolve().rglob("*_complex.pdb"))
    total = len(all_complexes)

    start = (args.batch_index - 1) * args.batch_size
    end = min(start + args.batch_size, total)
    batch = all_complexes[start:end]

    print(f"PLIP batch {args.batch_index}: molecules {start+1}-{end} of {total}")
    print(f"  Processing {len(batch)} complexes")

    n_ok = 0
    n_fail = 0
    for i, pdb_path in enumerate(batch):
        mol_id = pdb_path.stem.replace("_complex", "")
        try:
            interactions = run_plip_on_complex(pdb_path)
            scores = score_interactions(interactions)
            prof = {"interactions": interactions, "scores": scores}

            prof_path = out_dir / f"{mol_id}_interactions.json"
            prof_path.write_text(json.dumps(prof, indent=2, default=str))
            n_ok += 1

            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{len(batch)}] {n_ok} ok, {n_fail} failed")
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"  {mol_id}: FAILED - {e}")

    print(f"\nBatch {args.batch_index} complete: {n_ok} ok, {n_fail} failed")


if __name__ == "__main__":
    main()
