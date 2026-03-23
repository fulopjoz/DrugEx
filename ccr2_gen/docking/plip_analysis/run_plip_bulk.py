#!/usr/bin/env python3
"""PLIP analysis reading from bulk SDF files (no per-molecule directories).

Replaces run_plip_batch.py for the optimized pipeline. Instead of scanning
52K directories for *_complex.pdb files, reads ligand poses from a bulk SDF
and combines with the receptor in memory to create complexes for PLIP.

Output: JSONL file (one JSON per line) instead of one file per molecule.
At 52K molecules this produces ~53 JSONL files instead of 52K JSON files.

Usage:
    python run_plip_bulk.py \
        --poses-sdf chunk_001_poses.sdf \
        --receptor-pdb receptor.pdb \
        --out-jsonl profiles_chunk_001.jsonl
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_plip import run_plip_on_complex, score_interactions


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--poses-sdf", type=Path, required=True,
                   help="Bulk SDF with docked ligand poses (from redock_bulk.py or extract_glide_bulk.py)")
    p.add_argument("--receptor-pdb", type=Path, required=True)
    p.add_argument("--out-jsonl", type=Path, required=True,
                   help="Output JSONL file (one JSON per line per molecule)")
    args = p.parse_args()

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Load receptor ONCE
    print(f"Loading receptor: {args.receptor_pdb.name}")
    protein = Chem.MolFromPDBFile(str(args.receptor_pdb), removeHs=False, sanitize=False)
    if protein is None:
        raise RuntimeError(f"Failed to load receptor: {args.receptor_pdb}")

    # Read poses from SDF
    suppl = Chem.SDMolSupplier(str(args.poses_sdf), removeHs=True)
    n_ok = 0
    n_fail = 0

    with open(args.out_jsonl, "w") as out:
        for mol in suppl:
            if mol is None:
                n_fail += 1
                continue

            mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{n_ok + n_fail}"
            dock_id = mol.GetProp("dock_id") if mol.HasProp("dock_id") else mol_id

            try:
                # Combine protein + ligand in memory
                complex_mol = Chem.CombineMols(protein, mol)

                # Write to temp PDB for PLIP (PLIP needs a file path)
                with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tmp:
                    tmp_path = Path(tmp.name)
                    Chem.MolToPDBFile(complex_mol, str(tmp_path), flavor=4)

                # Run PLIP
                interactions = run_plip_on_complex(tmp_path)
                scores = score_interactions(interactions)

                # Write one JSON line
                record = {
                    "mol_id": dock_id,
                    "interactions": interactions,
                    "scores": scores,
                }
                out.write(json.dumps(record, default=str) + "\n")

                # Clean up temp file immediately
                tmp_path.unlink(missing_ok=True)

                n_ok += 1
                if n_ok % 100 == 0:
                    print(f"  [{n_ok + n_fail}] {n_ok} ok, {n_fail} failed")

            except Exception as e:
                n_fail += 1
                if n_fail <= 5:
                    print(f"  {dock_id}: FAILED - {e}")
                # Clean up temp on error too
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    print(f"\nDone: {n_ok} ok, {n_fail} failed → {args.out_jsonl.name}")


if __name__ == "__main__":
    main()
