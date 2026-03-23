#!/usr/bin/env python3
"""Extract Glide poses into bulk SDF files (no per-molecule directories).

Replaces extract_glide_poses.py for the optimized pipeline. Instead of
creating 52K directories with PDB complexes, writes ligand poses to bulk
SDF files (one per batch of ~1000 molecules). PLIP then reads these
directly via run_plip_bulk.py.

Output per batch:
  - glide_batch_NN_poses.sdf  — best Glide pose per molecule (ligand only)
  - glide_batch_NN_results.tsv — Glide scores

Usage:
    python extract_glide_bulk.py \
        --glide-results-dir ../results/glide_sp \
        --molecule-tsv filtered_molecules.tsv \
        --out-dir glide_bulk \
        --batch-size 1000
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glide-results-dir", type=Path, required=True,
                   help="Directory with glide_chunk_*.csv files")
    p.add_argument("--molecule-tsv", type=Path, required=True,
                   help="TSV with dock_id column (filtered molecules)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=1000)
    args = p.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load target IDs
    mol_df = pd.read_csv(args.molecule_tsv, sep="\t")
    target_ids = set(mol_df["dock_id"])
    print(f"Target molecules: {len(target_ids)}")

    # Scan all Glide chunks, collect best pose per molecule
    chunk_files = sorted(args.glide_results_dir.resolve().glob("glide_chunk_*.csv"))
    print(f"Scanning {len(chunk_files)} Glide chunk files...")

    best_poses = {}  # dock_id → {score, molblock, smiles}
    for i, f in enumerate(chunk_files):
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if "MOL_ID" not in df.columns or "MolBlock" not in df.columns:
            continue

        df = df[df["MOL_ID"].isin(target_ids)].copy()
        if len(df) == 0:
            continue

        df = df.sort_values("r_i_docking_score").drop_duplicates(
            subset=["MOL_ID"], keep="first")

        for _, row in df.iterrows():
            mol_id = row["MOL_ID"]
            score = float(row["r_i_docking_score"])
            if mol_id in best_poses and best_poses[mol_id]["score"] <= score:
                continue
            best_poses[mol_id] = {
                "score": score,
                "molblock": row["MolBlock"],
                "smiles": row.get("SMILES", ""),
            }

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(chunk_files)} chunks, {len(best_poses)} poses collected")

    print(f"Total poses: {len(best_poses)}")

    # Write in batches
    sorted_ids = sorted(best_poses.keys())
    n_batches = (len(sorted_ids) + args.batch_size - 1) // args.batch_size

    for bi in range(n_batches):
        start = bi * args.batch_size
        end = min(start + args.batch_size, len(sorted_ids))
        batch_ids = sorted_ids[start:end]

        sdf_path = out_dir / f"glide_batch_{bi+1:03d}_poses.sdf"
        tsv_path = out_dir / f"glide_batch_{bi+1:03d}_results.tsv"

        writer = Chem.SDWriter(str(sdf_path))
        results = []

        for mol_id in batch_ids:
            data = best_poses[mol_id]
            mol = Chem.MolFromMolBlock(data["molblock"], removeHs=True, sanitize=True)
            if mol is None:
                mol = Chem.MolFromMolBlock(data["molblock"], removeHs=True, sanitize=False)

            if mol is not None:
                mol.SetProp("_Name", mol_id)
                mol.SetProp("dock_id", mol_id)
                mol.SetProp("glide_score", f"{data['score']:.3f}")
                mol.SetProp("smiles", data["smiles"])
                writer.write(mol)
                results.append({"mol_id": mol_id, "smiles": data["smiles"],
                                "status": "success", "best_score": data["score"]})
            else:
                results.append({"mol_id": mol_id, "smiles": data["smiles"],
                                "status": "failed", "best_score": data["score"]})

        writer.close()

        with open(tsv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["mol_id", "smiles", "status", "best_score"],
                               delimiter="\t")
            w.writeheader()
            w.writerows(results)

    n_ok = sum(1 for r in best_poses.values())
    print(f"\nWrote {n_batches} batches to {out_dir}")
    print(f"  {n_ok} molecules, {n_batches} SDF files, {n_batches} TSV files")


if __name__ == "__main__":
    main()
