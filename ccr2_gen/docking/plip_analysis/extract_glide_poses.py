#!/usr/bin/env python3
"""Extract best Glide SP poses from raw CSV files and create PDB complexes for PLIP.

The original Glide docking saves MolBlock (3D SDF) for every pose in the chunk CSV.
This script:
  1. Loads all Glide chunk CSVs
  2. Deduplicates (keeps best r_i_docking_score per molecule)
  3. Filters to requested molecule IDs (from ECR top-N list)
  4. Parses MolBlock → RDKit mol
  5. Combines with receptor PDB → protein-ligand complex
  6. Writes PDB complex per molecule (same format as redock_top10.py output)

Output structure matches Vina re-dock output for PLIP compatibility:
  out_dir/
    CCR2_GEN_XXXXXX/
      CCR2_GEN_XXXXXX_complex.pdb
      CCR2_GEN_XXXXXX_poses.sdf  (best Glide pose)
    redock_results.tsv

Usage:
    python extract_glide_poses.py \
        --glide-results-dir ../results/glide_sp \
        --ecr-tsv redock_top10k/ecr_top10000.tsv \
        --receptor-pdb ../receptor/5T1A_clean_mutations_reversed_withHs.pdb \
        --out-dir redock_top10k/glide_poses
"""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)


def load_glide_best_poses(results_dir: Path, target_ids: set[str]) -> dict[str, dict]:
    """Load best Glide pose per molecule from raw chunk CSVs.

    Only loads molecules in target_ids to save memory.
    Returns {dock_id: {score, molblock, smiles_pose}}.
    """
    best = {}
    chunk_files = sorted(results_dir.glob("glide_chunk_*.csv"))
    print(f"  Loading {len(chunk_files)} Glide chunk files...")

    for i, f in enumerate(chunk_files):
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as e:
            print(f"    WARNING: {f.name}: {e}")
            continue

        if "MOL_ID" not in df.columns or "r_i_docking_score" not in df.columns:
            continue
        if "MolBlock" not in df.columns:
            print(f"    WARNING: {f.name} has no MolBlock column, skipping")
            continue

        # Filter to target molecules
        df = df[df["MOL_ID"].isin(target_ids)].copy()
        if len(df) == 0:
            continue

        # Keep best score per molecule (lower = better)
        df = df.sort_values("r_i_docking_score").drop_duplicates(
            subset=["MOL_ID"], keep="first"
        )

        for _, row in df.iterrows():
            mol_id = row["MOL_ID"]
            score = float(row["r_i_docking_score"])
            # Only keep if better than existing
            if mol_id in best and best[mol_id]["score"] <= score:
                continue
            best[mol_id] = {
                "score": score,
                "molblock": row["MolBlock"],
                "smiles_pose": row.get("SMILES_pose", ""),
                "smiles": row.get("SMILES", ""),
            }

        if (i + 1) % 20 == 0:
            print(f"    Processed {i + 1}/{len(chunk_files)} chunks, "
                  f"{len(best)} molecules collected")

    print(f"  Total: {len(best)} molecules with Glide poses")
    return best


def create_complexes(
    poses: dict[str, dict],
    receptor_pdb: Path,
    out_dir: Path,
) -> list[dict]:
    """Create PDB complexes from Glide poses + receptor."""
    print(f"  Loading receptor: {receptor_pdb.name}")
    protein = Chem.MolFromPDBFile(str(receptor_pdb), removeHs=False, sanitize=False)
    if protein is None:
        raise RuntimeError(f"Failed to load receptor: {receptor_pdb}")

    results = []
    n_ok = 0
    n_fail = 0

    for mol_id, data in sorted(poses.items()):
        mol_dir = out_dir / mol_id
        mol_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "mol_id": mol_id,
            "smiles": data.get("smiles", ""),
            "status": "failed",
            "best_score": data["score"],
            "n_poses": 1,
        }

        try:
            # Parse MolBlock → RDKit mol
            mol = Chem.MolFromMolBlock(data["molblock"], removeHs=True, sanitize=True)
            if mol is None:
                # Try without sanitize
                mol = Chem.MolFromMolBlock(data["molblock"], removeHs=True, sanitize=False)
            if mol is None:
                result["error"] = "MolBlock parse failed"
                n_fail += 1
                results.append(result)
                continue

            # Save pose SDF
            sdf_path = mol_dir / f"{mol_id}_poses.sdf"
            writer = Chem.SDWriter(str(sdf_path))
            mol.SetProp("glide_score", f"{data['score']:.3f}")
            mol.SetProp("_Name", f"{mol_id}_pose0")
            writer.write(mol)
            writer.close()

            # Create complex: protein + ligand
            complex_mol = Chem.CombineMols(protein, mol)
            complex_path = mol_dir / f"{mol_id}_complex.pdb"
            Chem.MolToPDBFile(complex_mol, str(complex_path), flavor=4)

            result["status"] = "success"
            result["complex_pdb"] = str(complex_path)
            n_ok += 1

        except Exception as e:
            result["error"] = str(e)
            n_fail += 1

        results.append(result)

        if (n_ok + n_fail) % 500 == 0:
            print(f"    Processed {n_ok + n_fail}/{len(poses)}: "
                  f"{n_ok} ok, {n_fail} failed")

    print(f"  Complexes: {n_ok} success, {n_fail} failed out of {len(poses)}")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glide-results-dir", type=Path, required=True,
                   help="Directory with glide_chunk_*.csv files")
    p.add_argument("--ecr-tsv", type=Path, required=True,
                   help="ECR ranked list (from prepare_redock_top200.py)")
    p.add_argument("--receptor-pdb", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-molecules", type=int, default=None)
    args = p.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load target molecule IDs
    ecr_df = pd.read_csv(args.ecr_tsv, sep="\t")
    if args.max_molecules:
        ecr_df = ecr_df.head(args.max_molecules)
    target_ids = set(ecr_df["dock_id"])
    print(f"Target molecules: {len(target_ids)}")

    # Extract best Glide poses
    print("\nExtracting Glide poses...")
    poses = load_glide_best_poses(args.glide_results_dir.resolve(), target_ids)

    # Report coverage
    found = target_ids & set(poses.keys())
    missing = target_ids - set(poses.keys())
    print(f"\n  Coverage: {len(found)}/{len(target_ids)} "
          f"({len(missing)} missing from Glide output)")

    # Create PDB complexes
    print("\nCreating PDB complexes...")
    results = create_complexes(poses, args.receptor_pdb.resolve(), out_dir)

    # Save results TSV (same format as redock_top10.py)
    results_path = out_dir / "redock_results.tsv"
    with open(results_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mol_id", "smiles", "status",
                                           "best_score", "n_poses"],
                           delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    n_ok = sum(1 for r in results if r["status"] == "success")
    print(f"\nResults: {results_path}")
    print(f"  {n_ok}/{len(results)} complexes created successfully")


if __name__ == "__main__":
    main()
