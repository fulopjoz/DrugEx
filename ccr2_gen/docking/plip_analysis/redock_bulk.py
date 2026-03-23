#!/usr/bin/env python3
"""Vina docking with bulk SDF output (no per-molecule directories).

Replaces redock_top10.py for large-scale runs. Instead of creating one
directory per molecule (52K dirs = NFS nightmare), writes two flat files:
  - chunk_NNN_poses.sdf   — best docked pose per molecule (ligand only)
  - chunk_NNN_results.tsv — docking scores and status

PLIP can then create protein-ligand complexes in memory from the SDF.

Usage:
    python redock_bulk.py \
        --input-tsv chunks/chunk_001.tsv \
        --receptor-pdbqt receptor.pdbqt \
        --out-prefix results/chunk_001 \
        --exhaustiveness 32 --n-poses 20
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BINDING_SITE


def dock_molecule(smiles: str, mol_id: str, receptor_pdbqt: str,
                  exhaustiveness: int, n_poses: int, seed: int = 42) -> dict:
    """Dock one molecule, return result dict with best pose RDKit mol."""
    result = {"mol_id": mol_id, "smiles": smiles, "status": "failed",
              "best_score": None, "n_poses": 0, "pose_mol": None}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        result["error"] = "Invalid SMILES"
        return result

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) < 0:
        result["error"] = "Embedding failed"
        return result
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    try:
        import meeko
        prep = meeko.MoleculePreparation(hydrate=True)
        prep.prepare(mol)
        pdbqt_str = prep.write_pdbqt_string()
    except Exception as e:
        result["error"] = f"Meeko: {e}"
        return result

    try:
        import vina
        v = vina.Vina(sf_name="vina", seed=seed)
        v.set_receptor(receptor_pdbqt)
        v.set_ligand_from_string(pdbqt_str)
        v.compute_vina_maps(center=BINDING_SITE["center"],
                            box_size=BINDING_SITE["box_size"])
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

        energies = v.energies()
        result["best_score"] = float(energies[0][0]) if len(energies) > 0 else None
        result["n_poses"] = len(energies)

        # Get best pose as RDKit mol
        poses_pdbqt = v.poses(n_poses=1)
        pmol = meeko.PDBQTMolecule(poses_pdbqt)
        rdkit_mol = meeko.RDKitMolCreate.from_pdbqt_mol(pmol)[0]
        rdkit_mol = Chem.RemoveHs(rdkit_mol)
        result["pose_mol"] = rdkit_mol
        result["status"] = "success"

    except Exception as e:
        result["error"] = f"Vina: {e}"

    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-tsv", type=Path, required=True)
    p.add_argument("--receptor-pdbqt", type=Path, required=True)
    p.add_argument("--out-prefix", type=str, required=True,
                   help="Output prefix: produces {prefix}_poses.sdf + {prefix}_results.tsv")
    p.add_argument("--exhaustiveness", type=int, default=32)
    p.add_argument("--n-poses", type=int, default=20)
    p.add_argument("--max-molecules", type=int, default=None)
    args = p.parse_args()

    out_sdf = Path(f"{args.out_prefix}_poses.sdf")
    out_tsv = Path(f"{args.out_prefix}_results.tsv")
    out_sdf.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_tsv, sep="\t")
    if args.max_molecules:
        df = df.head(args.max_molecules)

    smiles_col = next(c for c in ("std_SMILES", "SMILES", "smiles") if c in df.columns)
    id_col = "dock_id"

    writer = Chem.SDWriter(str(out_sdf))
    results = []

    for _, row in df.iterrows():
        mol_id = row[id_col]
        smiles = row[smiles_col]
        r = dock_molecule(smiles, mol_id, str(args.receptor_pdbqt),
                          args.exhaustiveness, args.n_poses)

        if r["pose_mol"] is not None:
            pose = r["pose_mol"]
            pose.SetProp("_Name", mol_id)
            pose.SetProp("dock_id", mol_id)
            pose.SetProp("vina_score", f"{r['best_score']:.3f}")
            pose.SetProp("smiles", smiles)
            writer.write(pose)

        results.append({
            "mol_id": mol_id, "smiles": smiles,
            "status": r["status"], "best_score": r.get("best_score"),
            "n_poses": r["n_poses"], "error": r.get("error", ""),
        })

        if r["status"] == "success":
            print(f"  {mol_id}: {r['best_score']:.2f} ({r['n_poses']} poses)")
        else:
            print(f"  {mol_id}: FAILED - {r.get('error', '?')}")

    writer.close()

    with open(out_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mol_id", "smiles", "status",
                                           "best_score", "n_poses", "error"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(results)

    n_ok = sum(1 for r in results if r["status"] == "success")
    print(f"\nDone: {n_ok}/{len(results)} success → {out_sdf.name}, {out_tsv.name}")


if __name__ == "__main__":
    main()
