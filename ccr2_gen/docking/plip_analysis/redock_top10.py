#!/usr/bin/env python3
"""Phase 5: Re-dock top consensus candidates with higher exhaustiveness.

Takes the top-10 (or top-N) consensus candidates from MW regression analysis,
prepares them with Gypsum-DL (if available) or ETKDGv3, and docks with Vina
at exhaustiveness=32 and n_poses=20.

Creates receptor-ligand complexes for PLIP interaction profiling.

Usage:
  python ccr2_gen/docking/plip_analysis/redock_top10.py \
    --consensus-tsv ccr2_gen/docking/analysis_mw_corrected/top_10_consensus.tsv \
    --receptor-pdbqt ccr2_gen/docking/receptor/5T1A_clean_mutations_reversed_withHs.pdbqt \
    --receptor-pdb ccr2_gen/docking/receptor/5T1A_clean_mutations_reversed_withHs.pdb \
    --out-dir ccr2_gen/docking/plip_analysis/redock_results \
    --exhaustiveness 32 \
    --n-poses 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Docking config
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BINDING_SITE


def prepare_and_dock_molecule(
    smiles: str,
    mol_id: str,
    receptor_pdbqt: Path,
    receptor_pdb: Path,
    out_dir: Path,
    exhaustiveness: int,
    n_poses: int,
    seed: int = 42,
) -> dict:
    """Prepare ligand, dock with Vina, save poses and complexes."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    def extract_pose_conformer(mol: Chem.Mol, conf_id: int) -> Chem.Mol:
        pose_mol = Chem.Mol(mol)
        conf_ids = [conf.GetId() for conf in pose_mol.GetConformers()]
        if conf_id not in conf_ids:
            raise ValueError(f"Conformer {conf_id} not present")
        for existing_conf_id in conf_ids:
            if existing_conf_id != conf_id:
                pose_mol.RemoveConformer(existing_conf_id)
        if pose_mol.GetNumConformers() != 1:
            raise RuntimeError(
                f"Expected one conformer after extraction, found {pose_mol.GetNumConformers()}"
            )
        return pose_mol

    mol_dir = out_dir / mol_id
    mol_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "mol_id": mol_id,
        "smiles": smiles,
        "status": "failed",
        "best_score": None,
        "n_poses": 0,
    }

    # Step 1: Prepare 3D conformer
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        result["error"] = "Invalid SMILES"
        return result

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    conf_id = AllChem.EmbedMolecule(mol, params)
    if conf_id < 0:
        result["error"] = "Embedding failed"
        return result

    AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)

    # Step 2: Convert to PDBQT via Meeko
    try:
        import meeko
        prep = meeko.MoleculePreparation(hydrate=True)
        prep.prepare(mol)
        pdbqt_str = prep.write_pdbqt_string()
    except Exception as e:
        result["error"] = f"Meeko failed: {e}"
        return result

    # Save ligand PDBQT
    lig_pdbqt_path = mol_dir / f"{mol_id}_ligand.pdbqt"
    lig_pdbqt_path.write_text(pdbqt_str)

    # Step 3: Dock with Vina
    try:
        import vina
        v = vina.Vina(sf_name="vina", seed=seed)
        v.set_receptor(str(receptor_pdbqt))
        v.set_ligand_from_string(pdbqt_str)
        v.compute_vina_maps(
            center=BINDING_SITE["center"],
            box_size=BINDING_SITE["box_size"],
        )
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

        # Get scores
        energies = v.energies()
        best_score = float(energies[0][0]) if len(energies) > 0 else None
        if best_score is None:
            result["error"] = "Vina returned no poses"
            print(f"  {mol_id}: FAILED - no poses returned")
            return result
        result["best_score"] = best_score
        result["n_poses"] = len(energies)

        # Save all poses as PDBQT
        poses_pdbqt = v.poses(n_poses=n_poses)
        poses_path = mol_dir / f"{mol_id}_poses.pdbqt"
        poses_path.write_text(poses_pdbqt)

        # Convert poses to SDF via Meeko
        try:
            pmol = meeko.PDBQTMolecule(poses_pdbqt)
            rdkit_mols = meeko.RDKitMolCreate.from_pdbqt_mol(pmol)
            if not rdkit_mols or rdkit_mols[0] is None:
                raise ValueError("Meeko returned no RDKit poses")
            rdkit_mol = rdkit_mols[0]
            rdkit_mol = Chem.RemoveHs(rdkit_mol)
            if rdkit_mol.GetNumConformers() == 0:
                raise ValueError("RDKit pose bundle has no conformers")

            sdf_path = mol_dir / f"{mol_id}_poses.sdf"
            writer = Chem.SDWriter(str(sdf_path))
            for pose_idx in range(rdkit_mol.GetNumConformers()):
                conf_id = rdkit_mol.GetConformer(pose_idx).GetId()
                pose_mol = extract_pose_conformer(rdkit_mol, conf_id)
                if pose_idx < len(energies):
                    pose_mol.SetProp("vina_score", f"{energies[pose_idx][0]:.3f}")
                pose_mol.SetProp("_Name", f"{mol_id}_pose{pose_idx}")
                pose_mol.SetProp("dock_id", mol_id)
                writer.write(pose_mol)
            writer.close()
            best_pose = extract_pose_conformer(rdkit_mol, rdkit_mol.GetConformer(0).GetId())
        except Exception as e:
            result["error"] = f"SDF conversion failed: {e}"
            print(f"  {mol_id}: FAILED - {result['error']}")
            return result

        # Create complexes (best pose + protein) for PLIP
        try:
            protein = Chem.MolFromPDBFile(
                str(receptor_pdb), removeHs=False, sanitize=False
            )
            if protein is None:
                raise ValueError("Failed to load receptor PDB")
            complex_mol = Chem.CombineMols(protein, best_pose)
            complex_path = mol_dir / f"{mol_id}_complex.pdb"
            Chem.MolToPDBFile(complex_mol, str(complex_path), flavor=4)
            result["complex_pdb"] = str(complex_path)
        except Exception as e:
            result["error"] = f"Complex creation failed: {e}"
            print(f"  {mol_id}: FAILED - {result['error']}")
            return result

        result["status"] = "success"
        print(f"  {mol_id}: score={best_score:.2f}, {len(energies)} poses")

    except Exception as e:
        result["error"] = f"Vina docking failed: {e}"
        print(f"  {mol_id}: FAILED - {e}")

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--consensus-tsv", type=Path, required=True,
                   help="Consensus candidates TSV from MW regression")
    p.add_argument("--receptor-pdbqt", type=Path, required=True)
    p.add_argument("--receptor-pdb", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--exhaustiveness", type=int, default=32)
    p.add_argument("--n-poses", type=int, default=20)
    p.add_argument("--max-molecules", type=int, default=10,
                   help="Max molecules to re-dock (default: 10)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RE-DOCKING TOP CONSENSUS CANDIDATES")
    print("=" * 60)
    print(f"  Exhaustiveness: {args.exhaustiveness}")
    print(f"  Poses: {args.n_poses}")
    print(f"  Receptor: {args.receptor_pdbqt}")

    df = pd.read_csv(args.consensus_tsv, sep="\t")
    df = df.head(args.max_molecules)
    print(f"  Candidates: {len(df)}")

    for col in ("std_SMILES", "SMILES", "smiles"):
        if col in df.columns:
            smiles_col = col
            break
    else:
        raise KeyError(f"No SMILES column found. Columns: {list(df.columns)}")
    id_col = "dock_id"

    results = []
    for _, row in df.iterrows():
        mol_id = row[id_col]
        smiles = row[smiles_col]
        print(f"\n  Docking {mol_id}...")
        r = prepare_and_dock_molecule(
            smiles, mol_id,
            args.receptor_pdbqt, args.receptor_pdb,
            out_dir, args.exhaustiveness, args.n_poses,
        )
        results.append(r)

    # Summary
    results_df = pd.DataFrame(results)
    results_path = out_dir / "redock_results.tsv"
    results_df.to_csv(results_path, sep="\t", index=False)

    n_success = sum(1 for r in results if r["status"] == "success")
    print(f"\n{'=' * 60}")
    print(f"Re-docking complete: {n_success}/{len(results)} successful")
    if n_success > 0:
        scores = [r["best_score"] for r in results if r["best_score"] is not None]
        print(f"  Best score: {min(scores):.2f}")
        print(f"  Median score: {sorted(scores)[len(scores)//2]:.2f}")
    print(f"Results: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
