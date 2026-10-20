"""Build the EGCG 3D reference SDF for the shape-matching pilot.

ETKDGv3 embedding with a fixed seed, MMFF94 optimisation, minimum-energy
conformer selected, written as a one-record SDF with explicit hydrogens.

The single-conformer choice is a STATED ASSUMPTION, not a neutral default:
EGCG's gallate ester admits several low-energy geometries and this fixes one.

Gate 1 runs before the file is accepted. A reference that silently failed to
embed scores every generated molecule 0.0 while exiting 0, which is the exact
failure this gate exists to prevent.

Usage:
    python build_reference.py            # build + verify
    python build_reference.py --verify   # verify an existing file only
"""

import argparse
import hashlib
import sys

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.Scaffolds import MurckoScaffold

import config_egcg as cfg


def gate1(path, expect_heavy=cfg.EGCG_HEAVY_ATOMS):
    """Gate 1: the SDF is a single, 3D, correctly sized molecule.

    Raises RuntimeError on any failure. Returns the parsed molecule.
    """
    if not path.exists():
        raise RuntimeError(f"Gate 1 FAILED: reference does not exist: {path}")

    mols = [m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m is not None]
    if len(mols) != 1:
        raise RuntimeError(
            f"Gate 1 FAILED: expected exactly 1 parsable molecule, got {len(mols)}. "
            "A multi-record or unparsable reference changes the scorer key and the "
            "meaning of every score."
        )
    mol = mols[0]

    if mol.GetNumConformers() < 1:
        raise RuntimeError("Gate 1 FAILED: reference has no conformer.")

    conf = mol.GetConformer()
    if not conf.Is3D():
        raise RuntimeError("Gate 1 FAILED: reference conformer is not 3D.")

    # A conformer can exist, claim 3D, and still be flat/degenerate if embedding
    # silently failed. All-zero or coplanar coordinates would score ~0 forever.
    zs = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
    if max(zs) - min(zs) < 1e-6:
        raise RuntimeError(
            "Gate 1 FAILED: reference is flat in z -- embedding produced "
            "degenerate coordinates."
        )

    heavy = mol.GetNumHeavyAtoms()
    if heavy != expect_heavy:
        raise RuntimeError(
            f"Gate 1 FAILED: {heavy} heavy atoms, expected {expect_heavy}. "
            "The reference is not EGCG."
        )

    return mol


def describe(mol):
    """Descriptors, computed rather than quoted."""
    flat = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.RemoveHs(mol)))
    scaffold = MurckoScaffold.GetScaffoldForMol(flat)
    return {
        "formula": rdMD.CalcMolFormula(flat),
        "mw": round(Descriptors.MolWt(flat), 2),
        "hbd": rdMD.CalcNumHBD(flat),
        "hba": rdMD.CalcNumHBA(flat),
        "tpsa": round(Descriptors.TPSA(flat), 1),
        "heavy_atoms": flat.GetNumHeavyAtoms(),
        "rotatable_bonds": rdMD.CalcNumRotatableBonds(flat),
        "murcko_heavy_atoms": scaffold.GetNumHeavyAtoms(),
        "canonical_smiles": Chem.MolToSmiles(flat),
    }


def build(path=cfg.REFERENCE_SDF, smiles=cfg.EGCG_SMILES, seed=cfg.REFERENCE_EMBED_SEED):
    """Embed, optimise, select the minimum-energy conformer, write the SDF."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"EGCG SMILES did not parse: {smiles}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    n_embedded = len(AllChem.EmbedMultipleConfs(mol, numConfs=200, params=params))
    if n_embedded == 0:
        raise RuntimeError("ETKDGv3 embedded 0 conformers for EGCG.")

    results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
    # results is [(converged, energy), ...]; converged == 0 means success.
    energies = [(energy, idx) for idx, (converged, energy) in enumerate(results)
                if converged == 0]
    if not energies:
        raise RuntimeError(
            "MMFF94 optimisation converged for 0 of "
            f"{n_embedded} conformers -- refusing to pick an unoptimised geometry."
        )
    best_energy, best_idx = min(energies)

    best = Chem.Mol(mol, False, best_idx)        # single-conformer copy
    best.SetProp("_Name", "EGCG")
    best.SetProp("source_smiles", smiles)
    best.SetProp("embed_seed", str(seed))
    best.SetProp("mmff_energy_kcal_mol", f"{best_energy:.4f}")
    best.SetProp("n_conformers_embedded", str(n_embedded))
    best.SetProp("n_conformers_converged", str(len(energies)))

    path.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(path)) as w:
        w.write(best)

    return best_energy, n_embedded, len(energies)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="verify an existing reference without rebuilding")
    args = ap.parse_args()

    path = cfg.REFERENCE_SDF

    if not args.verify:
        energy, n_emb, n_conv = build(path)
        print(f"Built {path}")
        print(f"  conformers embedded : {n_emb}")
        print(f"  MMFF converged      : {n_conv}")
        print(f"  selected MMFF energy: {energy:.4f} kcal/mol (minimum)")

    mol = gate1(path)
    print("Gate 1 PASSED: single 3D molecule, 33 heavy atoms, non-degenerate.")

    print("\nDescriptors (computed):")
    for k, v in describe(mol).items():
        print(f"  {k:20s} {v}")

    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"\nSHA-256: {sha}")
    if cfg.REFERENCE_SHA256:
        if sha != cfg.REFERENCE_SHA256:
            raise RuntimeError(
                f"Reference hash mismatch.\n  expected {cfg.REFERENCE_SHA256}\n"
                f"  actual   {sha}\nThe reference changed; recalibrate or restore."
            )
        print("Hash ASSERTED against config_egcg.REFERENCE_SHA256: match.")
    else:
        print("Hash RECORDED for provenance (REFERENCE_SHA256 not yet pinned).")
        print("Pin it by setting REFERENCE_SHA256 in config_egcg.py to the value above.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
