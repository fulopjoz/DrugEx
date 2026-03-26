import os
import subprocess
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Union, Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule


def smiles_to_mgz(smiles: list[str], ids: list[str]) -> str:
    """Convert SMILES strings to MAE format via SDF intermediate.

    Parses SMILES with RDKit, adds hydrogens, embeds 3D coordinates using
    ETKDGv3, and converts to Maestro MAE via sdconvert.

    Args:
        smiles (list[str]): The SMILES strings representing the molecules.
        ids (list[str]): The molecule IDs corresponding to the SMILES.

    Returns:
        str: Path to the MAE file.
    """
    from rdkit.Chem import rdDistGeom

    # Parse SMILES, add Hs, and embed 3D — store results back into the list.
    # ETKDGv3 uses CSD experimental torsion preferences and produces
    # crystal-structure-quality geometries; MMFF minimization is NOT applied
    # because it distorts ETKDG torsions toward gas-phase minima.
    # Ref: RDKit docs — "With [ETKDG] there should be no need to use a
    # minimisation step to clean up the structures."
    # Ref: Landrum, RDKit blog 2022-09-29 (optimizing confgen parameters)
    # Ref: github.com/rdkit/rdkit/discussions/8226 (best practices)
    mols = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            mols.append(None)
            continue
        mol = Chem.AddHs(mol)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        conf_id = EmbedMolecule(mol, params)
        if conf_id < 0:
            # Fallback: relax with random coordinates for difficult topologies
            params2 = rdDistGeom.ETKDGv3()
            params2.randomSeed = 42
            params2.useRandomCoords = True
            params2.maxIterations = 5000
            conf_id = EmbedMolecule(mol, params2)
        if conf_id < 0:
            # Second fallback: small-ring torsions + random coords
            params3 = rdDistGeom.srETKDGv3()
            params3.randomSeed = 42
            params3.useRandomCoords = True
            params3.useSmallRingTorsions = True
            conf_id = EmbedMolecule(mol, params3)
        if conf_id < 0:
            print(f"  WARNING: embedding failed for SMILES: {s[:60]}...")
        mols.append(mol)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".sdf")
    sdf_writer = Chem.SDWriter(temp_file.name)
    for mol, mol_id in zip(mols, ids):
        if mol is None:
            print(f"  WARNING: skipping {mol_id} — invalid SMILES")
            continue
        if mol.GetNumConformers() == 0:
            print(f"  WARNING: skipping {mol_id} — no 3D conformer generated")
            continue
        mol.SetProp("SMILES", Chem.MolToSmiles(Chem.RemoveHs(mol)))
        mol.SetProp("MOL_ID", mol_id)
        sdf_writer.write(mol)
    sdf_writer.close()
    # run the sdconvert command on command line to convert the SDF to MGZ
    mgz_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mae")
    schrodinger = os.environ["SCHRODINGER"]
    subprocess.run(
        [f"{schrodinger}/utilities/sdconvert", "-isd", temp_file.name,
         "-omae", mgz_file.name],
        check=True,
    )
    return mgz_file.name

def ligprep(mgz_path: str) -> str:
    """Run LigPrep on the given MAE file.

    Args:
        mgz_path (str): Path to the input MAE file.
    Returns:
        str: Path to the prepared ligand MAE file.
    """
    work_dir = tempfile.mkdtemp(prefix="ligprep_")
    mgz_path = os.path.abspath(mgz_path)
    out_mae = os.path.join(work_dir, "prepared.mae")
    inp_file = os.path.join(work_dir, "ligprep.inp")

    with open(inp_file, "w") as f:
        f.write(f"""INPUT_FILE_NAME   {mgz_path}
MAX_ATOMS   500
FORCE_FIELD   16
EPIK   yes
EPIK_METAL_BINDING   no
INCLUDE_ORIGINAL_STATE   no
DETERMINE_CHIRALITIES   no
IGNORE_CHIRALITIES   no
NUM_STEREOISOMERS   32""")

    subprocess.run(
        [f"{os.environ['SCHRODINGER']}/ligprep",
         "-inp", inp_file,
         "-NJOBS", "1",
         "-JOBNAME", "ligprep_job",
         "-HOST", "localhost",
         "-omae", out_mae],
        check=True,
        cwd=work_dir,
    )

    # LigPrep may write to -omae path or to {JOBNAME}-out.mae
    if not os.path.exists(out_mae):
        # Search for any .mae file LigPrep produced
        import glob
        candidates = glob.glob(os.path.join(work_dir, "*.mae"))
        candidates = [c for c in candidates if os.path.getsize(c) > 0]
        if candidates:
            out_mae = candidates[0]
            print(f"  LigPrep output found at: {os.path.basename(out_mae)}")
        else:
            # List directory for diagnostics
            contents = os.listdir(work_dir)
            raise FileNotFoundError(
                f"LigPrep produced no .mae output. "
                f"Work dir contents: {contents}"
            )
    return out_mae

def unzip_and_list(zip_path: Union[str, Path], extract_to: Optional[Union[str, Path]] = None) -> list[str]:
    """
    Extract `zip_path` into `extract_to` (or a new temp dir) and return all extracted file paths.
    """
    zip_path = Path(zip_path)
    if extract_to is None:
        extract_to = Path(tempfile.mkdtemp(prefix="unzip_"))
    else:
        extract_to = Path(extract_to)
        extract_to.mkdir(parents=True, exist_ok=True)

    extract_to_resolved = extract_to.resolve()
    extracted_files = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            # target path for this member
            target = extract_to_resolved.joinpath(member.filename)
            # prevent zip-slip
            if not str(target.resolve()).startswith(str(extract_to_resolved) + str(Path("/")) ) and target.resolve() != extract_to_resolved:
                raise RuntimeError(f"Zip file contains illegal path: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as source, open(target, "wb") as dest:
                    dest.write(source.read())
                extracted_files.append(str(target))

    # collect all extracted files
    return extracted_files

def glide_docking(ligprep_input: str, grid_file: str,
                  precision: str = "SP") -> tuple[dict, pd.DataFrame]:
    """Run Glide docking and return docking scores.

    Args:
        ligprep_input (str): Path to the prepared ligand MAE file.
        grid_file (str): Path to the Glide grid file (.zip).
        precision (str): Glide precision level — "HTVS", "SP", or "XP".
    Returns:
        dict: Best docking scores mapped to compound IDs.
        pd.DataFrame: DataFrame containing full docking results.
    """
    work_dir = tempfile.mkdtemp(prefix="glide_dock_")
    ligprep_input = os.path.abspath(ligprep_input)
    grid_file = os.path.abspath(grid_file)

    glide_in_path = os.path.join(work_dir, "glide_dock.in")
    with open(glide_in_path, "w") as f:
        f.write(f"""FORCEFIELD   OPLS_2005
GRIDFILE   {grid_file}
LIGANDFILE   {ligprep_input}
NREPORT   10
POSES_PER_LIG   10
PRECISION   {precision}
EXPANDED_SAMPLING   True
NENHANCED_SAMPLING   2""")

    print(f"Prepared Glide input file: {glide_in_path} (precision={precision})")
    subprocess.run(
        [f"{os.environ['SCHRODINGER']}/glide", glide_in_path,
         "-OVERWRITE", "-adjust", "-HOST", "localhost", "-TMPLAUNCHDIR"],
        check=True,
        cwd=work_dir,
    )

    # Find the Glide output poses (may be _subjob_poses.zip or _pv.maegz)
    glide_stem = os.path.join(work_dir, "glide_dock")
    poses_zip = f"{glide_stem}_subjob_poses.zip"
    pv_mae = f"{glide_stem}_pv.maegz"

    # Wait for output (up to 5 min)
    waited = 0
    while waited < 300:
        if os.path.exists(poses_zip) or os.path.exists(pv_mae):
            break
        time.sleep(2)
        waited += 2
    else:
        # List directory to help diagnose
        contents = os.listdir(work_dir)
        raise FileNotFoundError(
            f"Glide output not found after 5 min. "
            f"Expected {poses_zip} or {pv_mae}. "
            f"Work dir contents: {contents}"
        )

    # Determine which output format we got and extract MAE files
    mae_files = []
    if os.path.exists(poses_zip):
        mae_files = [f for f in unzip_and_list(poses_zip, extract_to=work_dir)
                     if f.endswith((".mae", ".maegz"))]
    elif os.path.exists(pv_mae):
        mae_files = [pv_mae]

    # Convert MAE → SDF
    sd_outs = []
    for mae_f in mae_files:
        sdf_f = f"{mae_f}.sdf"
        sd_outs.append(sdf_f)
        subprocess.run(
            [f"{os.environ['SCHRODINGER']}/utilities/sdconvert",
             "-imae", mae_f, "-osd", sdf_f],
            check=True,
            cwd=work_dir,
        )

    # Parse SDF outputs into a DataFrame
    props_list = []
    for sdf_f in sd_outs:
        suppl = Chem.SDMolSupplier(sdf_f)
        for mol in suppl:
            if mol is None:
                continue
            props = mol.GetPropsAsDict()
            props["SMILES_pose"] = Chem.MolToSmiles(mol, isomericSmiles=True)
            props["MolBlock"] = Chem.MolToMolBlock(mol)
            props_list.append(props)
    df_main = pd.DataFrame(props_list)

    if df_main.empty:
        print("  WARNING: Glide returned no docked poses")
        return {}, df_main

    # Keep best score per molecule (lowest r_i_docking_score)
    if "MOL_ID" in df_main.columns and "r_i_docking_score" in df_main.columns:
        df_best = df_main.loc[df_main.groupby("MOL_ID")["r_i_docking_score"].idxmin()]
        df_best = df_best.set_index("MOL_ID")
        return df_best["r_i_docking_score"].to_dict(), df_main

    print("  WARNING: Expected columns MOL_ID/r_i_docking_score not found in output")
    return {}, df_main

def run_workflow(smiles: list[str], ids: list[str], grid_file: str,
                 precision: str = "SP") -> tuple[float, pd.DataFrame]:
    """Run the full workflow: SMILES -> SDF -> MAE -> LigPrep -> Glide Docking.

    Args:
        smiles (list[str]): The SMILES strings representing the molecules.
        ids (list[str]): The molecule IDs corresponding to the SMILES.
        grid_file (str): The path to the Glide grid file.
        precision (str): Glide precision — "HTVS", "SP", or "XP".
    Returns:
        dict: Best docking scores mapped to compound IDs.
        pd.DataFrame: DataFrame containing full docking results.
    """
    mae_file = smiles_to_mgz(smiles, ids)
    print(f"MAE file created at: {mae_file}")
    prepared_ligand = ligprep(mae_file)
    print(f"Prepared ligand file created at: {prepared_ligand}")
    grid_file = os.path.abspath(grid_file)
    print(f"Running Glide docking with grid file: {grid_file}")
    return glide_docking(prepared_ligand, grid_file, precision=precision)

def run(smiles: list[str], mol_ids: list[str], grid_file: str,
        chunk_size: int = 10, n_workers: int = 4,
        precision: str = "SP") -> tuple[dict, pd.DataFrame]:
    """Run the full workflow in parallel for multiple SMILES strings.

    Uses ProcessPoolExecutor (not threads) to avoid os.chdir() race conditions,
    since each subprocess gets its own working directory state.

    Args:
        smiles (list[str]): The SMILES strings representing the molecules.
        mol_ids (list[str]): The molecule IDs corresponding to the SMILES.
        grid_file (str): The path to the Glide grid file.
        chunk_size (int): The number of SMILES to process in each chunk.
        n_workers (int): The number of parallel workers to use.
        precision (str): Glide precision — "HTVS", "SP", or "XP".
    Returns:
        dict: The best docking scores mapped to compound IDs.
        pd.DataFrame: DataFrame containing full docking results.
    """
    grid_file = os.path.abspath(grid_file)

    # Partition inputs into chunks
    chunks = []
    for i in range(0, len(smiles), chunk_size):
        chunk_smiles = smiles[i:i + chunk_size]
        chunk_ids = mol_ids[i:i + chunk_size]
        chunks.append((chunk_smiles, chunk_ids))

    # Run workflow in parallel using processes (thread-safe)
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(run_workflow, chunk[0], chunk[1], grid_file,
                            precision): chunk
            for chunk in chunks
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                chunk_info = futures[future]
                print(f"Error processing chunk (first ID: {chunk_info[1][0] if chunk_info[1] else '?'}): {e}")

    # Combine all results
    scores = {}
    for res in results:
        scores.update(res[0])
    dfs = [res[1] for res in results if not res[1].empty]
    df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return scores, df_all




