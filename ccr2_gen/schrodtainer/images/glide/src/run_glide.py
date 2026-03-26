import argparse
import os
import time

import pandas as pd

from glide import run

def _find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _read_lists_from_csv(path):
    # .smi files are tab-separated with no header (SMILES<TAB>ID per line)
    if path.endswith(".smi"):
        df = pd.read_csv(path, sep="\t", header=None, names=["smiles", "name"])
    else:
        df = pd.read_csv(path)
    smiles_col = _find_column(df, ["smiles", "SMILES", "Smiles"])
    id_col = _find_column(df, ["name", "NAME", "compound number", "compound_number", "id", "ID", "MOL_ID", "MolID"])
    if smiles_col is None or id_col is None:
        raise SystemExit(f"Could not find SMILES/ID columns in {path}. Found columns: {list(df.columns)}")
    return df[smiles_col].astype(str).tolist(), df[id_col].astype(str).tolist()

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run Glide docking workflow (wraps run())")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ligand-csv", help="CSV with columns for SMILES and ID (e.g. 'smiles' and 'name').", type=str)
    group.add_argument("--smiles", help="List of SMILES strings (use spaces to separate).", nargs="+")
    parser.add_argument("--ids", help="List of molecule IDs corresponding to --smiles (use spaces to separate).", nargs="+")
    parser.add_argument("--smiles-csv", help="CSV file containing SMILES column (can be repeated).", nargs="+")
    parser.add_argument("--ids-csv", help="CSV file containing corresponding IDs (can be repeated).", nargs="+")
    parser.add_argument("--grid-file", required=True, help="Path to Glide grid file (zip).")
    parser.add_argument("--chunk-size", type=int, default=100, help="Number of molecules to process per worker.")
    parser.add_argument("--workers", type=int, default=1 , help="Number of parallel workers.")
    parser.add_argument("--output-dir", default=os.getcwd(), help="Directory to save results.")
    parser.add_argument("--run-name", default="glide_run")
    parser.add_argument("--precision", default="SP", choices=["HTVS", "SP", "XP"],
                        help="Glide docking precision (default: SP).")
    args = parser.parse_args(argv)

    smiles = []
    ids = []

    if args.ligand_csv:
        s, i = _read_lists_from_csv(args.ligand_csv)
        smiles.extend(s); ids.extend(i)
    elif args.smiles and args.ids:
        if len(args.smiles) != len(args.ids):
            raise SystemExit("Length of --smiles and --ids must match.")
        smiles = args.smiles
        ids = args.ids
    elif args.smiles_csv and args.ids_csv:
        if len(args.smiles_csv) != len(args.ids_csv):
            raise SystemExit("--smiles-csv and --ids-csv must have the same number of files when used together.")
        for sfile, ifile in zip(args.smiles_csv, args.ids_csv):
            s_list, _ = _read_lists_from_csv(sfile)
            _, i_list = _read_lists_from_csv(ifile)
            if len(s_list) != len(i_list):
                raise SystemExit(f"Mismatch lengths in {sfile} and {ifile}")
            smiles.extend(s_list); ids.extend(i_list)
    else:
        raise SystemExit("Provide either --ligand-csv or both --smiles and --ids, or matching --smiles-csv/--ids-csv pairs.")

    os.makedirs(args.output_dir, exist_ok=True)
    time_start = time.time()
    scores, df_all = run(smiles, ids, args.grid_file, chunk_size=args.chunk_size,
                         n_workers=args.workers, precision=args.precision)
    time_end = time.time()

    out_path = os.path.join(args.output_dir, f"{args.run_name}.csv")
    df_all.to_csv(out_path, index=False)

    print(f"Completed in {time_end - time_start:.2f} s; results saved to `{out_path}`")
    print("Docking scores:")
    print(scores)

if __name__ == "__main__":
    main()
