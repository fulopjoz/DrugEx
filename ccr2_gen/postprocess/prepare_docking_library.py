#!/usr/bin/env python3
"""
Phase 5a: Prepare docking candidate library from Phase 3 analysis output.

Takes the all_docking_candidates.tsv (top-N per Butina cluster from all models)
and prepares a deduplicated, property-filtered library chunked for PBS array
submission to Glide docking.

Pipeline:
  1. Load all docking candidates from Phase 3
  2. Global deduplication (tautomer-aware canonical SMILES)
  3. Property filters: MW 200-600, LogP -2 to 6, TPSA 20-140, RotBonds <= 12
  4. Export as .smi files (SMILES<tab>ID, one per line) — Glide input format
  5. Split into chunks for PBS array submission
  6. Write chunk manifest for PBS array indexing

Outputs:
  - docking/library/docking_library.smi           Full filtered library
  - docking/library/docking_library_stats.json     Library statistics
  - docking/library/chunks/chunk_001.smi ... N     Per-chunk .smi files
  - docking/library/chunk_manifest.tsv             Manifest for PBS array

Usage:
  python ccr2_gen/postprocess/prepare_docking_library.py \
    --analysis-dir ccr2_gen/postprocess/analysis_20260117 \
    --out-dir ccr2_gen/docking/library \
    --chunk-size 1000 \
    --selected-models ccr2_gen/postprocess/model_selection/selected_models.tsv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.logger().setLevel(RDLogger.ERROR)


# ---------------------------------------------------------------------------
# Property filters
# ---------------------------------------------------------------------------

PROPERTY_FILTERS = {
    "MW": (200.0, 600.0),
    "LogP": (-2.0, 6.0),
    "TPSA": (20.0, 140.0),
    "RotBonds": (0, 12),
    "NumRings": (0, 5),       # max reference (CCX140) = 4 rings, +1 margin
}


def compute_properties(smi: str) -> dict | None:
    """Compute druglikeness properties for a SMILES string."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return {
        "MW": Descriptors.ExactMolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "NumRings": rdMolDescriptors.CalcNumRings(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
    }


def passes_filters(props: dict) -> bool:
    """Check if computed properties pass all drug-likeness filters."""
    for prop, (lo, hi) in PROPERTY_FILTERS.items():
        val = props.get(prop)
        if val is None:
            return False
        if val < lo or val > hi:
            return False
    return True


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--analysis-dir", type=Path, required=True,
                   help="Phase 3 output directory (contains all_docking_candidates.tsv).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for docking library.")
    p.add_argument("--chunk-size", type=int, default=1000,
                   help="Molecules per chunk for PBS array (default: 1000).")
    p.add_argument("--selected-models", type=Path, default=None,
                   help="TSV from select_models.py. If provided, only candidates "
                        "from these models are included in the docking library.")
    p.add_argument("--max-per-model", type=int, default=None,
                   help="Max molecules to sample per model before deduplication "
                        "(stratified equal-N sampling). Default: no limit.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for per-model subsampling (default: 42).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    out_dir = args.out_dir.resolve()
    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Load docking candidates
    candidates_path = analysis_dir / "all_docking_candidates.tsv"
    if not candidates_path.exists():
        print(f"ERROR: {candidates_path} not found. Run analyze_generated.py first.")
        return

    print("Loading docking candidates...")
    df = pd.read_csv(candidates_path, sep="\t")
    print(f"  Raw candidates: {len(df)}")

    # Filter to selected models if provided
    if args.selected_models:
        sel_path = args.selected_models.resolve()
        if sel_path.exists():
            sel_df = pd.read_csv(sel_path, sep="\t")
            selected_ids = set(sel_df["model_id"])
            print(f"  Filtering to {len(selected_ids)} selected models...")
            df = df[df["model_id"].isin(selected_ids)].reset_index(drop=True)
            print(f"  After model filter: {len(df)}")
        else:
            print(f"  WARNING: --selected-models {sel_path} not found, using all candidates")

    # Determine SMILES column
    smi_col = "std_SMILES" if "std_SMILES" in df.columns else "SMILES"

    # Per-model stratified sampling (before dedup, so each model contributes equally)
    if args.max_per_model is not None and "model_id" in df.columns:
        rng = np.random.default_rng(args.seed)
        sampled_frames = []
        for model_id, group in df.groupby("model_id"):
            if len(group) > args.max_per_model:
                idx = rng.choice(len(group), args.max_per_model, replace=False)
                group = group.iloc[idx]
            sampled_frames.append(group)
        df = pd.concat(sampled_frames, ignore_index=True)
        print(f"  After per-model sampling (max {args.max_per_model}): {len(df)}")

    # Global deduplication on canonical SMILES
    print("Deduplicating (tautomer-aware canonical SMILES)...")
    df = df.drop_duplicates(subset=[smi_col]).reset_index(drop=True)
    print(f"  After dedup: {len(df)}")

    # Compute properties and filter
    print("Computing properties and filtering...")
    props_list = []
    for smi in df[smi_col]:
        props_list.append(compute_properties(smi))

    df["_props"] = props_list
    df["_valid_props"] = df["_props"].apply(lambda p: p is not None)
    df["_passes_filter"] = df["_props"].apply(lambda p: passes_filters(p) if p else False)

    # Add property columns for the output
    for prop_name in ["MW", "LogP", "TPSA", "RotBonds", "HBA", "HBD", "HeavyAtoms"]:
        df[prop_name] = df["_props"].apply(lambda p: p.get(prop_name) if p else None)

    n_invalid = (~df["_valid_props"]).sum()
    n_filtered_out = df["_valid_props"].sum() - df["_passes_filter"].sum()

    df_filtered = df[df["_passes_filter"]].copy()
    df_filtered = df_filtered.drop(columns=["_props", "_valid_props", "_passes_filter"])
    print(f"  Invalid SMILES: {n_invalid}")
    print(f"  Filtered out (property): {n_filtered_out}")
    print(f"  Final library: {len(df_filtered)}")

    # Filter breakdown
    print("\n  Property filter breakdown:")
    for prop, (lo, hi) in PROPERTY_FILTERS.items():
        vals = df[df["_valid_props"]][prop]
        n_fail = ((vals < lo) | (vals > hi)).sum()
        print(f"    {prop} [{lo}, {hi}]: {n_fail} failed")

    # Assign unique IDs
    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered["dock_id"] = [f"CCR2_GEN_{i:06d}" for i in range(len(df_filtered))]

    # Write full library .smi
    smi_path = out_dir / "docking_library.smi"
    with open(smi_path, "w") as f:
        for _, row in df_filtered.iterrows():
            f.write(f"{row[smi_col]}\t{row['dock_id']}\n")
    print(f"\n  Full library: {smi_path}")

    # Write full library TSV (with all metadata)
    tsv_path = out_dir / "docking_library.tsv"
    df_filtered.to_csv(tsv_path, sep="\t", index=False)
    print(f"  Full library TSV: {tsv_path}")

    # Split into chunks
    print(f"\n  Splitting into chunks of {args.chunk_size}...")
    n_chunks = (len(df_filtered) + args.chunk_size - 1) // args.chunk_size
    manifest_rows = []

    for chunk_idx in range(n_chunks):
        start = chunk_idx * args.chunk_size
        end = min(start + args.chunk_size, len(df_filtered))
        chunk_df = df_filtered.iloc[start:end]

        chunk_name = f"chunk_{chunk_idx + 1:03d}.smi"
        chunk_path = chunk_dir / chunk_name
        with open(chunk_path, "w") as f:
            for _, row in chunk_df.iterrows():
                f.write(f"{row[smi_col]}\t{row['dock_id']}\n")

        manifest_rows.append({
            "chunk_idx": chunk_idx + 1,
            "chunk_file": chunk_name,
            "n_molecules": len(chunk_df),
            "first_id": chunk_df.iloc[0]["dock_id"],
            "last_id": chunk_df.iloc[-1]["dock_id"],
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "chunk_manifest.tsv"
    manifest_df.to_csv(manifest_path, sep="\t", index=False)
    print(f"  Chunks: {n_chunks} files in {chunk_dir}")
    print(f"  Manifest: {manifest_path}")

    # Write statistics JSON
    stats = {
        "raw_candidates": int(len(pd.read_csv(candidates_path, sep="\t"))),
        "max_per_model": args.max_per_model,
        "seed": args.seed,
        "after_dedup": int(len(df.drop_duplicates(subset=[smi_col]))),
        "invalid_smiles": int(n_invalid),
        "filtered_out_property": int(n_filtered_out),
        "final_library_size": int(len(df_filtered)),
        "n_chunks": n_chunks,
        "chunk_size": args.chunk_size,
        "property_filters": {k: list(v) for k, v in PROPERTY_FILTERS.items()},
        "property_stats": {},
    }
    for prop in ["MW", "LogP", "TPSA", "RotBonds", "HBA", "HBD", "HeavyAtoms"]:
        vals = df_filtered[prop].dropna()
        stats["property_stats"][prop] = {
            "mean": round(float(vals.mean()), 2),
            "std": round(float(vals.std()), 2),
            "min": round(float(vals.min()), 2),
            "max": round(float(vals.max()), 2),
        }

    stats_path = out_dir / "docking_library_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"  Stats: {stats_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Docking Library Summary")
    print("=" * 60)
    print(f"  {'Molecules':30s} {len(df_filtered):>10d}")
    print(f"  {'Chunks':30s} {n_chunks:>10d}")
    print(f"  {'Chunk size':30s} {args.chunk_size:>10d}")
    print("-" * 60)
    for prop in ["MW", "LogP", "TPSA", "RotBonds"]:
        vals = df_filtered[prop].dropna()
        print(f"  {prop:30s} {vals.mean():>8.1f} +/- {vals.std():.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
