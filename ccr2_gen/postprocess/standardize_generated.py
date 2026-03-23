#!/usr/bin/env python3
"""
Phase 2: Post-process generated molecules for cross-model chemical comparison.

Pipeline per model:
  1. Load generated_all.tsv (10K molecules with scores)
  2. Standardize SMILES: ChEMBL standardize_mol → get_parent_mol → TautomerEnumerator.Canonicalize
  3. Deduplicate (tautomer-aware canonical SMILES)
  4. Write standardized outputs (all + desired)
  5. Subsample desired molecules to fixed N for fair comparison

Outputs per model (in the same directory as generated_all.tsv):
  - standardized_all.tsv          Full standardized dataset
  - standardized_desired.tsv      Standardized desired molecules (deduplicated)
  - standardized_desired_N.tsv    Fixed-N subsample for comparison (if enough desired)

Global outputs (in --out-dir):
  - standardization_summary.tsv   Per-model statistics
  - comparison_desired_N.tsv      Combined fixed-N samples across all models

Usage:
  python ccr2_gen/postprocess/standardize_generated.py \
    --gen-root ccr2_gen/generation/campaigns/20260117_production_10k \
    --out-dir ccr2_gen/postprocess/standardized_20260117 \
    --comparison-n 500 \
    --n-jobs 8
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

# Suppress all RDKit messages (kekulization warnings are at ERROR level)
RDLogger.logger().setLevel(RDLogger.CRITICAL)

# ChEMBL structure pipeline (already installed as qsprpred dependency)
from chembl_structure_pipeline import standardizer as chembl_std


_TAUT_ENUM = rdMolStandardize.TautomerEnumerator()
_UNCHARGER = rdMolStandardize.Uncharger()


def get_parent_mol_safe(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Return a salt-stripped, neutralized parent or None if parent cleanup fails."""
    try:
        parent, _ = chembl_std.get_parent_mol(mol, neutralize=True)
        return parent
    except Exception:
        try:
            parent = rdMolStandardize.FragmentParent(mol)
            return _UNCHARGER.uncharge(parent)
        except Exception:
            return None


def standardize_smiles(smi: str) -> Optional[str]:
    """
    Full standardization pipeline for a single SMILES string.

    Steps:
      1. Parse SMILES → RDKit mol
      2. ChEMBL standardize_mol (normalize, kekulize, sanitize)
      3. ChEMBL get_parent_mol (salt strip + neutralize)
      4. RDKit canonical tautomer
      5. Canonical SMILES output

    Returns None if any step fails (molecule is dropped).
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    # ChEMBL standardization (normalize functional groups, sanitize)
    try:
        mol = chembl_std.standardize_mol(mol)
    except Exception:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

    # Salt stripping + charge neutralization
    mol = get_parent_mol_safe(mol)
    if mol is None:
        return None

    # Canonical tautomer
    try:
        te = rdMolStandardize.TautomerEnumerator()
        mol = te.Canonicalize(mol)
    except Exception:
        pass

    if mol is None:
        return None

    return Chem.MolToSmiles(mol)


def standardize_smiles_fast(smi: str) -> Optional[str]:
    """Optimized version reusing pre-initialized objects."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        mol = chembl_std.standardize_mol(mol)
    except Exception:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
    mol = get_parent_mol_safe(mol)
    if mol is None:
        return None
    try:
        mol = _TAUT_ENUM.Canonicalize(mol)
    except Exception:
        pass
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize SMILES in a generation output DataFrame.

    Adds columns:
      - std_SMILES: standardized canonical SMILES (None if failed)
      - std_failed: True if standardization failed

    Drops rows where standardization failed and deduplicates on standardized SMILES.
    """
    df = df.copy()
    df["std_SMILES"] = df["SMILES"].apply(standardize_smiles_fast)
    df["std_failed"] = df["std_SMILES"].isna()

    n_failed = df["std_failed"].sum()
    if n_failed > 0:
        warnings.warn(f"Standardization failed for {n_failed}/{len(df)} molecules")

    # Drop failed, deduplicate on standardized SMILES
    df = df[~df["std_failed"]].copy()
    df = df.drop_duplicates(subset=["std_SMILES"]).reset_index(drop=True)
    return df


@dataclass
class ModelStats:
    model_id: str
    campaign: str
    backend: str
    reference: str
    eps_dir: str
    n_generated: int
    n_valid: int
    n_desired_raw: int
    n_std_failed: int
    n_desired_std: int
    n_desired_std_unique: int
    desired_ratio_raw: float
    desired_ratio_std: float
    unique_ratio_desired: float
    rocs_score_col: str
    rocs_mean_desired: float
    sa_mean_desired: float


def detect_rocs_col(df: pd.DataFrame) -> str:
    """Detect the ROCS score column name (varies by backend)."""
    candidates = [c for c in df.columns if c not in {"SMILES", "Valid", "Desired", "SA",
                                                       "std_SMILES", "std_failed"}]
    if len(candidates) == 1:
        return candidates[0]
    # Heuristic: look for ROCS/TanimotoCombo
    for c in candidates:
        if "ROCS" in c or "Tanimoto" in c:
            return c
    return candidates[0] if candidates else ""


def process_model(gen_dir: Path, comparison_n: int) -> Optional[ModelStats]:
    """
    Process a single model's generation output.

    Args:
        gen_dir: Directory containing generated_all.tsv and generation_summary.json
        comparison_n: Fixed N for comparison subsampling

    Returns:
        ModelStats or None if no valid output found.
    """
    all_tsv = gen_dir / "generated_all.tsv"
    summary_json = gen_dir / "generation_summary.json"

    if not all_tsv.exists():
        return None

    # Load summary for model metadata
    meta = {}
    if summary_json.exists():
        meta = json.loads(summary_json.read_text())

    model_id = meta.get("model_id", gen_dir.name)
    campaign = meta.get("train_campaign", "")
    backend = meta.get("backend", "")
    reference = meta.get("reference", "")
    eps_dir = meta.get("eps_dir", "")

    print(f"  Processing {model_id}...", flush=True)

    # Load full generation output
    df = pd.read_csv(all_tsv, sep="\t")
    n_generated = len(df)
    n_valid = int(df["Valid"].sum()) if "Valid" in df.columns else n_generated
    n_desired_raw = int(df["Desired"].sum()) if "Desired" in df.columns else 0

    # Standardize
    df_std = standardize_dataframe(df)
    n_std_failed = n_generated - len(df_std)

    # Split desired
    df_desired = df_std[df_std["Desired"] == 1].copy() if "Desired" in df_std.columns else df_std.copy()

    # Deduplicate desired on standardized SMILES
    df_desired_unique = df_desired.drop_duplicates(subset=["std_SMILES"]).copy()
    n_desired_std = len(df_desired)
    n_desired_std_unique = len(df_desired_unique)

    # Detect ROCS column
    rocs_col = detect_rocs_col(df)

    # Score statistics on desired
    rocs_mean = float(df_desired_unique[rocs_col].mean()) if rocs_col and len(df_desired_unique) > 0 else float("nan")
    sa_mean = float(df_desired_unique["SA"].mean()) if "SA" in df_desired_unique.columns and len(df_desired_unique) > 0 else float("nan")

    # Write standardized outputs
    # All standardized (with std_SMILES column)
    df_std.to_csv(gen_dir / "standardized_all.tsv", sep="\t", index=False)

    # Desired standardized unique
    df_desired_unique.to_csv(gen_dir / "standardized_desired.tsv", sep="\t", index=False)

    # Fixed-N subsample for comparison
    if len(df_desired_unique) >= comparison_n:
        df_sample = df_desired_unique.sample(n=comparison_n, random_state=42)
        df_sample.to_csv(gen_dir / f"standardized_desired_{comparison_n}.tsv", sep="\t", index=False)

    return ModelStats(
        model_id=model_id,
        campaign=campaign,
        backend=backend,
        reference=reference,
        eps_dir=eps_dir,
        n_generated=n_generated,
        n_valid=n_valid,
        n_desired_raw=n_desired_raw,
        n_std_failed=n_std_failed,
        n_desired_std=n_desired_std,
        n_desired_std_unique=n_desired_std_unique,
        desired_ratio_raw=n_desired_raw / n_generated if n_generated else 0.0,
        desired_ratio_std=n_desired_std / n_generated if n_generated else 0.0,
        unique_ratio_desired=n_desired_std_unique / n_desired_std if n_desired_std else 0.0,
        rocs_score_col=rocs_col,
        rocs_mean_desired=rocs_mean,
        sa_mean_desired=sa_mean,
    )


def find_generation_dirs(gen_root: Path) -> list[Path]:
    """Find all directories containing a completed generated_all.tsv."""
    return sorted(p.parent for p in gen_root.rglob("generated_all.tsv"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen-root", type=Path, required=True,
                   help="Root directory of generation campaign (contains campaign/backend/ref/eps dirs).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for global summary and comparison files.")
    p.add_argument("--comparison-n", type=int, default=500,
                   help="Fixed N for desired molecule subsampling (default: 500).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for subsampling.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gen_root = args.gen_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_n = args.comparison_n

    np.random.seed(args.seed)

    gen_dirs = find_generation_dirs(gen_root)
    print(f"Found {len(gen_dirs)} completed models in {gen_root}")
    print(f"Comparison N: {comparison_n}")
    print()

    if not gen_dirs:
        print("No completed generation outputs found. Exiting.")
        return

    # Process all models
    all_stats: list[ModelStats] = []
    for gen_dir in gen_dirs:
        stats = process_model(gen_dir, comparison_n)
        if stats:
            all_stats.append(stats)

    if not all_stats:
        print("\nNo models had valid generated_all.tsv. Generation may still be in progress.")
        return

    # Write global summary
    summary_df = pd.DataFrame([asdict(s) for s in all_stats])
    summary_path = out_dir / "standardization_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"\nSummary written to {summary_path}")

    # Compute comparison-N eligibility
    eligible = [s for s in all_stats if s.n_desired_std_unique >= comparison_n]
    ineligible = [s for s in all_stats if s.n_desired_std_unique < comparison_n]

    print(f"\nComparison-{comparison_n} eligibility:")
    print(f"  Eligible:   {len(eligible)} / {len(all_stats)} models")
    if ineligible:
        print(f"  Ineligible ({len(ineligible)}):")
        for s in ineligible:
            print(f"    {s.model_id}: only {s.n_desired_std_unique} unique desired")

    # Build combined comparison file
    if eligible:
        comparison_frames = []
        for s in eligible:
            # Find the subsample file
            gen_dir = gen_root
            for part in [s.campaign, s.backend, s.reference, s.eps_dir]:
                gen_dir = gen_dir / part
            sample_path = gen_dir / f"standardized_desired_{comparison_n}.tsv"
            if sample_path.exists():
                df = pd.read_csv(sample_path, sep="\t")
                df["model_id"] = s.model_id
                df["campaign"] = s.campaign
                df["backend"] = s.backend
                df["reference"] = s.reference
                df["eps_dir"] = s.eps_dir
                comparison_frames.append(df)

        if comparison_frames:
            comparison_df = pd.concat(comparison_frames, ignore_index=True)
            comparison_path = out_dir / f"comparison_desired_{comparison_n}.tsv"
            comparison_df.to_csv(comparison_path, sep="\t", index=False)
            print(f"\nComparison dataset: {comparison_path}")
            print(f"  {len(comparison_frames)} models x {comparison_n} = {len(comparison_df)} molecules")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Standardization Summary")
    print("=" * 80)
    print(f"{'Metric':40s} {'Mean':>10s} {'Min':>10s} {'Max':>10s}")
    print("-" * 72)
    for col, label in [
        ("n_generated", "Generated (total)"),
        ("n_std_failed", "Standardization failures"),
        ("n_desired_raw", "Desired (raw)"),
        ("n_desired_std_unique", "Desired (std, unique)"),
        ("desired_ratio_raw", "Desired ratio (raw)"),
        ("unique_ratio_desired", "Unique ratio (desired)"),
        ("rocs_mean_desired", "ROCS mean (desired)"),
        ("sa_mean_desired", "SA mean (desired)"),
    ]:
        vals = summary_df[col].dropna()
        if len(vals) == 0:
            continue
        if col in ("desired_ratio_raw", "unique_ratio_desired"):
            print(f"{label:40s} {vals.mean():>9.1%} {vals.min():>9.1%} {vals.max():>9.1%}")
        else:
            print(f"{label:40s} {vals.mean():>10.1f} {vals.min():>10.1f} {vals.max():>10.1f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
