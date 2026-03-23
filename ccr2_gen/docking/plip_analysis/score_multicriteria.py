#!/usr/bin/env python3
"""Multi-criteria ranking for VS hit selection.

Replaces the binary IFP + docking score composite with a weighted
multi-criteria score that incorporates:

  1. IFP score (interaction fingerprint quality, 0-19 normalized)
  2. Docking score (ECR rank from Vina + Glide consensus)
  3. Ligand Efficiency (LE = -score / n_heavy_atoms)
  4. Synthetic Accessibility (SA from RL generation, lower = easier)
  5. ROCS shape similarity (TanimotoCombo to known CCR2 ligands)

Literature justification:
  - Ligand Efficiency: Schultes et al. (2010) Drug Discov. Today Technol.
    7:e157-e162. doi:10.1016/j.ddtec.2010.11.003 (228 citations).
    "Binding energy/non-hydrogen atom" as standard hit selection metric.
  - Weighted IFP: Jasper et al. (2018) J. Cheminform. 10:264.
    doi:10.1186/s13321-018-0264-0. PADIF uses "weighting factors reflecting
    the relative frequency of a specific interaction in the references."
  - Multi-filter cascade: Atanasova et al. (2022) Molecules 27:3139.
    doi:10.3390/molecules27103139. Docking → ADME → clustering → visual
    inspection → MD → experimental testing.
  - w-SIFt: Nandigam et al. (2009) JCIM 49:1185-1192.
    doi:10.1021/ci800466n. Weighted interaction fingerprint for target-
    specific scoring.

Composite formula:
  composite_v2 = 0.30 × IFP_norm
               + 0.25 × dock_norm  (ECR rank)
               + 0.20 × LE_norm    (ligand efficiency)
               + 0.15 × SA_norm    (synthetic accessibility)
               + 0.10 × ROCS_norm  (shape similarity)

Usage:
    python score_multicriteria.py \
        --three-track-dir full_pipeline/three_track_results \
        --library-tsv full_pipeline/filtered_molecules.tsv \
        --out-dir full_pipeline/final_ranking
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina


# ── Composite weights (literature-informed) ─────────────────────────────
W_IFP = 0.30    # Interaction quality
W_DOCK = 0.25   # Docking score (consensus)
W_LE = 0.20     # Ligand efficiency (penalizes molecular bloat)
W_SA = 0.15     # Synthetic accessibility (ease of synthesis)
W_ROCS = 0.10   # Shape similarity to known CCR2 ligands

IFP_MAX = 19.0  # Maximum tiered IFP score


def normalize_column(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Min-max normalize to [0, 1]. Handles NaN by filling with 0."""
    vals = series.copy()
    vmin, vmax = vals.min(), vals.max()
    if vmin == vmax:
        return pd.Series(0.5, index=series.index)
    if higher_is_better:
        normed = (vals - vmin) / (vmax - vmin)
    else:
        normed = (vmax - vals) / (vmax - vmin)
    return normed.fillna(0.0)


def compute_ligand_efficiency(dock_score: pd.Series, n_heavy: pd.Series) -> pd.Series:
    """LE = -dock_score / n_heavy_atoms (Schultes et al. 2010)."""
    return (-dock_score / n_heavy).replace([np.inf, -np.inf], np.nan)


def get_best_rocs(lib_row: pd.Series) -> float:
    """Get the best ROCS TanimotoCombo across all backends."""
    rocs_cols = [c for c in lib_row.index if "ROCS" in c or "TanimotoCombo" in c]
    vals = [lib_row[c] for c in rocs_cols if pd.notna(lib_row.get(c))]
    return max(vals) if vals else np.nan


def butina_cluster_smiles(smiles_list: list[str], threshold: float = 0.5) -> list[int]:
    """Butina clustering on ECFP4 fingerprints. Returns cluster labels."""
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
            valid_idx.append(i)

    if len(fps) < 2:
        return list(range(len(smiles_list)))

    # Condensed distance matrix
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])

    clusters = Butina.ClusterData(dists, len(fps), threshold, isDistData=True)

    labels = [-1] * len(smiles_list)
    for cluster_id, members in enumerate(clusters):
        for member_fp_idx in members:
            labels[valid_idx[member_fp_idx]] = cluster_id
    return labels


def score_track(track_df: pd.DataFrame, lib_df: pd.DataFrame,
                track_name: str) -> pd.DataFrame:
    """Apply multi-criteria scoring to one track."""
    # Merge with library for SA, ROCS, HeavyAtoms, scaffold
    df = track_df.merge(
        lib_df[["dock_id", "SA", "HeavyAtoms", "std_SMILES", "scaffold",
                "generic_scaffold", "MW", "LogP", "TPSA"]],
        left_on="mol_id", right_on="dock_id", how="left",
    )

    # Get best ROCS per molecule
    rocs_cols = [c for c in lib_df.columns if "ROCS" in c or "TanimotoCombo" in c]
    if rocs_cols:
        rocs_lookup = lib_df.set_index("dock_id")[rocs_cols].max(axis=1).to_dict()
        df["rocs_best"] = df["mol_id"].map(rocs_lookup)

    # Get best docking score for LE
    if "vina_score" in df.columns and "glide_score" in df.columns:
        df["best_dock"] = df[["vina_score", "glide_score"]].min(axis=1)
    elif "dock_score" in df.columns:
        df["best_dock"] = df["dock_score"]
    else:
        df["best_dock"] = df.get("vina_score", df.get("glide_score", np.nan))

    # Compute Ligand Efficiency
    df["LE"] = compute_ligand_efficiency(df["best_dock"], df["HeavyAtoms"])

    # Normalize all components
    df["ifp_norm"] = df["ifp_raw"] / IFP_MAX if "ifp_raw" in df.columns else 0.0

    # ECR rank normalization (rank 1 = best → 1.0)
    if "ecr_rank" in df.columns:
        max_rank = df["ecr_rank"].max()
        df["dock_norm"] = 1.0 - (df["ecr_rank"] - 1) / max(1, max_rank - 1)
    elif "composite" in df.columns:
        df["dock_norm"] = normalize_column(df["composite"])
    else:
        df["dock_norm"] = 0.5

    df["LE_norm"] = normalize_column(df["LE"], higher_is_better=True)
    # SA: lower = easier to synthesize → higher_is_better=False
    df["SA_norm"] = normalize_column(df["SA"], higher_is_better=False)
    df["ROCS_norm"] = normalize_column(df.get("rocs_best", pd.Series(dtype=float)),
                                        higher_is_better=True)

    # Multi-criteria composite
    df["composite_v2"] = (
        W_IFP * df["ifp_norm"]
        + W_DOCK * df["dock_norm"]
        + W_LE * df["LE_norm"]
        + W_SA * df["SA_norm"]
        + W_ROCS * df["ROCS_norm"]
    ).round(4)

    # Rank
    df = df.sort_values("composite_v2", ascending=False).reset_index(drop=True)
    df["rank_v2"] = range(1, len(df) + 1)

    return df


def select_diverse_top(df: pd.DataFrame, top_n: int = 100,
                       cluster_threshold: float = 0.5) -> pd.DataFrame:
    """Select diverse top candidates by Butina clustering."""
    top = df.head(top_n).copy()

    if "std_SMILES" not in top.columns:
        return top

    smiles = top["std_SMILES"].fillna("").tolist()
    labels = butina_cluster_smiles(smiles, threshold=cluster_threshold)
    top["diversity_cluster"] = labels

    # Best per cluster
    diverse = (
        top[top["diversity_cluster"] >= 0]
        .sort_values("composite_v2", ascending=False)
        .drop_duplicates(subset=["diversity_cluster"], keep="first")
        .sort_values("composite_v2", ascending=False)
        .reset_index(drop=True)
    )
    diverse["diverse_rank"] = range(1, len(diverse) + 1)
    return diverse


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--three-track-dir", type=Path, required=True,
                   help="Directory with scored_vina.tsv, scored_glide.tsv, scored_consensus.tsv")
    p.add_argument("--library-tsv", type=Path, required=True,
                   help="Filtered molecules TSV (with SA, ROCS, HeavyAtoms, scaffold)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--top-diverse", type=int, default=50,
                   help="Number of diverse top candidates to select (default: 50)")
    args = p.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MULTI-CRITERIA RANKING")
    print("=" * 70)
    print(f"Weights: IFP={W_IFP}, Dock={W_DOCK}, LE={W_LE}, SA={W_SA}, ROCS={W_ROCS}")
    print()

    lib = pd.read_csv(args.library_tsv, sep="\t", low_memory=False)
    print(f"Library: {len(lib)} molecules")

    summary = {}

    for track_name in ["vina", "glide", "consensus"]:
        path = args.three_track_dir / f"scored_{track_name}.tsv"
        if not path.exists():
            print(f"\n  {track_name}: SKIPPED (file not found)")
            continue

        print(f"\n--- {track_name.upper()} track ---")
        track_df = pd.read_csv(path, sep="\t")
        print(f"  Input: {len(track_df)} molecules")

        # Score
        scored = score_track(track_df, lib, track_name)

        # Save full ranked list
        out_cols = ["rank_v2", "mol_id", "composite_v2",
                    "ifp_raw", "ifp_norm", "dock_norm", "LE", "LE_norm",
                    "SA", "SA_norm", "rocs_best", "ROCS_norm",
                    "best_dock", "MW", "LogP", "HeavyAtoms",
                    "scaffold", "std_SMILES"]
        # Only include columns that exist
        out_cols = [c for c in out_cols if c in scored.columns]

        out_path = out_dir / f"final_{track_name}.tsv"
        scored[out_cols].to_csv(out_path, sep="\t", index=False)
        print(f"  Full ranking: {out_path} ({len(scored)} molecules)")

        # Top 5
        print(f"  Top 5:")
        for _, r in scored.head(5).iterrows():
            ifp = r.get("ifp_raw", 0)
            le = r.get("LE", 0)
            sa = r.get("SA", 0)
            rocs = r.get("rocs_best", 0)
            print(f"    #{int(r['rank_v2']):3d} {r['mol_id']:<20s} "
                  f"v2={r['composite_v2']:.3f} IFP={ifp:.0f} "
                  f"LE={le:.3f} SA={sa:.1f} ROCS={rocs:.2f}")

        # IFP=19 distribution in new ranking
        perfect = scored[scored.get("ifp_raw", pd.Series()) == 19.0]
        if len(perfect) > 0:
            print(f"\n  IFP=19 molecules: {len(perfect)}")
            print(f"    Old: all tied at composite~0.5-1.0 (undiscriminated)")
            print(f"    New rank range: #{int(perfect['rank_v2'].min())}-#{int(perfect['rank_v2'].max())}")
            print(f"    v2 score range: [{perfect['composite_v2'].min():.3f}, {perfect['composite_v2'].max():.3f}]")
            print(f"    Spread: {perfect['composite_v2'].max() - perfect['composite_v2'].min():.3f}")

        # Diverse selection
        diverse = select_diverse_top(scored, top_n=100,
                                      cluster_threshold=0.5)
        diverse_path = out_dir / f"final_{track_name}_diverse_top{len(diverse)}.tsv"
        diverse_cols = [c for c in out_cols + ["diversity_cluster", "diverse_rank"]
                        if c in diverse.columns]
        diverse[diverse_cols].to_csv(diverse_path, sep="\t", index=False)
        print(f"\n  Diverse top: {diverse_path} ({len(diverse)} clusters)")

        # Track summary
        summary[track_name] = {
            "n_scored": int(len(scored)),
            "n_ifp_19": int(len(perfect)) if len(perfect) > 0 else 0,
            "top1_mol": scored.iloc[0]["mol_id"],
            "top1_v2": round(float(scored.iloc[0]["composite_v2"]), 4),
            "n_diverse": int(len(diverse)),
            "ifp19_rank_range": [int(perfect["rank_v2"].min()), int(perfect["rank_v2"].max())]
                                if len(perfect) > 0 else None,
            "weights": {"IFP": W_IFP, "Dock": W_DOCK, "LE": W_LE, "SA": W_SA, "ROCS": W_ROCS},
        }

    # Cross-track agreement
    print(f"\n{'=' * 70}")
    print("CROSS-TRACK AGREEMENT (v2 composite)")
    print("=" * 70)
    top50_sets = {}
    for track in ["vina", "glide", "consensus"]:
        path = out_dir / f"final_{track}.tsv"
        if path.exists():
            df = pd.read_csv(path, sep="\t")
            top50_sets[track] = set(df.head(50)["mol_id"])

    if len(top50_sets) >= 2:
        for a, b in [("vina", "glide"), ("vina", "consensus"), ("glide", "consensus")]:
            if a in top50_sets and b in top50_sets:
                overlap = top50_sets[a] & top50_sets[b]
                print(f"  {a} ∩ {b} (top-50): {len(overlap)}")
        if all(t in top50_sets for t in ["vina", "glide", "consensus"]):
            all3 = top50_sets["vina"] & top50_sets["glide"] & top50_sets["consensus"]
            print(f"  All three: {len(all3)}")

    # Save summary
    summary_path = out_dir / "ranking_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  Summary: {summary_path}")
    print(f"  All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
