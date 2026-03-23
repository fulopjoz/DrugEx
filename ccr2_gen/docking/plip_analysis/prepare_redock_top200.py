#!/usr/bin/env python3
"""Prepare top 200 ECR-ranked candidates for re-docking + PLIP.

Loads existing Vina + Glide docking results, applies MW + ring count filters,
computes ECR consensus ranking, and outputs:
  - ecr_top200.tsv: ranked candidate list for redock_top10.py
  - chunks/chunk_NN.tsv: 20 chunks of 10 for PBS array submission

Uses the same ECR formula as rank_comparison.py (Palacio-Rodriguez et al., 2019):
    ECR_i = exp(-rank_vina_i / sigma_v) + exp(-rank_glide_i / sigma_g)
    sigma = N / 5

Usage:
    python prepare_redock_top200.py \
        --vina-tsv  ../analysis_vina/docking_results_merged.tsv \
        --glide-tsv ../analysis_glide_sp/docking_results_merged.tsv \
        --out-dir   redock_top200 \
        --top-n 200 --mw-cutoff 546 --max-rings 5
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def load_engine(path: Path, score_col: str, mw_cutoff: float,
                max_rings: int) -> dict[str, dict]:
    """Load docking results, apply MW + ring filters."""
    rows = {}
    n_total = 0
    n_mw_fail = 0
    n_ring_fail = 0
    for row in csv.DictReader(open(path), delimiter="\t"):
        n_total += 1
        try:
            score = float(row[score_col])
            mw = float(row["MW"])
        except (ValueError, KeyError):
            continue
        if mw > mw_cutoff:
            n_mw_fail += 1
            continue

        # Ring count filter
        smiles = row.get("std_SMILES", row.get("SMILES", ""))
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        if n_rings > max_rings:
            n_ring_fail += 1
            continue

        dock_id = row["dock_id"]
        rows[dock_id] = {
            "score": score, "mw": mw, "smiles": smiles,
            "n_rings": n_rings,
        }

    print(f"  {path.name}: {n_total} total, {n_mw_fail} MW>{mw_cutoff}, "
          f"{n_ring_fail} rings>{max_rings}, {len(rows)} pass")
    return rows


def compute_ecr(vina: dict, glide: dict, common_ids: set) -> list[dict]:
    """ECR consensus ranking (Palacio-Rodriguez 2019)."""
    # Assign ranks per engine (lower score = better = rank 1)
    v_sorted = sorted(common_ids, key=lambda k: vina[k]["score"])
    g_sorted = sorted(common_ids, key=lambda k: glide[k]["score"])
    v_ranks = {mol_id: rank for rank, mol_id in enumerate(v_sorted, 1)}
    g_ranks = {mol_id: rank for rank, mol_id in enumerate(g_sorted, 1)}

    n = len(common_ids)
    sigma_v = n / 5.0
    sigma_g = n / 5.0

    results = []
    for mol_id in common_ids:
        rv = v_ranks[mol_id]
        rg = g_ranks[mol_id]
        ecr = math.exp(-rv / sigma_v) + math.exp(-rg / sigma_g)
        results.append({
            "dock_id": mol_id,
            "ecr": ecr,
            "ecr_rank": 0,  # assigned below
            "vina_rank": rv,
            "glide_rank": rg,
            "vina_score": vina[mol_id]["score"],
            "glide_score": glide[mol_id]["score"],
            "mw": vina[mol_id]["mw"],
            "smiles": vina[mol_id]["smiles"],
            "n_rings": vina[mol_id]["n_rings"],
        })

    results.sort(key=lambda r: r["ecr"], reverse=True)
    for i, r in enumerate(results):
        r["ecr_rank"] = i + 1

    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vina-tsv", type=Path, required=True)
    p.add_argument("--glide-tsv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--top-n", type=int, default=200)
    p.add_argument("--mw-cutoff", type=float, default=546.0)
    p.add_argument("--max-rings", type=int, default=5,
                   help="Max ring count (default: 5 = max ref CCX140=4 + 1)")
    p.add_argument("--chunk-size", type=int, default=10,
                   help="Molecules per PBS array chunk (default: 10)")
    args = p.parse_args()

    out_dir = args.out_dir.resolve()
    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Vina results...")
    vina = load_engine(args.vina_tsv, "vina_score", args.mw_cutoff, args.max_rings)
    print("Loading Glide results...")
    glide = load_engine(args.glide_tsv, "r_i_docking_score", args.mw_cutoff, args.max_rings)

    common = set(vina.keys()) & set(glide.keys())
    print(f"\nCommon molecules (both engines, all filters): {len(common)}")

    ecr = compute_ecr(vina, glide, common)
    top = ecr[:args.top_n]
    print(f"Top {len(top)} by ECR selected")

    # Save full ranked list
    cols = ["ecr_rank", "dock_id", "ecr", "vina_rank", "glide_rank",
            "vina_score", "glide_score", "mw", "n_rings", "smiles"]
    tsv_path = out_dir / f"ecr_top{args.top_n}.tsv"
    with open(tsv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for row in top:
            out = dict(row)
            out["ecr"] = f"{row['ecr']:.6e}"
            out["vina_score"] = f"{row['vina_score']:.3f}"
            out["glide_score"] = f"{row['glide_score']:.3f}"
            out["mw"] = f"{row['mw']:.1f}"
            w.writerow(out)
    print(f"Saved: {tsv_path}")

    # Split into chunks for PBS array
    n_chunks = (len(top) + args.chunk_size - 1) // args.chunk_size
    for ci in range(n_chunks):
        start = ci * args.chunk_size
        end = min(start + args.chunk_size, len(top))
        chunk = top[start:end]
        chunk_path = chunk_dir / f"chunk_{ci + 1:03d}.tsv"
        with open(chunk_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            for row in chunk:
                out = dict(row)
                out["ecr"] = f"{row['ecr']:.6e}"
                out["vina_score"] = f"{row['vina_score']:.3f}"
                out["glide_score"] = f"{row['glide_score']:.3f}"
                out["mw"] = f"{row['mw']:.1f}"
                w.writerow(out)
    print(f"Split into {n_chunks} chunks of {args.chunk_size} in {chunk_dir}")

    # Print top 10 for quick review
    print(f"\n{'=' * 80}")
    print(f"TOP 10 ECR CANDIDATES (after MW≤{args.mw_cutoff} + rings≤{args.max_rings})")
    print("=" * 80)
    for r in top[:10]:
        print(f"  #{r['ecr_rank']:3d}  {r['dock_id']:<20s}  "
              f"Vina={r['vina_score']:+.2f}  Glide={r['glide_score']:+.2f}  "
              f"MW={r['mw']:.0f}  rings={r['n_rings']}  "
              f"ECR={r['ecr']:.3e}")
    print("=" * 80)


if __name__ == "__main__":
    main()
