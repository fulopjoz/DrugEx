#!/usr/bin/env python3
"""Score and rank molecules across three tracks: Vina, Glide, and Consensus.

Reads PLIP interaction profiles from both Vina and Glide pose sets, combines
with docking scores and ECR ranking, and produces three ranked lists:

  1. Vina track:      IFP(Vina poses) + Vina docking score
  2. Glide track:     IFP(Glide poses) + Glide docking score
  3. Consensus track: max(IFP_vina, IFP_glide) + ECR rank

Each track produces a scored_candidates.tsv and scoring_dashboard.html.

Usage:
    python score_three_tracks.py \
        --vina-profiles  redock_top10k/vina_profiles \
        --glide-profiles redock_top10k/glide_profiles \
        --vina-redock    redock_top10k/vina_merged \
        --glide-redock   redock_top10k/glide_poses \
        --ecr-tsv        redock_top10k/ecr_top10000.tsv \
        --out-dir        redock_top10k/three_track_results
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_plip import (
    TIER_A_ANCHORS, TIER_B_HYDROPHOBIC, TIER_C_AROMATIC, TIER_D_BONUS,
    WEIGHT_A, WEIGHT_B, WEIGHT_C, WEIGHT_D, IFP_MAX,
)


def load_profiles(profile_dir: Path) -> dict[str, dict]:
    """Load PLIP profiles from a directory (supports both JSON and JSONL).

    Scans for:
      - *_interactions.json (v1: one file per molecule)
      - *.jsonl (v2: one JSON per line per molecule)
    """
    profiles = {}

    # v1: per-molecule JSON files
    for f in sorted(profile_dir.glob("*_interactions.json")):
        mol_id = f.stem.replace("_interactions", "")
        profiles[mol_id] = json.loads(f.read_text())

    # v2: JSONL files (one JSON per line)
    for f in sorted(profile_dir.glob("*.jsonl")):
        for line in f.read_text().strip().split("\n"):
            if not line:
                continue
            record = json.loads(line)
            mol_id = record.get("mol_id", "")
            if mol_id:
                profiles[mol_id] = record

    return profiles


def load_redock_scores(redock_dir: Path) -> dict[str, float]:
    """Load docking scores from TSV files.

    Supports both v1 (single redock_results.tsv) and v2 (multiple *_results.tsv).
    """
    scores = {}
    # Find all result TSV files
    tsv_files = list(redock_dir.glob("*_results.tsv"))
    single = redock_dir / "redock_results.tsv"
    if single.exists() and single not in tsv_files:
        tsv_files.append(single)
    if not tsv_files:
        return scores
    for tsv in tsv_files:
        for row in csv.DictReader(open(tsv), delimiter="\t"):
            mol_id = row.get("mol_id", "")
            for col in ("best_score", "vina_score", "glide_score"):
                if col in row and row[col]:
                    try:
                        scores[mol_id] = float(row[col])
                        break
                    except ValueError:
                        pass
    return scores


def compute_ifp(interactions: list[dict]) -> dict:
    """Compute tiered IFP score from raw interaction list."""
    found = set()
    for ix in interactions:
        found.add((ix["type"], ix["resname"], ix["resnr"]))

    tier_a = found & TIER_A_ANCHORS
    tier_b = found & TIER_B_HYDROPHOBIC
    tier_c = found & TIER_C_AROMATIC
    tier_d = found & TIER_D_BONUS

    ifp_raw = (len(tier_a) * WEIGHT_A + len(tier_b) * WEIGHT_B
               + len(tier_c) * WEIGHT_C + len(tier_d) * WEIGHT_D)

    return {
        "n_a": len(tier_a), "n_b": len(tier_b),
        "n_c": len(tier_c), "n_d": len(tier_d),
        "ifp_raw": round(ifp_raw, 1),
        "ifp_norm": round(ifp_raw / IFP_MAX, 4) if IFP_MAX > 0 else 0,
    }


def score_track(
    profiles: dict[str, dict],
    dock_scores: dict[str, float],
    track_name: str,
) -> list[dict]:
    """Score a single track: IFP + docking score → composite."""
    candidates = []
    for mol_id, prof in profiles.items():
        ifp = compute_ifp(prof["interactions"])
        dock_score = dock_scores.get(mol_id)
        candidates.append({
            "mol_id": mol_id,
            "track": track_name,
            "dock_score": dock_score,
            **ifp,
        })

    # Normalize docking scores
    scored = [c for c in candidates if c["dock_score"] is not None]
    if scored:
        worst = max(c["dock_score"] for c in scored)
        best = min(c["dock_score"] for c in scored)
        rng = worst - best if worst != best else 1.0
        for c in scored:
            c["dock_norm"] = round((worst - c["dock_score"]) / rng, 4)
            c["composite"] = round(0.5 * c["ifp_norm"] + 0.5 * c["dock_norm"], 4)
        for c in candidates:
            if c["dock_score"] is None:
                c["dock_norm"] = 0.0
                c["composite"] = round(0.5 * c["ifp_norm"], 4)
    else:
        for c in candidates:
            c["dock_norm"] = 0.0
            c["composite"] = round(c["ifp_norm"] * 0.5, 4)

    candidates.sort(key=lambda c: c.get("composite", 0), reverse=True)
    for i, c in enumerate(candidates):
        c["rank"] = i + 1

    return candidates


def score_consensus(
    vina_profiles: dict[str, dict],
    glide_profiles: dict[str, dict],
    ecr_data: dict[str, dict],
) -> list[dict]:
    """Consensus track: max IFP from either engine + ECR rank.

    For each molecule, takes the BETTER IFP (Vina or Glide poses),
    combines with normalized ECR rank.
    """
    all_mols = set(vina_profiles.keys()) | set(glide_profiles.keys())
    all_mols = all_mols & set(ecr_data.keys())  # must have ECR

    candidates = []
    for mol_id in all_mols:
        # Compute IFP from both engines
        ifp_v = {"ifp_raw": 0, "ifp_norm": 0, "n_a": 0, "n_b": 0, "n_c": 0, "n_d": 0}
        ifp_g = dict(ifp_v)

        if mol_id in vina_profiles:
            ifp_v = compute_ifp(vina_profiles[mol_id]["interactions"])
        if mol_id in glide_profiles:
            ifp_g = compute_ifp(glide_profiles[mol_id]["interactions"])

        # Take the BETTER IFP
        if ifp_v["ifp_raw"] >= ifp_g["ifp_raw"]:
            best_ifp = ifp_v
            ifp_source = "vina"
        else:
            best_ifp = ifp_g
            ifp_source = "glide"

        ecr = ecr_data[mol_id]

        candidates.append({
            "mol_id": mol_id,
            "track": "consensus",
            "ifp_source": ifp_source,
            "ifp_vina": ifp_v["ifp_raw"],
            "ifp_glide": ifp_g["ifp_raw"],
            "ecr_rank": ecr["ecr_rank"],
            "ecr_score": ecr["ecr"],
            "vina_score": ecr.get("vina_score"),
            "glide_score": ecr.get("glide_score"),
            **best_ifp,
        })

    # Normalize ECR rank (rank 1 = best → 1.0, rank N → 0.0)
    if candidates:
        max_rank = max(c["ecr_rank"] for c in candidates)
        for c in candidates:
            c["ecr_norm"] = round(1.0 - (c["ecr_rank"] - 1) / max(1, max_rank - 1), 4)
            c["composite"] = round(0.5 * c["ifp_norm"] + 0.5 * c["ecr_norm"], 4)

    candidates.sort(key=lambda c: c["composite"], reverse=True)
    for i, c in enumerate(candidates):
        c["rank"] = i + 1

    return candidates


def write_track(candidates: list[dict], out_dir: Path, track_name: str):
    """Write scored candidates TSV for one track."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = out_dir / f"scored_{track_name}.tsv"

    if not candidates:
        print(f"  {track_name}: no candidates")
        return

    keys = list(candidates[0].keys())
    with open(tsv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        w.writeheader()
        w.writerows(candidates)

    print(f"  {track_name}: {len(candidates)} candidates → {tsv_path}")
    print(f"    Top 5:")
    for c in candidates[:5]:
        ifp = c.get("ifp_raw", 0)
        comp = c.get("composite", 0)
        dock = c.get("dock_score", c.get("vina_score", "N/A"))
        print(f"      #{c['rank']:3d} {c['mol_id']:<20s} "
              f"composite={comp:.3f} IFP={ifp:.1f}/19 "
              f"A={c['n_a']}/3 B={c['n_b']}/4")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vina-profiles", type=Path, required=True)
    p.add_argument("--glide-profiles", type=Path, required=True)
    p.add_argument("--vina-redock", type=Path, required=True,
                   help="Vina re-dock merged dir (has redock_results.tsv)")
    p.add_argument("--glide-redock", type=Path, required=True,
                   help="Glide pose extraction dir (has redock_results.tsv)")
    p.add_argument("--ecr-tsv", type=Path, required=True,
                   help="ECR ranked list")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    out_dir = args.out_dir.resolve()

    print("=" * 60)
    print("THREE-TRACK SCORING")
    print("=" * 60)

    # Load profiles
    print("\nLoading Vina PLIP profiles...")
    vina_profs = load_profiles(args.vina_profiles.resolve())
    print(f"  {len(vina_profs)} profiles")

    print("Loading Glide PLIP profiles...")
    glide_profs = load_profiles(args.glide_profiles.resolve())
    print(f"  {len(glide_profs)} profiles")

    # Load docking scores
    vina_scores = load_redock_scores(args.vina_redock.resolve())
    glide_scores = load_redock_scores(args.glide_redock.resolve())
    print(f"\nVina scores: {len(vina_scores)}, Glide scores: {len(glide_scores)}")

    # Load ECR data
    ecr_data = {}
    with open(args.ecr_tsv) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ecr_data[row["dock_id"]] = {
                "ecr_rank": int(row["ecr_rank"]),
                "ecr": float(row["ecr"]),
                "vina_score": float(row["vina_score"]),
                "glide_score": float(row["glide_score"]),
            }
    print(f"ECR data: {len(ecr_data)} molecules")

    # Score Track 1: Vina
    print("\n--- Track 1: VINA ---")
    vina_candidates = score_track(vina_profs, vina_scores, "vina")
    write_track(vina_candidates, out_dir, "vina")

    # Score Track 2: Glide
    print("\n--- Track 2: GLIDE ---")
    glide_candidates = score_track(glide_profs, glide_scores, "glide")
    write_track(glide_candidates, out_dir, "glide")

    # Score Track 3: Consensus
    print("\n--- Track 3: CONSENSUS ---")
    consensus_candidates = score_consensus(vina_profs, glide_profs, ecr_data)
    write_track(consensus_candidates, out_dir, "consensus")

    # Cross-track comparison
    print(f"\n{'=' * 60}")
    print("CROSS-TRACK AGREEMENT")
    print("=" * 60)
    v_top50 = {c["mol_id"] for c in vina_candidates[:50]}
    g_top50 = {c["mol_id"] for c in glide_candidates[:50]}
    c_top50 = {c["mol_id"] for c in consensus_candidates[:50]}

    vg = v_top50 & g_top50
    vc = v_top50 & c_top50
    gc = g_top50 & c_top50
    vgc = v_top50 & g_top50 & c_top50

    print(f"  Top-50 overlap:")
    print(f"    Vina ∩ Glide:     {len(vg)}")
    print(f"    Vina ∩ Consensus: {len(vc)}")
    print(f"    Glide ∩ Consensus:{len(gc)}")
    print(f"    All three:        {len(vgc)}")

    print(f"\n  Results in: {out_dir}")


if __name__ == "__main__":
    main()
