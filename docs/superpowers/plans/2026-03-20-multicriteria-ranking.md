# Multi-Criteria Ranking Pipeline

> **Goal:** Replace the binary IFP + docking score composite (which can't discriminate among 216 IFP=19 molecules) with a literature-backed multi-criteria ranking that incorporates ligand efficiency, synthetic accessibility, ROCS similarity, and scaffold diversity.

**Literature justification:**
- Ligand Efficiency: Schultes et al. (2010) Drug Discov. Today Technol. 7:e157 (228 citations)
- Weighted IFP: Jasper et al. (2018) J. Cheminform. 10:264 — PADIF with weighted scoring
- Weighted SIFt: Nandigam et al. (2009) JCIM 49:1185 — w-SIFt with importance weights
- Hit selection cascade: Atanasova et al. (2022) Molecules 27:3139 — multi-filter + clustering + visual

**New composite formula:**
```
composite_v2 = 0.30 × IFP_norm          # interaction quality (0-1)
             + 0.25 × dock_norm          # docking score (ECR rank, 0-1)
             + 0.20 × LE_norm            # ligand efficiency (penalizes bloat)
             + 0.15 × SA_norm            # synthetic accessibility (lower = easier)
             + 0.10 × ROCS_norm          # shape similarity to known actives
```

**Implementation:** Single Python script, no PBS needed, reads existing data.

---

### Task 1: `score_multicriteria.py`

- Create: `ccr2_gen/docking/plip_analysis/score_multicriteria.py`

Input:
- `scored_consensus.tsv` (from three-track scoring)
- `scored_vina.tsv`, `scored_glide.tsv`
- `filtered_molecules.tsv` (library with SA, ROCS, HeavyAtoms, scaffold)

Output per track:
- `final_vina.tsv`, `final_glide.tsv`, `final_consensus.tsv`
- `final_top50_diverse.tsv` (clustered, best per cluster)
- `ranking_summary.json` (statistics)

### Task 2: Verify on data
### Task 3: Document literature justification in script docstring
