# Visualization Regeneration Plan (Updated)

> **Goal:** Regenerate all HTML visualizations using existing scripts with new multi-criteria ranking data. No new HTML generators — refactor and reuse existing code. Debug with VSCode integrated browser.

---

## Existing Scripts → Outputs

| Script | Generates | Libraries | Status |
|--------|-----------|-----------|--------|
| `visualize_enhanced.py` | Per-molecule 3D viewers + dashboard | 3Dmol.js | Works, needs new data |
| `score_candidates.py` | `scoring_dashboard.html` | Plain HTML/CSS | Needs v2 composite columns |
| `rank_comparison.py` | `rank_comparison.html` | Plotly | Needs update with IFP data |
| `interactive_chemical_space.py` | Chemical space UMAP | Plotly + RDKit.js | Bug: models not showing |
| `visualize_docking_box.py` | `box_visualization.html` | 3Dmol.js | No changes needed |
| `ccr2_interactive_dashboard.html` | Training dashboard | Plotly + SmilesDrawer | No changes needed |

## What Needs Work (priority order)

### Step 1: Prepare PDB complexes for top-30 × 3 tracks
**Problem:** Glide top-30 has 22/30 missing PDB, Consensus has 20/30 missing.
**Script:** Small helper that extracts specific molecules from glide_bulk SDF → PDB complex.
**Output:** Per-molecule dirs ONLY for ~70 unique top-30 molecules (not 52K!).
**Location:** `final_ranking/pose_data/` (new, clean directory)
**Time:** ~2 min on head node.

### Step 2: Regenerate scoring dashboards (update `score_candidates.py`)
**Problem:** Old dashboard shows IFP+dock composite. Need v2 composite + LE + SA + ROCS columns.
**Approach:** Add option to `score_candidates.py` OR create thin wrapper that reads `final_*.tsv` and generates dashboard HTML using the same template.
**Output:** `final_ranking/dashboard_vina.html`, `dashboard_glide.html`, `dashboard_consensus.html`
**Fix:** Table sorted by composite_v2 descending (best on top).

### Step 3: Regenerate 3D pose viewers (reuse `visualize_enhanced.py`)
**Input:** `final_ranking/pose_data/` (from Step 1) + PLIP profiles
**Output:**
  - `final_ranking/viz_vina_top30/` → 30 HTML + dashboard
  - `final_ranking/viz_glide_top30/` → 30 HTML + dashboard
  - `final_ranking/viz_consensus_top30/` → 30 HTML + dashboard
**Command:** `python visualize_enhanced.py --redock-dir pose_data --interaction-dir profiles --out-dir viz_*_top30 --source *`

### Step 4: Debug chemical space viz (models not showing)
**File:** `interactive_chemical_space_diversity_25pct.html` (27 MB)
**Bug:** "models not showing" — likely RDKit.js WASM loading failure or data issue.
**Debug:** Open in VSCode browser → check console → fix in `interactive_chemical_space.py` → regenerate.

### Step 5: Update rank_comparison.html with IFP + v2 data
**Script:** `rank_comparison.py` — update to include IFP and multi-criteria composite.
**May skip** if not needed for thesis (lower priority).

### Step 6: Clean up old visualization directories
**Remove or archive:**
  - `plip_analysis/visualizations/` (old top-30, replaced)
  - `plip_analysis/visualizations_enhanced/` (old top-30, replaced)
  - `plip_analysis/visualizations_vina/` (old per-engine, replaced)
  - `plip_analysis/visualizations_glide/` (old per-engine, replaced)
  - `plip_analysis/redock_top200/` (superseded by full pipeline)
  - `plip_analysis/redock_top2000/visualizations_enhanced/` (superseded)
  - `plip_analysis/scored_candidates/` (old 30-molecule scoring)

**Keep:**
  - `plip_analysis/full_pipeline/` (current production data)
  - `plip_analysis/full_pipeline/final_ranking/` (multi-criteria results)
  - `docking/box_visualization.html` (static, no change)
  - `postprocess/ccr2_interactive_dashboard.html` (training dashboard, no change)

---

## Collaborative Debug Workflow (VSCode Browser)

For each HTML file after regeneration:
1. Claude generates/regenerates the HTML
2. User opens in VSCode integrated browser (Copilot)
3. User reports: what's broken (blank page, missing molecules, sort not working, etc.)
4. Claude reads the HTML source, identifies bug, fixes
5. User reloads and confirms

**Priority for debugging:**
1. Scoring dashboards (sortable table, molecule rendering)
2. 3D pose viewers (3Dmol.js loading, interaction lines)
3. Chemical space viz (RDKit.js WASM, Plotly traces)
