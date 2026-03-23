# CCR2 Pipeline Correction Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the post-generation pipeline so top 200 ECR-ranked molecules get PLIP analysis (instead of only 30), add a ring count filter, and declutter the chemical space visualization.

**Architecture:** Three surgical fixes to the existing pipeline. No new frameworks. Existing docking results (53K Vina + 52K Glide) are reused — we only re-dock the top 200 at higher exhaustiveness for PLIP.

**Tech Stack:** Python 3.12, RDKit, Vina, PLIP, PBS (HPC). All existing dependencies.

---

### Task 1: Add ring count filter to prepare_docking_library.py

**Files:**
- Modify: `ccr2_gen/postprocess/prepare_docking_library.py:49-54` (PROPERTY_FILTERS)
- Modify: `ccr2_gen/postprocess/prepare_docking_library.py:57-70` (compute_properties)

- [ ] **Step 1: Add NumRings to PROPERTY_FILTERS dict**

```python
PROPERTY_FILTERS = {
    "MW": (200.0, 600.0),
    "LogP": (-2.0, 6.0),
    "TPSA": (20.0, 140.0),
    "RotBonds": (0, 12),
    "NumRings": (0, 5),  # max ref (CCX140) = 4, +1 margin
}
```

- [ ] **Step 2: Add NumRings to compute_properties()**

Add `"NumRings": rdMolDescriptors.CalcNumRings(mol),` to the return dict.

- [ ] **Step 3: Verify filter works**

Run: `python3 -c "from ccr2_gen.postprocess.prepare_docking_library import compute_properties, passes_filters; print(passes_filters(compute_properties('c1ccccc1')))"`
Expected: True (benzene has 1 ring)

---

### Task 2: Create script to prepare top 200 ECR candidates with ring filter

**Files:**
- Create: `ccr2_gen/docking/plip_analysis/prepare_redock_top200.py`

This script:
1. Loads existing Vina + Glide merged results
2. Applies MW ≤ 546 + NumRings ≤ 5 filters
3. Computes ECR (reuses logic from rank_comparison.py)
4. Saves top 200 as TSV for redock_top10.py input
5. Splits into 20 chunks of 10 for PBS array

- [ ] **Step 1: Write the script**
- [ ] **Step 2: Test on existing data**

Run: `python3 ccr2_gen/docking/plip_analysis/prepare_redock_top200.py --vina-tsv ccr2_gen/docking/analysis_vina/docking_results_merged.tsv --glide-tsv ccr2_gen/docking/analysis_glide_sp/docking_results_merged.tsv --out-dir ccr2_gen/docking/plip_analysis/redock_top200 --top-n 200`

Expected: `redock_top200/ecr_top200.tsv` + 20 chunk files

---

### Task 3: Create PBS array script for re-docking 200

**Files:**
- Create: `ccr2_gen/scripts/34_redock_top200.sh`

PBS array job [1-20], each chunk re-docks 10 molecules at exhaustiveness=32.
Uses existing `redock_top10.py` with `--max-molecules 10`.
Reserve 46 CPUs on 48-core node.

- [ ] **Step 1: Write PBS script**
- [ ] **Step 2: Create post-merge script for PLIP + scoring**

After all 20 array jobs complete:
- Merge redock results into single directory
- Run `run_plip.py` on all 200
- Run `score_candidates.py` on all 200

---

### Task 4: Fix chemical space viz threshold

**Files:**
- Modify: `ccr2_gen/postprocess/interactive_chemical_space.py` (~L1190)

Add `--min-desired-ratio` CLI arg (default: None, for backward compat).
When set, filter models to those with desired_ratio_raw >= threshold before loading molecules.

- [ ] **Step 1: Add CLI arg**

```python
p.add_argument("--min-desired-ratio", type=float, default=None,
               help="Minimum desired ratio to include a model (e.g., 0.25)")
```

- [ ] **Step 2: Add filter logic after loading analysis_summary.tsv**

If `--min-desired-ratio` is set, load `analysis_summary.tsv`, filter model_ids by desired_ratio >= threshold, pass as selected_ids.

- [ ] **Step 3: Re-generate diversity viz**

Run: `python3 ccr2_gen/postprocess/interactive_chemical_space.py --analysis-dir ccr2_gen/postprocess/analysis_20260117 --known-ligands ... --reference-sdf ... --min-desired-ratio 0.25`

---

### Task 5: Run the corrected pipeline

- [ ] **Step 1: Run prepare_redock_top200.py** (local, fast)
- [ ] **Step 2: Submit PBS array** `34_redock_top200.sh` (HPC, ~2-4 hours)
- [ ] **Step 3: After PBS completes: run PLIP + scoring** (local or PBS)
- [ ] **Step 4: Re-generate chemical space viz** with 25% threshold
