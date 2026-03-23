# I/O Optimization: Eliminate Per-Molecule Directory Pattern

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 52K-directory-per-molecule pattern with bulk file I/O, cutting merge+extract from ~3 hours to ~5 minutes.

**Architecture:** Keep the chunk-based PBS array structure, but each chunk produces ONE bulk SDF file (not 10 directories). PLIP reads SDF + receptor → creates complex in memory → writes JSONL output. No merge step needed. No per-molecule directories.

**Tech Stack:** Python 3.12, RDKit, PLIP, PBS. All existing dependencies.

---

## Problem

| Step | Current | Time | Files created |
|------|---------|------|---------------|
| Vina docking (52K) | 5249 chunks × 10 dirs each | 1.5h | 210K files + 52K dirs |
| Merge Vina results | cp -r 52K dirs NFS→NFS | 1h | 210K files copied |
| Glide extraction | Parse CSV → 52K dirs | 40min | 103K files + 52K dirs |
| Copy-back Glide | rsync 52K dirs NFS→NFS | 1h+ | 103K files copied |
| **Total merge+extract** | | **~3h** | **521K filesystem objects** |

## Solution: Bulk File I/O

| Step | New approach | Est. time | Files created |
|------|-------------|-----------|---------------|
| Vina docking | Each chunk writes `chunk_NNN_poses.sdf` + `chunk_NNN_complexes.sdf` | 1.5h (same) | 5249 × 2 = **10,498 files** |
| Merge Vina | Just concatenate TSVs (no file copy!) | **10 sec** | 1 TSV |
| Glide extraction | Write `glide_batch_NN_complexes.sdf` (1 per batch of 1000) | 30min | **53 files** |
| PLIP | Read bulk SDF, create complex in memory, write JSONL | 2h | **53 JSONL files** |
| **Total** | | **same compute, 3h less I/O** | **~10,600 files** |

Reduction: **521K → 10.6K files** (50× fewer filesystem objects).

---

### Task 1: `redock_bulk.py` — Vina docking with bulk SDF output

**Files:**
- Create: `ccr2_gen/docking/plip_analysis/redock_bulk.py`

Replaces `redock_top10.py` for large-scale runs. Key changes:
- Input: same TSV chunk (dock_id + SMILES)
- Output per chunk:
  - `chunk_NNN_results.tsv` — scores (mol_id, best_score, status, n_poses)
  - `chunk_NNN_complexes.sdf` — best-pose complex for each molecule (receptor + ligand combined, or just ligand with receptor path in property)

Actually, PLIP needs a PDB file with both protein and ligand. We can't easily combine them in SDF format. **Revised approach:** Write ligand poses as SDF, create complexes on-the-fly in PLIP.

- `chunk_NNN_poses.sdf` — best docked pose per molecule (ligand only, with dock_id and score as SD properties)
- `chunk_NNN_results.tsv` — scores table

- [ ] **Step 1: Write redock_bulk.py** — same docking logic as redock_top10.py but writes to 2 files per chunk instead of 10 directories.

- [ ] **Step 2: Test on 1 chunk of 10 molecules**

---

### Task 2: `extract_glide_bulk.py` — Glide extraction with bulk SDF output

**Files:**
- Create: `ccr2_gen/docking/plip_analysis/extract_glide_bulk.py`

Replaces `extract_glide_poses.py`. Key changes:
- Input: raw Glide chunk CSVs + molecule ID list
- Output per batch (1000 molecules):
  - `glide_batch_NN_poses.sdf` — best Glide pose per molecule (from MolBlock column)
  - `glide_batch_NN_results.tsv` — scores table

- [ ] **Step 1: Write extract_glide_bulk.py**
- [ ] **Step 2: Test on 1 batch**

---

### Task 3: `run_plip_bulk.py` — PLIP that reads bulk SDF + creates complexes in memory

**Files:**
- Create: `ccr2_gen/docking/plip_analysis/run_plip_bulk.py`

Replaces `run_plip_batch.py`. Key changes:
- Input: `*_poses.sdf` (ligand poses) + receptor PDB path
- For each molecule: load ligand from SDF → combine with receptor in memory → write temp PDB → run PLIP → delete temp
- Output: `plip_batch_NN.jsonl` — one JSON per line (not one file per molecule)

- [ ] **Step 1: Write run_plip_bulk.py**
- [ ] **Step 2: Test on 1 batch**

---

### Task 4: Updated scoring to read JSONL

**Files:**
- Modify: `ccr2_gen/docking/plip_analysis/score_three_tracks.py` (add JSONL reader alongside JSON dir reader)

- [ ] **Step 1: Add `load_profiles_jsonl()` function**
- [ ] **Step 2: Test with output from Task 3**

---

### Task 5: Updated PBS scripts (no merge step)

**Files:**
- Create: `ccr2_gen/scripts/50_vina_dock_bulk.sh` — uses redock_bulk.py
- Create: `ccr2_gen/scripts/51_glide_extract_bulk.sh` — uses extract_glide_bulk.py
- Create: `ccr2_gen/scripts/52_plip_bulk_array.sh` — uses run_plip_bulk.py, reads from chunk dirs directly
- Create: `ccr2_gen/scripts/53_score_final.sh` — reads JSONL, produces three rankings

Pipeline: 50 → (51 parallel) → 52 Vina PLIP + 52 Glide PLIP → 53 scoring
**No merge step at all.**

- [ ] **Step 1: Write PBS scripts**
- [ ] **Step 2: Test end-to-end on 100 molecules**

---

### Task 6: Documentation and memory

- [ ] **Step 1: Update cluster_lich_compute.md memory with bulk I/O best practice**
- [ ] **Step 2: Add docstrings explaining why bulk files instead of per-molecule dirs**
