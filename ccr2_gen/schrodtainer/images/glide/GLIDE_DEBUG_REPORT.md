# Glide Docking Pipeline Debug Report

**Date:** 2026-03-26
**Scope:** `ccr2_gen/schrodtainer/images/glide/src/glide.py` and `run_glide.py`
**Target:** CCR2 allosteric pocket (PDB 5T1A) Glide SP docking via Apptainer

---

## Summary

Seven bugs were identified in the Glide docking pipeline. Two are critical (causing incorrect docking results), two are high/medium severity (causing intermittent failures and wrong precision), and three are low/medium severity. All fixes have been implemented.

Additionally, MMFF minimization was removed from conformer generation per RDKit best practices (see Bug #7 below).

---

## Bug #1 (CRITICAL): Molecules Written Without 3D Coordinates

**File:** `glide.py:30-41` (original)
**Root cause:** `Chem.AddHs()` returns a NEW RDKit molecule — it does not modify in place. The loop variable reassignment `mol = Chem.AddHs(mol)` does not update the `mols` list. The SDF writer then iterates over the original list, writing molecules with **no 3D coordinates**.

```python
# BEFORE (broken):
mols = [Chem.MolFromSmiles(s) for s in smiles]
for mol in mols:
    mol = Chem.AddHs(mol)        # NEW object, list unchanged
    EmbedMolecule(mol, randomSeed=42)
for mol, mol_id in zip(mols, ids):  # Original flat molecules
    sdf_writer.write(mol)            # Written without 3D coords
```

**Impact:** All ligands passed to LigPrep as flat 2D structures. LigPrep may partially recover by re-generating 3D, but this produces inferior starting geometries and may fail for complex molecules.

**Fix:** Build a new `mols` list explicitly, using ETKDGv3 + MMFF optimization (matching the Vina pipeline quality). Added fallback to `useRandomCoords=True` for difficult molecules.

---

## Bug #2 (CRITICAL): Thread-Unsafe `os.chdir()` in Parallel Execution

**File:** `glide.py:77-86, 149-152, 241` (original)
**Root cause:** Both `ligprep()` and `glide_docking()` use `os.chdir()` to switch working directories. The `run()` function uses `ThreadPoolExecutor` for parallelization. Since `os.chdir()` is **process-global**, multiple threads race on the current working directory, causing:
- Glide/LigPrep unable to find input files
- Output written to wrong directories
- Intermittent, non-reproducible failures

**Impact:** With `n_workers > 1`, docking jobs fail randomly. The error manifests differently each time, making it extremely hard to debug.

**Fix:**
1. Replaced all `os.chdir()` calls with `subprocess.run(cwd=work_dir)` and `tempfile.mkdtemp()` for isolated working directories
2. Replaced `ThreadPoolExecutor` with `ProcessPoolExecutor` — each process gets its own CWD state

---

## Bug #3 (HIGH): Precision Hardcoded to HTVS, Patched via Fragile `sed`

**File:** `glide.py:147` (original), `29_glide_sp_production.sh:101`
**Root cause:** The Glide precision was hardcoded to `HTVS` in `glide.py`. The PBS script used `sed -i 's/PRECISION   HTVS/PRECISION   SP/'` to patch it. If whitespace doesn't match exactly (tabs vs spaces), `sed` fails **silently** and docking runs at HTVS instead of SP.

**Impact:** Potential for entire production runs at wrong precision with no warning.

**Fix:** Added `precision` parameter to `glide_docking()`, `run_workflow()`, `run()`, and `--precision` CLI flag to `run_glide.py`. The PBS script now passes `--precision "$PRECISION"` instead of sed-patching. The sed line has been removed.

---

## Bug #4 (MEDIUM): `ligprep()` Wait Loop Was Dead Code

**File:** `glide.py:63,88-91` (original)
**Root cause:** `tempfile.NamedTemporaryFile(delete=False)` creates the file immediately. The wait loop `while not os.path.exists(prepared_ligand.name)` exits instantly because the (empty) file already exists. This means if LigPrep produced output at a different path, the pipeline would silently use an empty file.

**Fix:** Replaced with explicit `FileNotFoundError` check after `subprocess.run()` completes. LigPrep output path is now deterministic (`work_dir/prepared.mae`).

---

## Bug #5 (MEDIUM): Missing Enhanced Sampling Parameters

**File:** `glide.py:142-148` (original)
**Root cause:** The Glide input was missing `EXPANDED_SAMPLING True` and `NENHANCED_SAMPLING 2`, which are present in the reference enamine implementation. Without these, Glide uses minimal sampling, producing fewer and lower-quality poses.

**Fix:** Added `EXPANDED_SAMPLING True` and `NENHANCED_SAMPLING 2` to the Glide input template.

---

## Bug #6 (LOW): LigPrep `EPIK no` Contradicted Production Intent

**File:** `glide.py:68` (original)
**Root cause:** `EPIK no` disabled Epik protonation state enumeration, while the DOCKING_VALIDATION_REPORT.md documented "Epik ionization at pH 7.0 +/- 2.0". `EPIKX yes` alone provides extended tautomers but not full pKa-based ionization.

**Fix:** Changed to `EPIK yes` to enable proper ionization state enumeration matching the documented protocol.

---

## Bug #7 (MEDIUM): MMFF Minimization After ETKDGv3 Is Counterproductive

**File:** `glide.py:46` (previous fix)
**Root cause:** The previous fix applied `AllChem.MMFFOptimizeMolecule()` after ETKDGv3 embedding. Per the RDKit documentation and Greg Landrum's recommendations, MMFF minimization is **not needed** and is **counterproductive** for docking:

- RDKit docs: *"With [ETKDG] there should be no need to use a minimisation step to clean up the structures."*
- ETKDGv3 produces CSD experimental torsion-quality geometries (ideal for protein-ligand docking)
- MMFF minimizes toward gas-phase energy minima, distorting the CSD-biased torsion angles
- Greg Landrum's 2022-09-29 blog post optimizes ETKDGv3's **internal** DG force field parameters, not external MMFF
- GitHub Discussion #8226: ETKDGv3 is recommended for docking (ET terms = crystal structure torsions)

**Fix:** Removed `AllChem.MMFFOptimizeMolecule()` call entirely. Added tiered embedding fallback:
1. ETKDGv3 (default)
2. ETKDGv3 + `useRandomCoords=True` + `maxIterations=5000`
3. srETKDGv3 (small-ring torsions) + `useRandomCoords=True`
4. Skip molecule with warning if all tiers fail

**References:**
- https://greglandrum.github.io/rdkit-blog/posts/2022-09-29-optimizing-conformer-generation-parameters.html
- https://greglandrum.github.io/rdkit-blog/posts/2025-08-30-confgen-scaling.html
- https://github.com/rdkit/rdkit/discussions/8226
- https://greglandrum.github.io/rdkit-blog/posts/2024-07-28-confgen-and-intramolecular-hbonds.html
- RDKit Getting Started: https://www.rdkit.org/docs/GettingStartedInPython.html

---

## Files Modified

| File | Changes |
|------|---------|
| `ccr2_gen/schrodtainer/images/glide/src/glide.py` | Fixed all 6 bugs: 3D embedding, thread safety, precision parameter, wait loop, sampling, Epik |
| `ccr2_gen/schrodtainer/images/glide/src/run_glide.py` | Added `--precision` CLI argument |
| `ccr2_gen/scripts/29_glide_sp_production.sh` | Removed sed patching, passes `--precision` flag |

---

## Impact on Existing Results

The existing 52,555 Glide SP production results (`docking/results/glide_sp/`) were generated with the **broken** pipeline. However:

1. **Bug #1 (no 3D coords):** LigPrep likely recovered most molecules by re-generating 3D from the 2D input, since LigPrep is designed to handle 2D SDF. The scores may be slightly different from what properly embedded input would produce, but the results are not completely invalid.

2. **Bug #2 (thread race):** The PBS script used `--workers 10`, so this affected production runs. However, each PBS array element processes one chunk independently via Apptainer, so the race only affects the internal parallelization within a single chunk. Failed chunks would have produced empty or partial output files.

3. **Bug #3 (precision):** The `sed` pattern in the PBS script matched the exact whitespace in the original code, so SP was correctly applied for the production run. This is fragile but happened to work.

**Recommendation:** For publication-quality results, re-run a validation subset (e.g., the 36 consensus molecules + 5 references) with the fixed pipeline and compare scores.

---

## Verification Plan

1. Run the test script with known molecules (cocaine, paracetamol) using the fixed pipeline
2. Compare scores of 5 reference ligands (CCR2RAR, CCX140, COMPOUND39, JNJ27141491, SD24) between old and new pipeline
3. Spot-check 10 random molecules from the production library
4. Verify that `--precision HTVS` and `--precision SP` produce different scores (confirming the parameter is respected)
