# CCR2 Swarm Audit And Fix Plan

## Scope

This audit validates the pipeline findings reported for the five-stage CCR2 screening workflow and maps each confirmed issue to a concrete fix and verification step.

## Validated Findings

| ID | Severity | File | Validation | Fix | Verification |
| --- | --- | --- | --- | --- | --- |
| C1 | Critical | `ccr2_gen/scripts/40_vina_dock_all.sh` | Full-pipeline chunk files are generated as `chunk_%03d.tsv`, while the array script looked for `chunk_%04d.tsv`. Jobs `1..999` would miss their input files. | Align chunk lookup and output folder naming to `%03d`. | Run array-script naming check against generated chunk manifest or sample files. |
| C2 | Critical | `ccr2_gen/docking/plip_analysis/redock_top10.py` | SDF conversion failure only printed a warning, then complex creation referenced `rdkit_mol` and the function still reported `status=success`. | Make SDF export and complex creation hard requirements for success and return `failed` on artifact creation errors. | Compile and run targeted regression around docking result handling. |
| C3 | Critical | `ccr2_gen/docking/plip_analysis/run_plip_bulk.py` | `Chem.SDMolSupplier(..., removeHs=True)` strips ligand hydrogens before PLIP profiling. This biases ligand-donor H-bond recovery. | Preserve hydrogens when loading docked poses. | Compile script and inspect PLIP input path behavior. |
| I1 | Important | `ccr2_gen/docking/plip_analysis/score_three_tracks.py` | Molecules with `dock_score=None` were excluded from composite assignment when at least one scored molecule existed. | Assign explicit fallback `dock_norm=0.0` and `composite=0.5*ifp_norm`. | Regression test for mixed scored and unscored candidates. |
| I2 | Important | `ccr2_gen/docking/plip_analysis/score_multicriteria.py` | `normalize_column` filled NaNs with `0.0`, treating missing SA and missing ROCS as worst-case values. | Use neutral imputation at normalized value `0.5`. | Regression test for NaN normalization. |
| I3 | Important | `ccr2_gen/docking/plip_analysis/redock_top10.py` | Pose export relied on `Chem.Mol(..., confId=pose_idx)` instead of explicit conformer extraction, which is fragile and can collapse to the wrong conformer. | Extract each conformer explicitly by conformer ID before writing SDF poses and best-pose complex. | Compile script and inspect SDF export loop. |
| I4 | Important | `ccr2_gen/postprocess/standardize_generated.py` | The documented pipeline says standardized outputs are deduplicated, but `standardized_all.tsv` kept duplicates. | Deduplicate the standardized dataframe on `std_SMILES` before writing outputs. | Regression test for duplicate collapse. |
| I5 | Important | `ccr2_gen/scripts/42_plip_vina_array.sh` | `cp *.json` is not robust once profile count crosses shell expansion limits; prior batch loss is consistent with that failure mode. | Replace the glob copy with `rsync` include/exclude rules. | Shell syntax check and manual inspection. |
| I6 | Important | `ccr2_gen/docking/plip_analysis/score_multicriteria.py` | `IFP_MAX` was hardcoded locally instead of imported from the PLIP source of truth. | Import `IFP_MAX` from `run_plip.py`. | Regression test for constant parity. |
| I7 | Important | `ccr2_gen/docking/plip_analysis/run_plip_bulk.py` | `tmp_path` was referenced in the exception path before guaranteed assignment. | Initialize `tmp_path=None` before the temp-file block and guard unlink. | Compile script. |
| I8 | Important | `ccr2_gen/postprocess/standardize_generated.py` | Parent-molecule cleanup failure was silently ignored, allowing salted or un-neutralized structures to leak into standardized outputs. | Add a safe parent-cleanup fallback using RDKit fragment parent and uncharging; drop molecules if parent cleanup still fails. | Regression test via standardized dataframe behavior and compile. |

## Implementation Plan

1. Fix blockers in the batch-entry scripts and artifact-producing code paths.
2. Fix score normalization and missing-data handling so rankings remain defined and unbiased.
3. Fix postprocessing so standardized outputs match the documented contract.
4. Add a focused regression suite for pure-Python logic that can run quickly without docking backends.
5. Run targeted validation: `unittest`, syntax compilation, and git diff review.

## Test Plan

1. `python -m unittest discover -s ccr2_gen/docking/test -p 'test_pipeline_regressions.py'`
2. `python -m py_compile` on all modified Python scripts.
3. `bash -n` on modified shell scripts.
4. Review git diff to ensure only audited files changed.

## Expected Impact

1. Full-pipeline Vina batch jobs resolve the correct chunk inputs.
2. Redocking no longer produces silent false positives when downstream artifacts are missing.
3. PLIP bulk mode preserves ligand-donor hydrogen bonds.
4. Three-track and multicriteria ranking remain defined for incomplete metadata without worst-case bias.
5. Standardized screening outputs become deduplicated and salt-cleaned by construction.