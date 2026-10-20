# EGCG ligand-based shape-matching RL — results

Branch: `egcg` (from `feature/shape-matching`)
Design spec: `docs/superpowers/specs/2026-10-20-egcg-ligand-based-rl-design.md`
(on `manuscript-evidence-rebuild`)

Status: **pipeline built and smoke-tested; the production run has not been
submitted.** No scientific result exists yet. Everything below the
"Measurements" heading is apparatus, not findings.

---

## What this experiment is

RL on a Papyrus pretrained SMILES-RNN, driven by 3D shape+colour similarity to
a single EGCG conformer plus synthetic accessibility. No supervised
fine-tuning; no scaffold constraint; RDKit backend only;
`ParetoCrowdingDistance`; one seed; descriptive.

"No transfer learning" means no *supervised* fine-tuning stage. RL **does**
update the agent's weights — that is the mechanism under test.

## Apparatus findings (measured while building)

These are properties of the pipeline, established before any experimental run.
They are recorded because several of them invalidate plausible-sounding
assumptions.

### The TanimotoCombo range is nominal, not achievable

The score is nominally 0–2 (shape 0–1 + colour 0–1). Measured EGCG self-score
against its own reference SDF:

| `max_conformers` | EGCG self-score |
| --- | --- |
| 10 | 1.1835 |
| 30 | 1.2597 |
| 60 | 1.2597 (saturated) |

Query conformers are generated independently of the reference geometry, so even
the reference molecule cannot recover its own pose. **~1.26 is the practical
ceiling for any molecule here.** Benzene scores ~0.26 as a dissimilar control.

Consequences, all applied:
- `MAX_CONFORMERS = 30` — chosen as the empirical knee, not guessed. 60 costs
  double for no gain.
- Gate 2 does **not** expect a self-score near 2.0; it asserts self ≥ 1.0 and
  control ≤ 0.6. The earlier draft's "approaching 2.0" expectation would have
  failed a healthy pipeline.
- Scores are additionally reported as a fraction of the measured ceiling
  (`config_egcg.SELF_SCORE_CEILING`), because "1.1 out of a nominal 2.0" badly
  understates what 1.1 means.
- The ROCS threshold is calibrated from a baseline percentile, never a round
  absolute number: a plausible-looking cutoff of 1.0 is ~79% of the maximum.

### The SA threshold is a logistic, and the usual gloss is wrong

`SmoothClippedScore` is `low + L / (1 + exp(-k(x - middle)))` with
`k = 4/(upper_x - lower_x)` — asymptotic, never attaining 0 or 1. For
`SmoothClippedScore(lower_x=5, upper_x=3)`:

| Raw SAscore | Modified |
| --- | --- |
| 3.0 | 0.881 |
| 4.0 | 0.500 |
| 4.5 | 0.269 |
| 5.0 | 0.119 |
| **5.0986** | **0.100** |

So the CCR2 campaign's `SA_THRESHOLD = 0.1` corresponds to raw SAscore
**≈ 5.099** — not 4.5, and not 4.8. This experiment states its SA cutoff in
**physical raw units** (`RAW_SA_CUTOFF = 4.5`) and derives the modified
threshold analytically, with a round-trip assertion, so the two cannot
disagree.

### Thresholds would have leaked into training

`SequenceExplorer.fit` defaults to `criteria='desired_ratio'`,
`reload_interval=50`, `patience=50`, `min_epochs=100`. Under
`ParetoCrowdingDistance` the thresholds do not enter the *reward* — but they
would still have selected which weights survived, via best-state tracking and
periodic reload. `reload_interval > epochs` alone is insufficient.

All of the following are set, and the final-epoch weights are saved separately
with `torch.save(agent.getModel(), ...)` and evaluated. The monitor's
best-by-criterion `.pkg` is deliberately **not** the evaluation artifact:

| Default | Our setting |
| --- | --- |
| `reload_interval=50` | `RL_EPOCHS + 1` |
| `patience=50` | `RL_EPOCHS + 1` |
| `min_epochs=100` | `= RL_EPOCHS` |

With this in place, thresholds are reporting/desirability criteria only.

### `generate()` does not return `Valid` or `Desired`

`generate(evaluator=env)` concatenates only `env.getScorerKeys()`. The shipped
tutorial's `df_gen[["Valid","Desired"]]` cell would raise `KeyError`. Its
`raw_scores=True` default also routes to `getUnmodifiedScores`, which applies no
modifiers and does not zero invalids.

This notebook generates with `evaluator=None` and calls `env.getScores()`
itself. With `drop_invalid=False`, `filterNewMolecules` is skipped entirely, so
raw samples pass through unfiltered **and uncanonicalized**
(`drop_duplicates` is inert on that path) — uniqueness is therefore computed by
canonicalizing explicitly, over all raw samples.

### Not every zero is a conformer failure

Invalid molecules are forced to 0.0 by `getScores`; cap-rejected molecules score
0.0; and a genuinely dissimilar valid molecule can score near 0. Cap violations
*are* measurable from the structure (heavy atoms, rotatable bonds) and are
counted separately. The remainder is reported as "valid but ROCS == 0" and
explicitly **not** attributed to either cause.

### Conformer threads and scorer workers do NOT multiply

Job 14052 was submitted with 190 cores and ran at **load average 1.00** for
nine minutes before being killed — one core busy, 189 idle. PBS's own
accounting hid it (`cput = 4s` while the kernel process showed 9 min of CPU);
only `ps` and `uptime` on the execution node revealed the truth.

The cause was a `CONFORMER_NUM_THREADS = 1` "oversubscription guard" resting on
a false premise. In `RDKitROCSScorer.getScores` the stages are **sequential,
not concurrent**: `genConformers` runs to completion in the main process and
writes an SDF, and only then does the alignment `Pool` spin up. They never
overlap, so they cannot oversubscribe each other. The guard simply pinned the
most expensive stage to a single core.

Measured on 20 molecules at `MAX_CONFORMERS=30`:

| `num_threads` | s/molecule |
| --- | --- |
| 1 | 9.33 |
| 0 (all cores) | 4.58 |

A 2× win, not 30×. `EmbedMultipleConfs` threads the embedding, but the
per-conformer MMFF loop is serial and molecules are iterated serially
(`for i, smi in enumerate(smiles_list)`). **Conformer generation has a
~4.6 s/molecule floor that no number of cores can lower**; only alignment
scales with `n_jobs`.

Consequences: `CONFORMER_NUM_THREADS = 0`; the CPU request dropped from 190 to
**64**, since asking for 190 cores for a workload that can use a few dozen
blocks other users for nothing; and `RL_EPOCHS` 30 → 20 (~7,900 molecules,
≈11 h at the measured rate). After the fix the same job ran at load average
~15 with 24 min of CPU in 76 s of walltime.

### A failed scorer pool would have been invisible

The alignment `Pool` sits in a `try/except` that falls back to
`_score_sequential`, and the warning prints **only when `show_progress=True`**
— which is off, for readability. A failed pool would therefore downgrade an
entire run to one core and still exit 0.

**Gate 4** now measures per-molecule throughput against the conformer floor and
raises above 3× it, rather than trusting the configuration. Note that in
`EGCG_SMOKE=1` mode the gate passes trivially (3 conformers, far below the
30-conformer floor); it is meaningful only at production settings.

### The interpreter, not just the module path, needed a gate

The env's `drugex` is installed **editable against the main checkout**, which
sits on `fix/rocs-optimization-modes` — a line that never ran in production.
`PYTHONPATH` does win over the editable finder (verified), and Gate 0 asserts
the resolved path of `drugex`, `rocs_rdkit` and `config_egcg` regardless.

Separately, the env's own `python3` kernelspec uses a **bare `"python"`** in
`argv`, so `nbconvert` silently launched the base interpreter and died on
`import torch` with no hint the kernel was wrong. Fixed by registering an
`egcg-drugex` kernel with an absolute interpreter path, pinning it in the
notebook metadata and the PBS script, and having Gate 0 assert
`sys.executable`.

## Gates

| Gate | Checks | Status |
| --- | --- | --- |
| 0 | interpreter, module paths, revision, asserted hashes, absence of `optimization_mode` | passes |
| 1 | reference is one parsable 3D non-degenerate molecule, 33 heavy atoms | passes |
| 2 | EGCG self-score ≥ 1.0, dissimilar control ≤ 0.6 | passes |
| 3 | calibration subset not degenerate (validity, non-zero max) | passes |
| 4 | scoring throughput vs the conformer floor (catches silent serial fallback) | passes |

## Reproducing

```bash
export PYTHONPATH=/home/fulopj/drx-run/DrugEx/.worktrees/egcg
cd tutorial/advanced/rocs/egcg

python build_reference.py          # build + Gate 1 + hash assertion
python config_egcg.py              # print the full configuration

# fast end-to-end code test (meaningless numbers, front node is fine)
EGCG_SMOKE=1 python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=egcg-drugex \
  --output smoke.ipynb --output-dir smoke_out egcg_rl.ipynb

# production run
qsub submit_egcg_rl.sh
```

The production run must go through PBS. The front node is for testing code
only.

---

## Measurements

*(Empty. To be filled from `runs/provenance.json` and `runs/comparison.csv`
after the production run. Report changes relative to baseline, including any
that fail the pre-registered bounds, and including anything that failed
outright.)*

Pre-registered bounds, fixed before any run:

| Quantity | Bound |
| --- | --- |
| Median raw TanimotoCombo | increase vs baseline |
| p90 raw TanimotoCombo | increase vs baseline |
| Median raw SAscore | increase ≤ 0.5 |
| Validity | ≥ 0.80 |
| Uniqueness | ≥ 0.50 |

A median shape increase with uniqueness below 0.50 is **mode collapse, not
success**.

Single seed, descriptive: no significance test is applied and no causal claim
is made beyond this run. Multi-seed replication is the prerequisite to any
stronger claim.
