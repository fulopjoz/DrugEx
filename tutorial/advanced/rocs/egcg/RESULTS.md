# EGCG ligand-based shape-matching RL — results

Branch: `egcg` (from `feature/shape-matching`)
Design spec: `docs/superpowers/specs/2026-10-20-egcg-ligand-based-rl-design.md`
(on `manuscript-evidence-rebuild`)

Status: **two production runs complete.**

- **v1** (job 14053, 20 epochs x 256, n=1,000): shape did **not** improve.
- **v2** (job 14054, 60 epochs x 512, n=10,000): shape **did** improve,
  robustly — but **uniqueness collapsed to 0.226**, failing the pre-registered
  diversity bound. Scored as **mode collapse, not success**.

See Measurements (v1) and Measurements — v2 (below it).

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

Job 14053, `lich-compute-emcpu-1`, 64 cores, 2 h 15 m, **exit status 0**, zero
cell errors. `SMOKE_MODE: False` verified in `runs/provenance.json`. Branch
`egcg` @ `af619251`. RDKit 2025.09.3, CPU. Gate 2 self-score **1.2597**
(exactly the predicted ceiling); Gate 4 2.78 s/molecule across 62 workers.
Calibrated ROCS threshold **0.7342** (85th percentile of a separate
300-molecule calibration subset); SA cutoff raw 4.5 → modified 0.2689.

### Headline: shape did not improve. Synthetic accessibility improved a lot.

1,000 raw samples per arm, shape over valid molecules.

| Quantity | Baseline | Reinforced | Δ |
| --- | --- | --- | --- |
| Median raw TanimotoCombo | 0.6401 | 0.6338 | **−0.006** |
| p90 raw TanimotoCombo | 0.7502 | 0.7253 | **−0.025** |
| Max raw TanimotoCombo | 0.9105 | 0.9257 | +0.015 |
| Fraction ≥ threshold | 0.139 | 0.089 | −0.050 |
| `desired_ratio` | 0.131 | 0.086 | −0.045 |
| Median **raw SAscore** | 2.593 | **1.664** | **−0.929** |
| Validity | 0.963 | 0.978 | +0.015 |
| Uniqueness | 0.963 | 0.861 | −0.102 |
| Invalid | 37 | 22 | −15 |
| Cap violations | 19 | 0 | −19 |
| Valid but ROCS = 0 | 36 | 1 | −35 |

Pre-registered bounds:

| Check | Result |
| --- | --- |
| Median raw TanimotoCombo increased | **FAIL** |
| p90 raw TanimotoCombo increased | **FAIL** |
| Median raw SA increase ≤ 0.5 | PASS (it *fell* 0.93) |
| Validity ≥ 0.80 | PASS |
| Uniqueness ≥ 0.50 | PASS |

### Mechanism: the agent shrank the molecules

| Quantity | Baseline | Reinforced | Δ |
| --- | --- | --- | --- |
| Median heavy atoms | 27 | 18 | **−9** |
| Median MW | 377.5 | 254.3 | **−123** |
| Median rings | 3 | 2 | −1 |

EGCG has 33 heavy atoms and MW 458. The agent moved *away* from that size,
toward small, simple, trivially synthesizable molecules. That is a coherent
optimum, not a malfunction:

- SA is **easy and unbounded in the rewarded direction**. `SmoothClippedScore`
  keeps paying for lower SAscore indefinitely, and nothing in the environment
  constrains size, so "make it smaller" is a globally winning move.
- Shape similarity to a 33-heavy-atom reference is **hard**, and small
  molecules cannot achieve it.
- `ParetoCrowdingDistance` ranks by non-dominated fronts, so a molecule with
  much better SA and marginally worse ROCS is non-dominated and well rewarded.
  The population drifts along the cheap axis.

The collapse in cap violations (19 → 0) and valid-but-zero scores (36 → 1) is
the same effect seen from another angle: smaller molecules stopped tripping the
heavy-atom and rotatable-bond caps.

Uniqueness fell 0.963 → 0.861 — above the 0.50 bound, so **not** mode collapse
by the pre-registered definition, but a real narrowing consistent with
convergence onto a smaller chemotype.

### What did improve

The *best* molecules got slightly better (max 0.9105 → 0.9257, i.e. 73% of the
1.2597 ceiling), so the agent can find EGCG-like shapes; it just is not
rewarded enough for doing so. Top reinforced molecules by raw TanimotoCombo:

| SMILES | TanimotoCombo | raw SA |
| --- | --- | --- |
| `COc1ccc(Cc2ccccc2OC2CC(O)C(O)C(O)C2)cc1` | 0.9257 | 2.97 |
| `COc1ccccc1N1C(=O)c2ccccc2NC1Sc1ccccc1` | 0.9199 | 2.77 |
| `Cc1ccccc1OC(c1ccccc1)C1(Br)C(=O)Sc2ccccc21` | 0.8733 | 3.36 |

The first is recognisably EGCG-adjacent: a polyhydroxylated ring linked through
an ether to an aromatic system.

### Training log

`desired_ratio` per epoch oscillates between 0.0 and 0.24 with no trend. It is
computed on `n_samples/10` = **26 molecules**, so it moves in steps of ~0.04 —
two molecules. It is noise at this sample size and should not be read as a
learning curve. `avg_amean` drifts up weakly (0.775 → ~0.80), consistent with
the SA gain rather than shape.

Best epoch by `desired_ratio` was 11, and the monitor's checkpoint reflects
that. **It was not evaluated** — the final-epoch weights were, as designed.

### Verdict

Under this configuration, unconstrained RL with ROCS + SA against a single EGCG
reference **does not improve shape similarity**; it optimizes synthetic
accessibility by reducing molecular size. The pipeline is sound — every gate
passed and the apparatus behaved exactly as specified — so this is a result
about the *objective design*, not a bug.

Single seed, descriptive. No significance test, no causal claim beyond this
run.

### What this implies for a next experiment

Not performed; listed so the finding is actionable rather than merely negative.

1. **Constrain or cap the SA reward.** A one-sided unbounded objective against
   a hard one will win every time. Clipping SA desirability at a "good enough"
   raw value (e.g. flat below 3.0) would stop paying for further shrinkage.
2. **Add a size or MW term**, or score shape in a size-normalised way, so that
   drifting away from the reference's volume is not free.
3. **Reconsider the reward scheme for this pair.** `WeightedSum` with explicit
   weights would let shape be weighted above SA; Pareto ranking deliberately
   refuses to make that trade and is therefore the wrong tool when one
   objective is far cheaper than the other.
4. **Replicate across seeds** before trusting even this negative result.

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


---

## Measurements — v2 (raised RL budget)

Job 14054, `lich-compute-emcpu-1`, 64 cores, **10 h 44 m** (vs ~43 h costed),
195 h CPU, **exit 0**, zero cell errors. `SMOKE_MODE: False`, commit
`a15f87fe`. 60 epochs x 512; calibration 1,000; baseline and evaluation 10,000
each. Gate 2 self-score 1.2597; Gate 4 2.78 s/molecule. Calibrated threshold
0.7289. `reload_interval`/`patience`/`min_epochs` = 61/61/60, so the threshold
still could not select weights.

The run finished in a quarter of the costed time because the agent again
shrank the molecules, and small molecules are far cheaper to embed. The cost
model's dependence on that behaviour was stated in advance.

### Shape improved — and the author's prediction was wrong

Before this run I predicted that raising the RL budget on an unbalanced
objective would amplify the size drift and *not* improve shape. **That was
wrong.** The user's hypothesis — that more exploration would find better
matches — is the one the data supports.

| Quantity | Baseline | Reinforced | Δ |
| --- | --- | --- | --- |
| Median raw TanimotoCombo | 0.6429 | **0.7083** | **+0.0654** |
| p90 raw TanimotoCombo | 0.7497 | **0.8487** | **+0.0990** |
| `desired_ratio` | 0.134 | **0.451** | +0.317 |
| Median raw SAscore | 2.639 | 1.425 | −1.214 |
| Validity | 0.955 | 0.990 | +0.035 |
| **Uniqueness** | 0.955 | **0.226** | **−0.728** |
| Invalid | 447 | 98 | −349 |
| Cap violations | 170 | 0 | −170 |
| Valid but ROCS = 0 | 282 | 1 | −281 |

Bootstrap 95% CI on the median difference: **[+0.0589, +0.0725]**, does not
cross zero. At n=10,000 this is well resolved, unlike v1's
[−0.0151, +0.0028].

**The gain survives deduplication**, so it is not an artifact of repeating a
few good molecules: on unique molecules only, baseline median 0.6429
(n=9,546) vs reinforced 0.6976 (n=2,264), **Δ +0.0546**.

The reinforced median sits at **56.2%** of the measured 1.2597 ceiling, up from
51.0%.

### But diversity failed the pre-registered bound

| Check | Result |
| --- | --- |
| Median raw TanimotoCombo increased | PASS |
| p90 raw TanimotoCombo increased | PASS |
| Median raw SA increase ≤ 0.5 | PASS |
| Validity ≥ 0.80 | PASS |
| **Uniqueness ≥ 0.50** | **FAIL (0.226)** |

The rule fixed before any run was explicit: *a median shape increase with
uniqueness below 0.50 is mode collapse, not success.* **That rule applies
here.** Four of five bounds pass and the headline number is the best this
project has produced, which is exactly the situation the bound was written to
survive.

**It is a soft collapse, not a single-molecule one.** 9,902 valid samples
contain 2,264 distinct molecules — ~4.4 copies each on average:

| | share of valid output |
| --- | --- |
| Top 1 molecule | 1.0% |
| Top 5 | 4.7% |
| Top 10 | 8.3% |
| Top 50 | 28.4% |

Most-repeated molecules are small aromatic amides and biphenyl ethers, e.g.
`COc1ccc(-c2ccccc2OCc2ccccc2F)cc1` (combo 0.826, 100 copies) and
`O=C(Nc1ccccc1C(=O)Nc1ccccc1)c1ccccc1` (0.776, 100 copies). Several distinct
molecules appear exactly 100 or 83 times, which is suspicious enough to flag
as a possible sampling artifact rather than pure model behaviour — worth
checking before any publication claim.

### Size still fell, but shape rose anyway

| | Baseline | Reinforced |
| --- | --- | --- |
| Median heavy atoms | 27 | 20 |
| Median MW | 384.5 | 266.1 |

So v1's shrinkage was not the whole story. With three times the budget the
agent found *small* molecules that nonetheless match EGCG's shape far better —
combo 0.78–0.83 at ~20 heavy atoms. Shrinking and shape-matching turned out
not to be mutually exclusive, which is precisely what v1's shorter run could
not reveal.

### Trajectory

From `epoch_trajectory.csv` (per-epoch validation sample, ~50 molecules):
median heavy atoms drops 23 → ~18–20 within the first dozen epochs and then
oscillates without recovering. Median TanimotoCombo oscillates 0.63–0.73 with a
weak upward drift late (0.726 at epoch 43, 0.729 at 55, 0.734 at 60). The size
drift does **not** reverse; the shape gain accrues on top of it.

### Verdict

Raising the RL budget **does** improve shape similarity against a single EGCG
reference, robustly and resolvably — but at this budget it also costs
three-quarters of the output diversity, which the pre-registered rule scores as
mode collapse.

Single seed, descriptive. No causal claim beyond this run.

### What follows

1. **Diversity is now the binding constraint, not shape.** The obvious next
   lever is `epsilon` (exploration) and/or a diversity term; `ParetoCrowdingDistance`
   already carries crowding distance, so the knob to examine first is epsilon,
   not the reward scheme.
2. **Check the exact-100/83 repeat counts** before trusting the uniqueness
   figure — it may be a sampling artifact.
3. Replicate across seeds. A single run cannot separate "more budget improves
   shape" from "this seed found a good basin".
