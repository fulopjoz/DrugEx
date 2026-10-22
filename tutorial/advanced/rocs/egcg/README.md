# EGCG ligand-based shape-matching RL

Reinforcement learning on a **Papyrus pretrained SMILES-RNN**, driven by 3D
shape + colour similarity to a single reference molecule, **EGCG**
(epigallocatechin gallate), plus synthetic accessibility.

- **No supervised fine-tuning.** The agent starts from the unmodified Papyrus
  checkpoint. RL *does* update its weights — that is the mechanism under test.
- **No scaffold constraint.** EGCG is a 3D *scoring reference*, not a required
  substructure. (Its Bemis–Murcko scaffold covers 25 of 33 heavy atoms, so
  scaffold conditioning would pre-determine most of the molecule.)
- **RDKit backend only**, single reference, `ParetoCrowdingDistance`, one seed.

---

## Which notebook is which

There are **two**, and they answer different questions.

### `egcg_rocs_minimal.ipynb` ← start here

**The ROCS tutorial, minimally adapted to EGCG.** 20 cells, same structure as
`../rocs_rl_tutorial.ipynb`, no extra apparatus. This is the one to read or run
if you want to see the method working.

Eight changes from the CCR2 tutorial, all listed in its first cell:

| # | Change | Why |
|---|---|---|
| 1 | EGCG reference SDF instead of `CCR2_reference_ligands.sdf` | the point of the experiment |
| 2 | No `prepare_models.py`; `agent` and `mutate` both from Papyrus | no supervised fine-tuning |
| 3 | `mutate` is a **separate frozen instance**, not the same object as `agent` | with `mutate=agent`, `epsilon` has no effect at all — see below |
| 4 | `optimization_mode='combo'` removed | not a parameter of `RDKitROCSScorer` on this branch |
| 5 | Score column read from `scorer.getKey()` | one reference gives `RDKit_Supermol_…`, not `…_5refs_…` |
| 6 | Threshold 0.73 instead of 0.871 | 0.871 came from a CCR2 active/decoy ROC and does not transfer |
| 7 | Dropped the CCR2 threshold-validation cell | there are no EGCG actives/decoys |
| 8 | Added a before/after sample | `desired_ratio` alone is scored on 10% of the batch and is too noisy to read |

RL settings are **the tutorial's own**: 50 epochs × 500 samples.

Run it: `qsub submit_minimal.sh` (32 cores, ~10 h). Output in `minimal_out/`.

**It reproduces the full experiment.** Job 14060, 9 h 42 m, exit 0, 1,000
molecules per arm:

| Quantity | before | after | Δ | v3 for comparison |
|---|---|---|---|---|
| Median TanimotoCombo | 0.6416 | **0.6681** | **+0.0265** | +0.0258 |
| p90 TanimotoCombo | 0.7516 | 0.7914 | +0.0398 | +0.0476 |
| `desired_ratio` | 0.140 | 0.236 | +0.096 | +0.120 |
| Validity | 0.955 | 0.968 | +0.013 | +0.014 |
| Uniqueness | 0.955 | **0.967** | **+0.012** | −0.021 |

Bootstrap 95% CI on the median difference **[+0.0192, +0.0356]**, excludes
zero. The effect matches v3's +0.0258 to within noise, using a tenth of the
evaluation samples and none of the gating apparatus — so the plain tutorial,
minimally adapted, is enough to demonstrate the method.

### `egcg_rl.ipynb` — the instrumented version

Same experiment with the measurement apparatus that produced the evidence
below: six gates, provenance hashes, calibrated thresholds, 10,000-molecule
arms, per-epoch trajectory tracking, pre-registered bounds. Heavier to read;
this is where the numbers in `RESULTS.md` come from.

Run it: `qsub submit_egcg_rl.sh`. Output in `runs/`.

**Use the minimal notebook to demonstrate the method; use this one to defend a
number.**

---

## Results at a glance

Three production runs. Full detail in [`RESULTS.md`](RESULTS.md).

| | v1 | v2 | **v3 (final)** |
|---|---|---|---|
| Job | 14053 | 14054 | **14055** |
| RL budget | 20 × 256 | 60 × 512 | 60 × 512 |
| Samples/arm | 1,000 | 10,000 | 10,000 |
| Exploration | **inert** | **inert** | **working** |
| Median TanimotoCombo | 0.6401 → 0.6338 ❌ | 0.6429 → 0.7083 | 0.6429 → **0.6687** |
| 95% CI on Δ | [−0.015, +0.003] | [+0.059, +0.073] | **[+0.023, +0.029]** |
| Uniqueness | 0.861 | **0.226** ❌ | **0.933** ✅ |
| Bounds passed | 3/5 | 4/5 | **5/5** |
| Runtime | 2 h 15 m | 10 h 44 m | 19 h 32 m |

**v3 is the final result**: the only configuration that satisfies every
pre-registered criterion — shape similarity improves *and* output diversity is
preserved.

### Which notebook produced which run

`egcg_rl.ipynb` was **edited in place** between runs, so the copy in this
directory is the **v3** version. Nothing is lost — each run's fully executed
notebook is kept with its outputs, and each recorded the commit it ran from:

| Run | Executed notebook | Commit | Recover the source with |
|---|---|---|---|
| v1 | `runs_v1_20epochs/executed_egcg_rl_20261020_182604.ipynb` | `af619251` | `git show af619251:tutorial/advanced/rocs/egcg/egcg_rl.ipynb` |
| v2 | `runs_v2_60epochs/executed_egcg_rl_20261020_214646.ipynb` | `a15f87fe` | `git show a15f87fe:…` |
| v3 | `runs/executed_egcg_rl_20261021_114825.ipynb` | `96b7d4e2` | the file in this directory |

Verified: the committed source at `96b7d4e2` is code-cell-identical to the
executed v3 notebook, so the `dirty=True` flag in `provenance.json` reflects
untracked scratch files, not notebook edits.

### How they differ, and why it matters

**v1 → v2 is a budget change.** v1 used 20 × 256, a 5× reduction of the
tutorial's own 50 × 500 that I made for cost. At that budget the shape effect
was invisible and the run looked like a clean negative. It was not: v2 at the
tutorial's own scale found a clear effect. *A short-run negative was not
trustworthy here.*

**v2 → v3 is a bug fix.** Both v1 and v2 passed `mutate=agent` — the same
object. DrugEx documents `mutate` as "the **pre-trained** network which
increases the exploration of the chemical space", and `SequenceRNN.evolve`
replaces token probabilities with `mutate`'s for an `epsilon` fraction of rows.
With `mutate` aliased to the agent, exploration draws from the very network RL
is collapsing, so **`epsilon = 0.2` did nothing in either run**. Measured on
v2's final checkpoint:

| ε | `mutate=agent` uniqueness | frozen prior uniqueness |
|---|---|---|
| 0.0 | 0.530 | 0.530 |
| 0.1 | 0.530 | 0.768 |
| 0.2 | 0.531 | 0.871 |
| 0.4 | 0.528 | 0.926 |

Epsilon moved nothing at any value. v2's mode collapse was therefore not a
mystery — it was pure on-policy REINFORCE with no exploration at all.

v3 fixes it with a separate frozen prior. The trade-off was stated in advance
and is visible in the table: ~60% of v2's shape gain traded for essentially
all of the diversity.

---

## Files

| Path | What |
|---|---|
| `egcg_rocs_minimal.ipynb` | minimal tutorial adaptation — **read this first** |
| `egcg_rl.ipynb` | instrumented experiment (source) |
| `egcg_rl_v3_executed.html` | **v3 results**, open in a browser |
| `egcg_rl_v2_executed.html` | v2 results |
| `egcg_rl_v1_executed.html` | v1 results |
| `egcg_rocs_minimal_executed.html` | **minimal notebook results** |
| `runs*/executed_*.ipynb` | the executed notebook for each run, with outputs |
| `RESULTS.md` | full write-up, all three runs + CPU benchmark |
| `config_egcg.py` | all parameters, `__file__`-anchored |
| `build_reference.py` | EGCG → one-conformer SDF, with its gate |
| `_figstyle.py` | figure style (modular, no on-canvas titles) |
| `submit_minimal.sh`, `submit_egcg_rl.sh` | PBS scripts |
| `bench/` | CPU scaling benchmark |
| `runs/`, `panels/` | v3 outputs |
| `runs_v2_60epochs/`, `runs_v1_20epochs/` | archived earlier runs |

---

## Running it

```bash
export PYTHONPATH=/home/fulopj/drx-run/DrugEx/.worktrees/egcg
cd tutorial/advanced/rocs/egcg

python build_reference.py      # EGCG reference + gate + hash check
python config_egcg.py          # print the full configuration

qsub submit_minimal.sh         # the minimal demonstration (~10 h)
qsub submit_egcg_rl.sh         # the instrumented run (~20 h)
```

Fast end-to-end code test (meaningless numbers, front node is fine):

```bash
EGCG_SMOKE=1 python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=egcg-drugex \
  --output smoke.ipynb --output-dir smoke_out egcg_rl.ipynb
```

### How many CPUs?

**16–32.** Measured, not assumed — benchmark job 14059 with 190 cores
allocated, scoring 64 molecules:

| `n_jobs` | wall | s/molecule | speedup |
|---|---|---|---|
| 16 | 172.5 s | 2.696 | 1.00× |
| 64 | 170.6 s | 2.666 | 1.01× |
| 190 | 171.7 s | 2.683 | 1.01× |

No gain beyond ~16 workers. Conformer generation is serial across molecules in
the main process and is ~86% of the runtime; only shape/colour alignment uses
the scorer pool. Requesting 190 cores wastes 174 of them.

### Three traps worth knowing

1. **`num_threads` and `n_jobs` do not multiply.** `genConformers` runs to
   completion in the main process *before* the alignment `Pool` exists, so they
   are sequential phases. Setting `num_threads=1` as an "oversubscription
   guard" pinned a 190-core job to one core.
2. **TanimotoCombo's 0–2 range is nominal, not achievable.** Query conformers
   are generated independently of the reference pose, so EGCG scored against
   *itself* ceilings at **1.26**. Never gate on "approaching 2.0".
3. **`SmoothClippedScore` is an asymptotic logistic, not a clipped ramp.** For
   `(lower_x=5, upper_x=3)`, a modified threshold of 0.1 is raw SAscore
   **5.099** — not 4.5. State SA cutoffs in physical units and derive the
   modified value.
