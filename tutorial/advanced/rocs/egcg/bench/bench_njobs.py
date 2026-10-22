"""Measure how ROCS scoring throughput scales with scorer workers.

Answers: does asking for 190 cores beat 64? Conformer generation is serial
across molecules in the main process; only alignment uses the pool, so the
speedup is bounded by Amdahl's law on the alignment fraction.

Same molecules at every worker count, so the comparison is clean.
"""
import sys, time
sys.path.insert(0, "/home/fulopj/drx-run/DrugEx/.worktrees/egcg/tutorial/advanced/rocs/egcg")
import pandas as pd
import config_egcg as cfg

N_MOLS = int(sys.argv[1]) if len(sys.argv) > 1 else 64
WORKERS = [16, 64, 190]

# A fixed, real molecule set: baseline samples from the v2 run.
src = cfg.HERE / "runs_v2_60epochs" / "baseline_raw.csv.gz"
d = pd.read_csv(src)
smiles = d[d.Valid == 1]["canonical"].dropna().unique().tolist()[:N_MOLS]
print(f"{len(smiles)} unique molecules, MAX_CONFORMERS={cfg.MAX_CONFORMERS}, "
      f"conformer threads={cfg.CONFORMER_NUM_THREADS}\n")
print(f"{'n_jobs':>7} {'wall_s':>9} {'s/mol':>8} {'speedup':>8} {'efficiency':>11}")

base = None
for w in WORKERS:
    env = cfg.create_environment(rocs_threshold=0.73, n_jobs=w)
    t0 = time.time()
    env.getScores(smiles)
    dt = time.time() - t0
    if base is None:
        base = dt
    print(f"{w:>7} {dt:>9.1f} {dt/len(smiles):>8.3f} {base/dt:>8.2f}x "
          f"{base/dt/(w/WORKERS[0]):>10.1%}")
