"""Configuration for the EGCG ligand-based shape-matching RL pilot.

Design spec: ``docs/superpowers/specs/2026-10-20-egcg-ligand-based-rl-design.md``
(on the manuscript-evidence-rebuild branch).

Scope: RDKit backend only, single EGCG reference, no supervised fine-tuning,
no scaffold constraint, ParetoCrowdingDistance, single-seed descriptive pilot.

Every path here is anchored on ``__file__``. The original ``config.py`` uses
``ROOT = Path.cwd()``, which under ``jupyter nbconvert --execute`` makes every
relative path depend on the launch directory -- a silent correctness hazard.
"""

import os
from pathlib import Path

import numpy as np

from drugex.data.corpus.vocabulary import VocSmiles
from drugex.training.environment import DrugExEnvironment
from drugex.training.generators import SequenceRNN
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.modifiers import SmoothClippedScore
from drugex.training.scorers.properties import Property
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer

# ============================================================================
# Paths
# ============================================================================

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[3]                      # .../.worktrees/egcg
REFERENCE_SDF = HERE / "EGCG_reference.sdf"
OUTPUT_DIR = HERE / "runs"
PANEL_DIR = HERE / "panels"

# --- MACHINE-LOCAL ABSOLUTE PATH -------------------------------------------
# The pretrained Papyrus checkpoint lives under the main checkout's ``data/``,
# which is gitignored (.gitignore:91, *.pkg at :99) and therefore absent from
# every worktree. Resolving it relative to this file would point at nothing.
#
# TO RUN THIS ON ANOTHER MACHINE, CHANGE THIS ONE CONSTANT. It is the only
# machine-local path in this experiment. Gate 0 asserts both files exist and
# verifies their SHA-256 against PRETRAINED_SHA256 below, so a moved or
# replaced checkpoint fails loudly rather than silently training from
# different weights.
PRETRAINED_DIR = Path(
    "/home/fulopj/drx-run/DrugEx/data/models/pretrained/smiles-rnn/"
    "Papyrus05.5_smiles_rnn_PT"
)
# ---------------------------------------------------------------------------

PRETRAINED_PKG = PRETRAINED_DIR / "Papyrus05.5_smiles_rnn_PT.pkg"
PRETRAINED_VOCAB = PRETRAINED_DIR / "Papyrus05.5_smiles_rnn_PT.vocab"

# Expected hashes. Populated by ``python config_egcg.py --print-hashes``.
# An empty string means "not yet pinned" -- Gate 0 then RECORDS the hash for
# provenance and says so, rather than pretending to verify it. A non-empty
# value is ASSERTED. Logging a hash is not asserting against one.
PRETRAINED_SHA256 = {
    "pkg": "34dbc312454635e374dee6ce61897fc4889b2598cddaf292fd243f41c2369f7f",
    "vocab": "5da364a8b64662462c5b7a7b29c3cbe3c4aad4aaf982ec1ba339ffa6735154ce",
}
REFERENCE_SHA256 = "39bce6edd56e661c049ac42f2e18e27648d6c6080673ceb292cccb16ec1c02bb"

# Monitor writes best-by-criterion here; we save final-epoch weights separately.
RUN_NAME = "EGCG_rdkit_reinforced"
MONITOR_BASE = OUTPUT_DIR / RUN_NAME                     # FileMonitor prefix
FINAL_CHECKPOINT = OUTPUT_DIR / f"{RUN_NAME}_final_epoch.pkg"

# ============================================================================
# Reference molecule
# ============================================================================

# EGCG (epigallocatechin gallate), PubChem CID 65064, CHEMBL297453.
# Verified with RDKit: C22H18O11, MW 458.38, HBD/HBA 8/11, TPSA 197.4,
# 33 heavy atoms, 3 rotatable bonds, Murcko scaffold 25/33 heavy atoms.
EGCG_SMILES = "O=C(O[C@@H]1Cc2c(O)cc(O)cc2O[C@@H]1c1cc(O)c(O)c(O)c1)c1cc(O)c(O)c(O)c1"
EGCG_HEAVY_ATOMS = 33
REFERENCE_EMBED_SEED = 0xE6C6                           # fixed, documented

# ============================================================================
# Seeds
# ============================================================================

# Conformer embedding is already deterministic inside DrugEx
# (params.randomSeed = 0xc0ffee, conformer_generators.py:475). RL sampling is
# the only stochasticity, and drugex/ contains no seeding call, so the seed is
# set from the notebook (torch + numpy + random) rather than by modifying drugex.
SEED = 42

# ============================================================================
# Conformer generation
# ============================================================================

# CCR2 campaign caps, reused for continuity. Identical in every arm.
MAX_ISOMERS = 4
MAX_HEAVY_ATOMS = 45
MAX_ROTATABLE_BONDS = 15

# Pilot conformer count, chosen EMPIRICALLY rather than guessed. Measured EGCG
# self-score against this reference:
#     max_conformers=10 -> 1.1835
#     max_conformers=30 -> 1.2597
#     max_conformers=60 -> 1.2597   (saturated)
# 30 is the knee; 60 costs twice as much for no gain.
MAX_CONFORMERS = 30

# MEASURED CEILING. The nominal TanimotoCombo range is 0-2 (shape 0-1 + colour
# 0-1), but that is NOT achievable: query conformers are generated
# independently of the reference geometry, so even EGCG scored against itself
# cannot recover its own pose. At MAX_CONFORMERS=30 the EGCG self-score is
# ~1.26, and that is the practical ceiling for ANY molecule here.
#
# Two consequences:
#   * Gate 2 must NOT expect a self-score near 2.0 -- that would fail a healthy
#     pipeline. It compares the self-score against a dissimilar control.
#   * An absolute threshold like 1.0 is ~79% of the achievable maximum, i.e.
#     brutally strict. This is why the ROCS threshold is calibrated from a
#     baseline percentile rather than set to a round number.
#
# Re-measure if the reference, conformer settings or colour typing change.
SELF_SCORE_CEILING = 1.2597

# Gate 2 bound: the self-score must exceed a dissimilar control by this margin.
# Benzene scores ~0.26 against this reference, so the gap is large and a
# failure here means the scoring path is broken, not that EGCG is unusual.
GATE2_MIN_SELF_SCORE = 1.0
GATE2_CONTROL_SMILES = "c1ccccc1"
GATE2_MAX_CONTROL_SCORE = 0.6

# OVERSUBSCRIPTION GUARD (spec 15). num_threads and n_jobs MULTIPLY: the
# tutorial's num_threads=0 (all cores) with n_jobs=-1 (all cores) spawns
# cores^2 workers. Conformer generation is single-threaded here; parallelism
# comes from the scorer's process pool only.
CONFORMER_NUM_THREADS = 1
SCORER_N_JOBS = -1          # notebook overrides from PBS_NCPUS (minus reserve)

# ============================================================================
# RL parameters
# ============================================================================

RL_EPOCHS = 30
RL_EPSILON = 0.2
RL_N_SAMPLES = 256

# fit() decoupling (spec 6). reload_interval > epochs stops best-state resets;
# min_epochs == epochs and patience > epochs stop criterion-based early
# stopping. Together these keep the calibrated threshold out of the training
# loop entirely, so it remains a reporting criterion only.
RL_RELOAD_INTERVAL = RL_EPOCHS + 1
RL_PATIENCE = RL_EPOCHS + 1
RL_MIN_EPOCHS = RL_EPOCHS
RL_CRITERIA = "desired_ratio"        # recorded; inert given the settings above

# ============================================================================
# Sampling sizes
# ============================================================================

# Stages 1a and 1b are DISTINCT subsets (spec 12): calibrating the threshold
# and measuring the baseline on the same molecules would make the baseline
# desired_ratio a tautology of the chosen percentile.
N_CALIBRATION = 300
N_BASELINE = 1000
N_EVALUATION = 1000
GEN_BATCH_SIZE = 64

# ============================================================================
# Smoke-test mode
# ============================================================================
# EGCG_SMOKE=1 shrinks every size so the whole notebook can be executed
# end-to-end on the front node in minutes, to prove the CODE runs. It produces
# numbers that are scientifically meaningless -- far too few samples and
# conformers -- and the notebook says so loudly when it is active.
#
# This exists because "the notebook is syntactically valid" is not evidence
# that it runs; only running it is.
SMOKE = os.environ.get("EGCG_SMOKE", "") not in ("", "0", "false", "False")

if SMOKE:
    MAX_CONFORMERS = 3
    RL_EPOCHS = 2
    RL_N_SAMPLES = 16
    RL_RELOAD_INTERVAL = RL_EPOCHS + 1
    RL_PATIENCE = RL_EPOCHS + 1
    RL_MIN_EPOCHS = RL_EPOCHS
    N_CALIBRATION = 24
    N_BASELINE = 32
    N_EVALUATION = 32
    GEN_BATCH_SIZE = 16
    # The measured ceiling and Gate 2 bounds assume MAX_CONFORMERS=30; with 3
    # conformers the self-score is lower, so relax the gate rather than let a
    # smoke test fail on a correctly working pipeline.
    GATE2_MIN_SELF_SCORE = 0.8

# ============================================================================
# Thresholds -- reporting criteria only, NOT optimization parameters
# ============================================================================
#
# ParetoCrowdingDistance does not use thresholds to compute rewards, and the
# fit() settings above remove their influence on checkpoint selection and early
# stopping. They therefore determine the `Desired` flag and `desired_ratio`,
# and nothing else. See spec 5.

# ROCS: raw TanimotoCombo on its original 0-2 scale. No modifier is set on the
# ROCS scorer, so DrugExEnvironment compares the threshold against the raw
# score directly (environment.py: scores[keys] < self.thresholds). We do NOT
# add a ClippedScore and halve the threshold.
ROCS_CALIBRATION_PERCENTILE = 85.0
ROCS_THRESHOLD_FALLBACK = 1.0        # used only if calibration is skipped

# SA: expressed in PHYSICAL raw-SAscore units first; the modified threshold is
# derived from it numerically below, so the two can never disagree.
RAW_SA_CUTOFF = 4.5                  # raw SAscore; lower = more accessible
SA_MODIFIER_LOWER_X = 5.0
SA_MODIFIER_UPPER_X = 3.0


def sa_modifier() -> SmoothClippedScore:
    """The SA modifier used everywhere in this experiment."""
    return SmoothClippedScore(lower_x=SA_MODIFIER_LOWER_X, upper_x=SA_MODIFIER_UPPER_X)


def raw_sa_to_modified(raw: float) -> float:
    """Apply the SA modifier to a raw SAscore."""
    return float(sa_modifier()(np.array([float(raw)]))[0])


def modified_sa_to_raw(modified: float) -> float:
    """Invert the SA modifier analytically.

    SmoothClippedScore is a LOGISTIC, not a clipped ramp:
        y = low + L / (1 + exp(-k * (x - middle)))
    with k = 4 / (upper_x - lower_x), middle = (upper_x + lower_x) / 2,
    L = high - low. It is asymptotic and never attains 0 or 1.

    Inverting:  x = middle - ln(L / (y - low) - 1) / k
    """
    m = sa_modifier()
    y = float(modified)
    if not (m.low_score < y < m.low_score + m.L):
        raise ValueError(
            f"modified score {y} outside the open interval "
            f"({m.low_score}, {m.low_score + m.L}) -- the logistic is asymptotic "
            "and never reaches its bounds"
        )
    return m.middle_x - np.log(m.L / (y - m.low_score) - 1.0) / m.k


# Derived, never hand-written.
SA_THRESHOLD = raw_sa_to_modified(RAW_SA_CUTOFF)

# Self-check: the analytic inverse must round-trip against the modifier itself.
# If this ever fails, SmoothClippedScore's implementation changed underneath us.
assert abs(modified_sa_to_raw(SA_THRESHOLD) - RAW_SA_CUTOFF) < 1e-9, (
    "SA modifier inverse disagrees with the modifier -- do not trust the threshold"
)

# For the record (computed, not inferred): the CCR2 campaign's SA_THRESHOLD of
# 0.1 corresponds to raw SAscore ~5.0986, NOT 4.5 and NOT 4.8.
CAMPAIGN_SA_THRESHOLD_RAW_EQUIVALENT = modified_sa_to_raw(0.1)

# ============================================================================
# Factories
# ============================================================================


def create_conformer_generator(num_threads: int = CONFORMER_NUM_THREADS):
    """Conformer generator. Identical in calibration, baseline and evaluation."""
    return RDKitConformerGenerator(
        max_conformers=MAX_CONFORMERS,
        max_isomers=MAX_ISOMERS,
        max_heavy_atoms=MAX_HEAVY_ATOMS,
        max_rotatable_bonds=MAX_ROTATABLE_BONDS,
        num_threads=num_threads,
        show_progress=False,
    )


def create_rocs_scorer(n_jobs: int = SCORER_N_JOBS, num_threads: int = CONFORMER_NUM_THREADS):
    """RDKit ROCS scorer against the single EGCG reference.

    Deliberately omits:
      * ``optimization_mode`` -- NOT a parameter of RDKitROCSScorer on this
        branch (the shipped tutorial and config.py both pass it anyway, which
        is why they only run when shadowed by a divergent checkout).
      * ``inject_color`` / ``color_ff`` / ``score_variant`` -- these DO exist
        here, but we keep their defaults, so colour uses rdShapeAlign's
        5-pattern fallback with no hydrophobe type. Same colour behaviour as
        the shipped tutorial, and the threshold is calibrated against exactly
        this scoring.

    No modifier is attached, so the score stays on the raw TanimotoCombo 0-2
    scale.
    """
    if not REFERENCE_SDF.exists():
        raise FileNotFoundError(
            f"EGCG reference not found: {REFERENCE_SDF}\nRun: python build_reference.py"
        )
    return RDKitROCSScorer(
        conformer_generator=create_conformer_generator(num_threads=num_threads),
        references=str(REFERENCE_SDF),
        score_type="TanimotoCombo",
        use_colors=True,
        show_progress=False,
        n_jobs=n_jobs,
    )


def create_environment(rocs_threshold: float, n_jobs: int = SCORER_N_JOBS,
                       num_threads: int = CONFORMER_NUM_THREADS):
    """DrugExEnvironment with RDKit ROCS + SA.

    Parameters
    ----------
    rocs_threshold : float
        RAW TanimotoCombo (0-2). Reporting criterion only.
    """
    rocs_scorer = create_rocs_scorer(n_jobs=n_jobs, num_threads=num_threads)

    sa_scorer = Property("SA")
    sa_scorer.setModifier(sa_modifier())

    return DrugExEnvironment(
        scorers=[rocs_scorer, sa_scorer],
        thresholds=[rocs_threshold, SA_THRESHOLD],
        reward_scheme=ParetoCrowdingDistance(),
    )


def create_raw_sa_scorer():
    """Unmodified SA scorer, for the PHYSICAL SA distribution.

    Scored separately so the raw SA distribution costs nothing extra -- a
    second env.getScores() call would redo the expensive ROCS pass.
    """
    return Property("SA")


def load_agent():
    """SequenceRNN initialised from the Papyrus pretrained checkpoint.

    Agent and prior both come from here: no supervised fine-tuning. RL does
    update the agent's weights -- that is the mechanism under test.
    """
    voc = VocSmiles.fromFile(str(PRETRAINED_VOCAB), encode_frags=False)
    agent = SequenceRNN(voc, is_lstm=True)
    agent.loadStatesFromFile(str(PRETRAINED_PKG))
    return agent, voc


# ============================================================================
# Reporting bounds (fixed before any run -- spec 13)
# ============================================================================

BOUNDS = {
    "max_median_raw_sa_increase": 0.5,
    "min_validity": 0.80,
    "min_uniqueness": 0.50,
}


def summary() -> dict:
    """Everything a reader needs to reproduce the configuration."""
    return {
        "seed": SEED,
        "rl_epochs": RL_EPOCHS,
        "rl_epsilon": RL_EPSILON,
        "rl_n_samples": RL_N_SAMPLES,
        "rl_reload_interval": RL_RELOAD_INTERVAL,
        "rl_patience": RL_PATIENCE,
        "rl_min_epochs": RL_MIN_EPOCHS,
        "rl_criteria": RL_CRITERIA,
        "max_conformers": MAX_CONFORMERS,
        "max_isomers": MAX_ISOMERS,
        "max_heavy_atoms": MAX_HEAVY_ATOMS,
        "max_rotatable_bonds": MAX_ROTATABLE_BONDS,
        "conformer_num_threads": CONFORMER_NUM_THREADS,
        "n_calibration": N_CALIBRATION,
        "n_baseline": N_BASELINE,
        "n_evaluation": N_EVALUATION,
        "rocs_calibration_percentile": ROCS_CALIBRATION_PERCENTILE,
        "raw_sa_cutoff": RAW_SA_CUTOFF,
        "sa_threshold_modified": SA_THRESHOLD,
        "campaign_sa_0.1_raw_equivalent": CAMPAIGN_SA_THRESHOLD_RAW_EQUIVALENT,
        "colour_typing": "scorer defaults (inject_color=False, color_ff='default')",
        "rocs_scale": "raw TanimotoCombo 0-2, no modifier",
        "reward_scheme": "ParetoCrowdingDistance",
        "SMOKE_MODE": SMOKE,
    }


if __name__ == "__main__":
    import hashlib
    import json
    import sys

    if "--print-hashes" in sys.argv:
        for label, path in (("pkg", PRETRAINED_PKG), ("vocab", PRETRAINED_VOCAB),
                            ("reference", REFERENCE_SDF)):
            if path.exists():
                h = hashlib.sha256(path.read_bytes()).hexdigest()
                print(f"{label:10s} {h}  {path}")
            else:
                print(f"{label:10s} MISSING                {path}")
    else:
        print(json.dumps(summary(), indent=2, default=str))
