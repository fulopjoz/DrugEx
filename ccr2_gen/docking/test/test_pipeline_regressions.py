from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_module(relative_path: str, module_name: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


run_plip = load_module(
    "ccr2_gen/docking/plip_analysis/run_plip.py",
    "run_plip_for_regression_tests",
)
score_three_tracks = load_module(
    "ccr2_gen/docking/plip_analysis/score_three_tracks.py",
    "score_three_tracks_for_regression_tests",
)
score_multicriteria = load_module(
    "ccr2_gen/docking/plip_analysis/score_multicriteria.py",
    "score_multicriteria_for_regression_tests",
)
standardize_generated = load_module(
    "ccr2_gen/postprocess/standardize_generated.py",
    "standardize_generated_for_regression_tests",
)


class PipelineRegressionTests(unittest.TestCase):
    def test_score_track_assigns_composite_when_dock_score_missing(self):
        interactions = [{"type": "hbond", "resname": "GLU", "resnr": 310}]
        profiles = {
            "with_score": {"interactions": interactions},
            "missing_score": {"interactions": interactions},
        }

        candidates = score_three_tracks.score_track(
            profiles=profiles,
            dock_scores={"with_score": -9.5},
            track_name="vina",
        )
        by_id = {candidate["mol_id"]: candidate for candidate in candidates}

        self.assertIn("composite", by_id["missing_score"])
        self.assertEqual(by_id["missing_score"]["dock_norm"], 0.0)
        self.assertAlmostEqual(
            by_id["missing_score"]["composite"],
            round(0.5 * by_id["missing_score"]["ifp_norm"], 4),
        )

    def test_multicriteria_normalize_column_uses_neutral_imputation(self):
        series = pd.Series([1.0, np.nan, 3.0])
        normed = score_multicriteria.normalize_column(series, higher_is_better=False)

        self.assertAlmostEqual(normed.iloc[1], 0.5)
        self.assertGreater(normed.iloc[0], normed.iloc[2])

    def test_multicriteria_uses_shared_ifp_max_constant(self):
        self.assertEqual(score_multicriteria.IFP_MAX, run_plip.IFP_MAX)

    def test_standardize_dataframe_deduplicates_standardized_smiles(self):
        df = pd.DataFrame(
            {
                "SMILES": ["mol_a", "mol_b", "mol_c"],
                "Desired": [1, 1, 1],
            }
        )

        replacement = lambda value: {
            "mol_a": "canonical",
            "mol_b": "canonical",
            "mol_c": None,
        }[value]

        with mock.patch.object(standardize_generated, "standardize_smiles_fast", new=replacement):
            standardized = standardize_generated.standardize_dataframe(df)

        self.assertEqual(len(standardized), 1)
        self.assertEqual(standardized.iloc[0]["std_SMILES"], "canonical")
        self.assertFalse(bool(standardized.iloc[0]["std_failed"]))


if __name__ == "__main__":
    unittest.main()