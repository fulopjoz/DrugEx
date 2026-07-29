"""Portable regression tests for the ROCS scorer follow-up."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from drugex.training.scorers import rocs_openeye
from drugex.training.scorers.rocs_openeye import OpenEyeROCSScorer


class _DummyConformerGenerator:
    """Record SMILES passed to the OpenEye scorer in tests."""

    def __init__(self):
        self.smiles = None

    def genConformers(self, smiles, output_dir):
        self.smiles = list(smiles)
        return str(Path(output_dir) / "conformers.sdf")


def _bare_scorer(**overrides):
    """Construct a scorer without optional OpenEye imports."""
    scorer = object.__new__(OpenEyeROCSScorer)
    values = {
        "binary_path": "rocs",
        "shape_only": False,
        "score_type": "TanimotoCombo",
        "_score_column": "TanimotoCombo",
        "color_force_field": "ImplicitMillsDean",
        "optimize": True,
        "color_optimize": True,
        "timeout": 7,
        "show_progress": False,
        "queries": {"reference": ["reference.sdf"]},
    }
    values.update(overrides)
    for name, value in values.items():
        setattr(scorer, name, value)
    return scorer


class OpenEyeROCSFollowupTests(unittest.TestCase):
    """Characterize behavior that does not require an OpenEye license."""

    def test_deduplicate_smiles_preserves_original_mapping(self):
        unique, mapping = OpenEyeROCSScorer._deduplicate_smiles(
            ["CCO", None, "CCN", "CCO"]
        )

        self.assertEqual(unique, ["CCO", "CCN"])
        self.assertEqual(dict(mapping), {0: [0, 3], 1: [2]})

    def test_get_scores_maps_unique_scores_back_to_duplicates(self):
        generator = _DummyConformerGenerator()
        scorer = _bare_scorer(conformer_generator=generator)
        scorer._score = lambda _: {"reference": {0: 0.8, 1: 0.3}}

        scores = scorer.getScores(["CCO", None, "CCN", "CCO"])

        self.assertEqual(generator.smiles, ["CCO", "CCN"])
        np.testing.assert_allclose(scores[:, 0], [0.8, 0.0, 0.3, 0.8])

    def test_shape_only_command_does_not_request_color_optimization(self):
        scorer = _bare_scorer(shape_only=True, _score_column="ShapeTanimoto")

        command = scorer._build_rocs_command("query.sq", "input.sdf", "output.tsv")

        self.assertIn("-shapeonly", command)
        self.assertNotIn("-optchem", command)

    def test_combo_command_respects_color_optimization(self):
        scorer = _bare_scorer(color_optimize=False)

        command = scorer._build_rocs_command("query.sdf", "input.sdf", "output.tsv")

        optchem_index = command.index("-optchem")
        self.assertEqual(command[optchem_index + 1], "false")

    def test_shape_only_results_use_shape_tanimoto_column(self):
        scorer = _bare_scorer(shape_only=True, _score_column="ShapeTanimoto")
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Path(tmpdir) / "report.tsv"
            report.write_text(
                "Name\tShapeTanimoto\nmol_0+0\t0.4\nmol_0+1\t0.7\n",
                encoding="utf-8",
            )

            scores = scorer._parse_results(str(report))

        self.assertEqual(scores, {0: 0.7})

    def test_timeout_reports_configured_limit(self):
        scorer = _bare_scorer(timeout=13)
        with tempfile.TemporaryDirectory() as tmpdir:
            query = Path(tmpdir) / "query.sdf"
            inputs = Path(tmpdir) / "input.sdf"
            query.touch()
            inputs.touch()
            with (
                patch.object(rocs_openeye.shutil, "which", return_value="/opt/rocs"),
                patch.object(
                    rocs_openeye.subprocess,
                    "run",
                    side_effect=subprocess.TimeoutExpired("rocs", 13),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "timed out after 13s"):
                    scorer._execute_rocs(str(query), str(inputs), tmpdir)

    def test_parallel_request_warns_and_uses_one_process(self):
        with (
            patch.object(rocs_openeye, "OE_AVAILABLE", True),
            patch.object(OpenEyeROCSScorer, "_validate_query_files"),
            patch.object(rocs_openeye.shutil, "which", return_value="/opt/rocs"),
        ):
            with self.assertWarnsRegex(UserWarning, "n_jobs has no effect"):
                scorer = OpenEyeROCSScorer(
                    conformer_generator=_DummyConformerGenerator(),
                    references={"reference": "reference.sdf"},
                    n_jobs=4,
                    show_progress=False,
                )

        self.assertEqual(scorer.n_jobs, 1)


if __name__ == "__main__":
    unittest.main()
