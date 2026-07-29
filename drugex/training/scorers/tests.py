"""Portable regression tests for the ROCS scorer follow-up."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from drugex.training.scorers import rocs_cdpkit, rocs_openeye, rocs_rdkit
from drugex.training.scorers.rocs_cdpkit import (
    CDPKitROCSScorer,
    _align_and_score_helper,
)
from drugex.training.scorers.rocs_openeye import OpenEyeROCSScorer
from drugex.training.scorers.rocs_rdkit import (
    RDKitROCSScorer,
    _score_single_reference,
)


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
        "optimization_mode": None,
    }
    values.update(overrides)
    for name, value in values.items():
        setattr(scorer, name, value)
    return scorer


def _embedded_molecule(smiles: str, seed: int) -> Chem.Mol:
    """Create one deterministic 3D conformer for pose-invariance tests."""
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(molecule, params) != 0:
        raise RuntimeError(f"Could not embed test molecule: {smiles}")
    return molecule


def _rigid_transform(molecule: Chem.Mol, seed: int) -> Chem.Mol:
    """Return a copy under a deterministic proper rotation and translation."""
    random = np.random.default_rng(seed)
    matrix = random.normal(size=(3, 3))
    rotation, triangular = np.linalg.qr(matrix)
    rotation *= np.sign(np.diag(triangular))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = random.normal(size=3) * 10.0
    output = Chem.Mol(molecule)
    rdMolTransforms.TransformConformer(output.GetConformer(), transform)
    return output


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


class ROCSOptimizationModeTests(unittest.TestCase):
    """Verify explicit modes while retaining the legacy path."""

    @classmethod
    def setUpClass(cls):
        molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        params = AllChem.ETKDGv3()
        params.randomSeed = 7
        if AllChem.EmbedMolecule(molecule, params) != 0:
            raise RuntimeError("Could not build the portable ROCS test molecule")
        cls.molecule = molecule
        cls.pose_reference = _embedded_molecule(
            "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
            1,
        )
        cls.pose_probe = _embedded_molecule(
            "CCCCOc1ccc(CC(=O)NCc2ccccc2OC)cc1",
            2,
        )

    def test_rdkit_legacy_omits_opt_param(self):
        with patch.object(
            rocs_rdkit.rdShapeAlign,
            "AlignMol",
            return_value=(0.4, 0.2),
        ) as align:
            score = _score_single_reference(
                self.molecule,
                self.molecule,
                "TanimotoCombo",
                True,
            )

        self.assertAlmostEqual(score, 0.6)
        self.assertNotIn("opt_param", align.call_args.kwargs)

    def test_rdkit_modes_select_optimizer_and_component(self):
        cases = {
            "shape": ("shape", False, 1.0, 0.4),
            "combo": ("TanimotoCombo", True, 0.5, 0.6),
            "color": ("color", True, 0.0, 0.2),
        }
        for mode, (score_type, use_colors, opt_param, expected) in cases.items():
            with self.subTest(mode=mode):
                with patch.object(
                    rocs_rdkit.rdShapeAlign,
                    "AlignMol",
                    return_value=(0.4, 0.2),
                ) as align:
                    score = _score_single_reference(
                        self.molecule,
                        self.molecule,
                        score_type,
                        use_colors,
                        mode,
                    )

                self.assertAlmostEqual(score, expected)
                self.assertEqual(align.call_args.kwargs["opt_param"], opt_param)
                self.assertEqual(
                    align.call_args.kwargs["useColors"], use_colors
                )

    def test_rdkit_explicit_combo_has_distinct_key(self):
        legacy = RDKitROCSScorer(
            _DummyConformerGenerator(),
            self.molecule,
            n_jobs=1,
            show_progress=False,
        )
        explicit = RDKitROCSScorer(
            _DummyConformerGenerator(),
            self.molecule,
            n_jobs=1,
            show_progress=False,
            optimization_mode="combo",
        )

        self.assertNotEqual(legacy.getKey(), explicit.getKey())
        self.assertTrue(explicit.getKey()[0].endswith("_mode_combo"))

    def test_rdkit_color_mode_changes_flexible_overlay(self):
        reference = Chem.AddHs(
            Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        )
        reference_params = AllChem.ETKDGv3()
        reference_params.randomSeed = 13
        self.assertEqual(
            AllChem.EmbedMolecule(reference, reference_params),
            0,
        )
        AllChem.MMFFOptimizeMolecule(reference)

        probe = Chem.AddHs(
            Chem.MolFromSmiles("CN(C)CCCOc1ccc(C(=O)O)cc1")
        )
        probe_params = AllChem.ETKDGv3()
        probe_params.randomSeed = 7
        conformer_ids = AllChem.EmbedMultipleConfs(
            probe,
            numConfs=20,
            params=probe_params,
        )
        self.assertEqual(len(conformer_ids), 20)
        AllChem.MMFFOptimizeMoleculeConfs(probe)

        legacy_color = _score_single_reference(
            probe,
            reference,
            "color",
            True,
        )
        explicit_color = _score_single_reference(
            probe,
            reference,
            "color",
            True,
            "color",
        )

        self.assertGreater(explicit_color, legacy_color + 0.01)
        self.assertGreaterEqual(explicit_color, 0.0)
        self.assertLessEqual(explicit_color, 1.0)

    def test_rdkit_score_is_rigid_pose_invariant(self):
        scores = [
            _score_single_reference(
                _rigid_transform(self.pose_probe, seed),
                self.pose_reference,
                "TanimotoCombo",
                True,
                "combo",
            )
            for seed in range(6)
        ]

        self.assertLessEqual(max(scores) - min(scores), 5e-3)

    def test_rdkit_score_is_atom_order_invariant(self):
        renumbered = Chem.RenumberAtoms(
            self.pose_probe,
            list(reversed(range(self.pose_probe.GetNumAtoms()))),
        )
        original_score = _score_single_reference(
            self.pose_probe,
            self.pose_reference,
            "TanimotoCombo",
            True,
            "combo",
        )
        renumbered_score = _score_single_reference(
            renumbered,
            self.pose_reference,
            "TanimotoCombo",
            True,
            "combo",
        )

        self.assertAlmostEqual(
            original_score,
            renumbered_score,
            delta=5e-3,
        )

    def test_rdkit_alignment_does_not_mutate_inputs(self):
        probe_before = np.asarray(
            self.pose_probe.GetConformer().GetPositions()
        ).copy()
        reference_before = np.asarray(
            self.pose_reference.GetConformer().GetPositions()
        ).copy()

        _score_single_reference(
            self.pose_probe,
            self.pose_reference,
            "TanimotoCombo",
            True,
            "combo",
        )

        np.testing.assert_allclose(
            self.pose_probe.GetConformer().GetPositions(),
            probe_before,
        )
        np.testing.assert_allclose(
            self.pose_reference.GetConformer().GetPositions(),
            reference_before,
        )

    def test_rdkit_self_alignment_is_pose_invariant(self):
        scores = [
            _score_single_reference(
                _rigid_transform(self.pose_reference, 100 + seed),
                self.pose_reference,
                "shape",
                False,
                "shape",
            )
            for seed in range(4)
        ]

        self.assertGreaterEqual(min(scores), 0.99)
        self.assertLessEqual(max(scores) - min(scores), 5e-3)

    def test_rdkit_degenerate_geometry_fails_conservatively(self):
        helium = Chem.MolFromSmiles("[He]")
        conformer = Chem.Conformer(1)
        conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
        helium.AddConformer(conformer)

        score = _score_single_reference(
            helium,
            helium,
            "shape",
            False,
            "shape",
        )

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_openeye_modes_map_flags_columns_and_keys(self):
        cases = {
            "shape": ("ShapeTanimoto", True, False),
            "combo": ("TanimotoCombo", False, True),
            "color": ("ColorTanimoto", False, True),
        }
        with (
            patch.object(rocs_openeye, "OE_AVAILABLE", True),
            patch.object(OpenEyeROCSScorer, "_validate_query_files"),
            patch.object(rocs_openeye.shutil, "which", return_value="/opt/rocs"),
        ):
            for mode, (column, shape_only, color_optimize) in cases.items():
                with self.subTest(mode=mode):
                    scorer = OpenEyeROCSScorer(
                        _DummyConformerGenerator(),
                        {"reference": "reference.sdf"},
                        optimization_mode=mode,
                        show_progress=False,
                    )

                    self.assertEqual(scorer._score_column, column)
                    self.assertEqual(scorer.shape_only, shape_only)
                    self.assertEqual(scorer.color_optimize, color_optimize)
                    self.assertTrue(
                        scorer.getKey()[0].endswith(f"_mode_{mode}")
                    )

    def test_invalid_modes_fail_early(self):
        with self.assertRaisesRegex(ValueError, "optimization_mode"):
            RDKitROCSScorer(
                _DummyConformerGenerator(),
                self.molecule,
                optimization_mode="invalid",
                show_progress=False,
            )
        with (
            patch.object(rocs_openeye, "OE_AVAILABLE", True),
            self.assertRaisesRegex(ValueError, "optimization_mode"),
        ):
            OpenEyeROCSScorer(
                _DummyConformerGenerator(),
                {"reference": "reference.sdf"},
                optimization_mode="invalid",
                show_progress=False,
            )
        with (
            patch.object(rocs_cdpkit, "CDPL_AVAILABLE", True),
            self.assertRaisesRegex(ValueError, "optimization_mode"),
        ):
            CDPKitROCSScorer(
                _DummyConformerGenerator(),
                "reference.sdf",
                optimization_mode="invalid",
                show_progress=False,
            )

    def test_cdpkit_modes_select_scores_and_color_starts(self):
        class StartGenerator:
            last = None

            def __init__(self):
                self.color_starts = False
                self.aligned_centers = False
                StartGenerator.last = self

            def genColorCenterStarts(self, enabled):
                self.color_starts = enabled

            def genForAlignedShapeCenters(self, enabled):
                self.aligned_centers = enabled

        class Aligner:
            def setStartGenerator(self, generator):
                self.generator = generator

            def setMaxNumOptimizationIterations(self, iterations):
                self.iterations = iterations

            def setOptimizationStopGradient(self, gradient):
                self.gradient = gradient

            def addReferenceShape(self, reference):
                self.reference = reference

            def align(self, query):
                return True

            def getNumResults(self):
                return 1

            def getResult(self, index):
                return object()

        class ShapeModule:
            GaussianShapeAlignment = Aligner
            PrincipalAxesAlignmentStartGenerator = StartGenerator
            calcShapeTanimotoScore = staticmethod(lambda _: 0.4)
            calcTanimotoComboScore = staticmethod(lambda _: 0.6)
            calcColorTanimotoScore = staticmethod(lambda _: 0.2)

        expected = {"shape": 0.4, "combo": 0.6, "color": 0.2}
        with patch.object(rocs_cdpkit, "CDPLShape", ShapeModule):
            for mode, value in expected.items():
                with self.subTest(mode=mode):
                    score = _align_and_score_helper(object(), object(), mode)
                    self.assertEqual(score, value)
                    self.assertEqual(
                        StartGenerator.last.color_starts, mode == "color"
                    )
                    self.assertEqual(
                        StartGenerator.last.aligned_centers, mode == "color"
                    )

    @unittest.skipUnless(
        rocs_cdpkit.CDPL_AVAILABLE,
        "CDPKit is not available",
    )
    def test_cdpkit_modes_reach_worker_end_to_end(self):
        from drugex.training.scorers.conformer_generators import (
            RDKitConformerGenerator,
        )

        reference = Chem.AddHs(
            Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        )
        params = AllChem.ETKDGv3()
        params.randomSeed = 13
        self.assertEqual(AllChem.EmbedMolecule(reference, params), 0)
        AllChem.MMFFOptimizeMolecule(reference)

        with tempfile.TemporaryDirectory() as tmpdir:
            reference_path = Path(tmpdir) / "reference.sdf"
            with Chem.SDWriter(str(reference_path)) as writer:
                writer.write(reference)

            generator = RDKitConformerGenerator(
                max_conformers=3,
                max_isomers=1,
                num_threads=1,
                timeout=30,
            )
            scores = {}
            keys = {}
            for mode in ("shape", "combo", "color"):
                scorer = CDPKitROCSScorer(
                    generator,
                    str(reference_path),
                    show_progress=False,
                    n_jobs=1,
                    optimization_mode=mode,
                )
                scores[mode] = float(
                    scorer.getScores(
                        ["CN(C)CCCOc1ccc(C(=O)O)cc1"]
                    )[0, 0]
                )
                keys[mode] = scorer.getKey()[0]

        self.assertLessEqual(scores["shape"], 1.0)
        self.assertLessEqual(scores["color"], 1.0)
        self.assertLessEqual(scores["combo"], 2.0)
        self.assertEqual(len(set(keys.values())), 3)
        self.assertNotAlmostEqual(
            scores["combo"],
            scores["color"],
            delta=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
