"""Regression tests for scorer temporary-directory behavior."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from drugex.training.scorers import rocs_cdpkit, rocs_rdkit
from drugex.training.scorers.rocs_cdpkit import CDPKitROCSScorer
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer


class _MissingConformerGenerator:
    """Return a missing file after the scorer enters its temporary directory."""

    def genConformers(self, smiles, output_dir):
        return f"{output_dir}/missing.sdf"


def _temporary_directory_mock():
    manager = MagicMock()
    manager.__enter__.return_value = "/tmp/drugex-scorer-test"
    manager.__exit__.return_value = False
    return manager


class TemporaryDirectoryTests(unittest.TestCase):
    """Verify both open-source backends request cleanup-error tolerance."""

    def test_rdkit_ignores_cleanup_errors(self):
        scorer = object.__new__(RDKitROCSScorer)
        scorer.group_to_indices = [[0]]
        scorer.show_progress = False
        scorer.conformer_generator = _MissingConformerGenerator()
        manager = _temporary_directory_mock()

        with patch.object(
            rocs_rdkit.tempfile,
            "TemporaryDirectory",
            return_value=manager,
        ) as temporary_directory:
            scores = scorer.getScores(["CCO"])

        temporary_directory.assert_called_once_with(ignore_cleanup_errors=True)
        np.testing.assert_array_equal(scores, np.zeros((1, 1)))

    @unittest.skipUnless(
        rocs_cdpkit.CDPL_AVAILABLE,
        "CDPKit is not available",
    )
    def test_cdpkit_ignores_cleanup_errors(self):
        scorer = object.__new__(CDPKitROCSScorer)
        scorer.group_to_indices = [[0]]
        scorer.show_progress = False
        scorer.conformer_generator = _MissingConformerGenerator()
        manager = _temporary_directory_mock()

        with patch.object(
            rocs_cdpkit.tempfile,
            "TemporaryDirectory",
            return_value=manager,
        ) as temporary_directory:
            scores = scorer.getScores(["CCO"])

        temporary_directory.assert_called_once_with(ignore_cleanup_errors=True)
        np.testing.assert_array_equal(scores, np.zeros((1, 1)))


if __name__ == "__main__":
    unittest.main()
