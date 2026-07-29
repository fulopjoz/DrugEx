"""Regression tests for CPU device handling in generators and explorers."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from drugex.training.generators.sequence_rnn import SequenceRNN
from drugex.training.interfaces import _maybe_data_parallel


class _Vocabulary:
    """Minimal vocabulary required to construct a sequence generator."""

    size = 4
    max_len = 8
    tk2ix = {"GO": 0, "EOS": 1}


class DeviceHandlingTests(unittest.TestCase):
    """Verify CPU execution never constructs CUDA devices or DataParallel."""

    def test_cpu_device_object_skips_data_parallel(self):
        module = nn.Linear(2, 2)
        with patch("drugex.training.interfaces.nn.DataParallel") as wrapper:
            result = _maybe_data_parallel(module, torch.device("cpu"), (-1,))

        self.assertIs(result, module)
        wrapper.assert_not_called()

    def test_cpu_device_string_skips_data_parallel(self):
        module = nn.Linear(2, 2)
        with patch("drugex.training.interfaces.nn.DataParallel") as wrapper:
            result = _maybe_data_parallel(module, "cpu", (-1,))

        self.assertIs(result, module)
        wrapper.assert_not_called()

    def test_cuda_device_uses_requested_ids(self):
        module = nn.Linear(2, 2)
        sentinel = object()
        with patch(
            "drugex.training.interfaces.nn.DataParallel", return_value=sentinel
        ) as wrapper:
            result = _maybe_data_parallel(module, "cuda:1", (1, 2))

        self.assertIs(result, sentinel)
        wrapper.assert_called_once_with(module, device_ids=(1, 2))

    def test_sequence_rnn_accepts_cpu_string_and_sentinel(self):
        generator = SequenceRNN(
            _Vocabulary(),
            embed_size=2,
            hidden_size=4,
            device="cpu",
            use_gpus=(-1,),
        )

        self.assertEqual(generator.device, torch.device("cpu"))
        self.assertEqual(generator.gpus, (-1,))
        self.assertEqual(next(generator.parameters()).device, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
