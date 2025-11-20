import pathlib
import sys
import pytest
import torch

_TESTS_DIR = pathlib.Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from common import _load_impl_module

_ = _load_impl_module()
from lib.gatebased_quantum_cell import GateBasedQuantumLSTMCell
from lib.model import SequenceModel


def test_gatebased_quantum_cell_forward_shapes():
    input_size, hidden_size, output_size = 1, 2, 1
    cell = GateBasedQuantumLSTMCell(input_size, hidden_size, output_size, vqc_depth=1).double()

    bsz = 2
    x = torch.randn(bsz, input_size, dtype=torch.double)
    h = torch.zeros(bsz, hidden_size, dtype=torch.double)
    c = torch.zeros_like(h)

    out, (h_new, c_new) = cell(x, (h, c))

    assert out.shape == (bsz, output_size)
    assert h_new.shape == (bsz, hidden_size)
    assert c_new.shape == (bsz, hidden_size)


def test_gatebased_sequence_model_shapes():
    input_size, hidden_size, output_size = 1, 2, 1
    cell = GateBasedQuantumLSTMCell(input_size, hidden_size, output_size, vqc_depth=1).double()
    model = SequenceModel(cell, hidden_size).double()

    bsz, T = 2, 3
    x = torch.randn(bsz, T, input_size, dtype=torch.double)
    y, (h, c) = model(x)

    assert y.shape == (bsz, T, output_size)
