import pathlib
import sys
import torch

# Ensure this tests directory is on sys.path to import shared helper
_TESTS_DIR = pathlib.Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from common import _load_impl_module

_ = _load_impl_module()  # injects QLSTM folder in sys.path
from lib.classical_cell import ClassicalLSTMCell


def test_classical_cell_forward_shapes():
    input_size, hidden_size, output_size = 1, 5, 1
    cell = ClassicalLSTMCell(input_size, hidden_size, output_size).double()
    bsz = 3
    x = torch.randn(bsz, input_size, dtype=torch.double)
    h = torch.zeros(bsz, hidden_size, dtype=torch.double)
    c = torch.zeros_like(h)

    out, (h_new, c_new) = cell(x, (h, c))

    assert out.shape == (bsz, output_size)
    assert h_new.shape == (bsz, hidden_size)
    assert c_new.shape == (bsz, hidden_size)


def test_classical_sequence_step_matches_model():
    # quick check that wrapping in SequenceModel yields consistent time dimension
    from lib.model import SequenceModel
    input_size, hidden_size, output_size = 1, 4, 1
    cell = ClassicalLSTMCell(input_size, hidden_size, output_size).double()
    model = SequenceModel(cell, hidden_size).double()

    bsz, T = 2, 6
    x = torch.randn(bsz, T, input_size, dtype=torch.double)
    y, (h, c) = model(x)

    assert y.shape == (bsz, T, output_size)
