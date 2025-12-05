import pytest
import torch

pytest.importorskip("merlin")


def test_photonic_cell_forward_shapes():
    from lib.photonic_quantum_cell import PhotonicQuantumLSTMCell

    input_size, hidden_size, output_size = 1, 2, 1

    cell = PhotonicQuantumLSTMCell(input_size, hidden_size, output_size)

    bsz = 2
    x = torch.randn(bsz, input_size, dtype=torch.float64)
    h = torch.zeros(bsz, hidden_size, dtype=torch.float64)
    c = torch.zeros_like(h)

    out, (h_new, c_new) = cell(x, (h, c))

    assert out.shape == (bsz, output_size)
    assert h_new.shape == (bsz, hidden_size)
    assert c_new.shape == (bsz, hidden_size)
