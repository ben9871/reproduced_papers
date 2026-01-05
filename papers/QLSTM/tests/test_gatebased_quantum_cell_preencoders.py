import torch
from lib.gatebased_quantum_cell import GateBasedQuantumLSTMCell
from lib.model import SequenceModel


def test_gatebased_quantum_cell_with_preencoders_shapes():
    input_size, hidden_size, output_size = 1, 2, 1
    cell = GateBasedQuantumLSTMCell(
        input_size, hidden_size, output_size, vqc_depth=1, use_preencoders=True
    ).double()

    bsz = 2
    x = torch.randn(bsz, input_size, dtype=torch.double)
    h = torch.zeros(bsz, hidden_size, dtype=torch.double)
    c = torch.zeros_like(h)

    out, (h_new, c_new) = cell(x, (h, c))

    assert out.shape == (bsz, output_size)
    assert h_new.shape == (bsz, hidden_size)
    assert c_new.shape == (bsz, hidden_size)


def test_sequence_model_with_preencoders_shapes():
    input_size, hidden_size, output_size = 1, 2, 1
    cell = GateBasedQuantumLSTMCell(
        input_size, hidden_size, output_size, vqc_depth=1, use_preencoders=True
    ).double()
    model = SequenceModel(cell, hidden_size).double()

    bsz, T = 2, 3
    x = torch.randn(bsz, T, input_size, dtype=torch.double)
    y, (h, c) = model(x)

    assert y.shape == (bsz, T, output_size)
