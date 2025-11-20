"""Model assembly for QLSTM reproduction."""
from __future__ import annotations

import torch
import torch.nn as nn

from .classical_cell import ClassicalLSTMCell
from .gatebased_quantum_cell import GateBasedQuantumLSTMCell
from .photonic_quantum_cell import PhotonicQuantumLSTMCell


class SequenceModel(nn.Module):
    def __init__(self, cell: nn.Module, hidden_size: int):
        super().__init__()
        self.cell = cell
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor):
        bsz, timesteps, _ = x.shape
        h = torch.zeros(bsz, self.hidden_size, device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)
        outs = []
        for t in range(timesteps):
            o, (h, c) = self.cell(x[:, t, :], (h, c))
            outs.append(o.unsqueeze(1))
        return torch.cat(outs, dim=1), (h, c)


def build_model(
    model_type: str,
    input_size: int,
    hidden_size: int,
    vqc_depth: int,
    output_size: int = 1,
    shots: int = 0,
    device_name: str = 'default.qubit', # gate-based QLSTM only
    use_preencoders: bool = False, # gate-based QLSTM only
    use_photonic_head: bool = False, # photonic QLSTM only
):
    if model_type == 'lstm':
        cell = ClassicalLSTMCell(input_size, hidden_size, output_size)
    elif model_type == 'qlstm':
        cell = GateBasedQuantumLSTMCell(
            input_size, hidden_size, output_size, vqc_depth, device_name,
            use_preencoders=use_preencoders,
            shots=shots,
        )
    elif model_type == 'qlstm_photonic':
        # Provide sensible defaults; vqc_depth is unused here
        cell = PhotonicQuantumLSTMCell(
            input_size, hidden_size, output_size,
            shots=shots, use_photonic_head=use_photonic_head,
        )
    else:
        raise ValueError(f"Unknown model_type {model_type}")
    return SequenceModel(cell, hidden_size)
