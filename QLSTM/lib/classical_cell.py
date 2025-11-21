from __future__ import annotations

import logging

import torch
import torch.nn as nn


class ClassicalLSTMCell(nn.Module):
    """Classical LSTM cell using linear gates.

    Input shape: (batch, input_size)
    Hidden state: (h, c), each of shape (batch, hidden_size)
    Output: (batch, output_size)
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        in_h = input_size + hidden_size
        self.input_gate = nn.Linear(in_h, hidden_size)
        self.forget_gate = nn.Linear(in_h, hidden_size)
        self.cell_gate = nn.Linear(in_h, hidden_size)
        self.output_gate = nn.Linear(in_h, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

        # --- Logging: architecture summary ---
        logger = logging.getLogger(__name__)

        def _params(m: nn.Module) -> int:
            return sum(
                p.numel() for p in m.parameters() if getattr(p, "requires_grad", False)
            )

        total_params = _params(self)
        parts = [
            "ClassicalLSTMCell initialized:",
            f"  - input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}",
            f"  - gate layers: Linear({in_h}->{hidden_size}) x4",
            f"  - head layer: Linear({hidden_size}->{output_size})",
            f"  - total trainable parameters ≈ {total_params}",
        ]
        logger.info("\n".join(parts))

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h_prev, c_prev = state
        comb = torch.cat([x, h_prev], dim=1)
        i = torch.sigmoid(self.input_gate(comb))
        f = torch.sigmoid(self.forget_gate(comb))
        g = torch.tanh(self.cell_gate(comb))
        o = torch.sigmoid(self.output_gate(comb))
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        out = self.output_proj(h_t)
        return out, (h_t, c_t)
