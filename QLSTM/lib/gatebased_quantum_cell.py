from __future__ import annotations

import logging
import pennylane as qml
import torch
import torch.nn as nn


def _hadamard_layer(n: int):
    for i in range(n):
        qml.Hadamard(wires=i)


def _ry_layer(weights):
    for i, w in enumerate(weights):
        qml.RY(w, wires=i)


def _entangling_layer(n: int):
    for i in range(0, n - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, n - 1, 2):
        qml.CNOT(wires=[i, i + 1])


def _q_node(x, q_weights, n_class):
    depth = q_weights.shape[0]
    n_qubits = q_weights.shape[1]
    _hadamard_layer(n_qubits)
    # If input feature vector is shorter than n_qubits, pad with zeros to match wires.
    x_padded = x
    if x.shape[0] < n_qubits:
        pad = torch.zeros(n_qubits - x.shape[0], dtype=x.dtype, device=x.device)
        x_padded = torch.cat([x, pad], dim=0)
    _ry_layer(x_padded)
    for _ in range(depth):
        _entangling_layer(n_qubits)
        _ry_layer(q_weights[_])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_class)]


class VQC(nn.Module):
    def __init__(self, depth: int, n_qubits: int, n_class: int, device_name: str = 'default.qubit', shots: int | None = None):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(depth, n_qubits))
        # shots=None -> analytic expectation; shots>0 -> finite-shot sampling
        _shots = None if (shots is None or int(shots) <= 0) else int(shots)
        self.dev = qml.device(device_name, wires=n_qubits, shots=_shots)
        self.n_class = n_class
        self.qnode = qml.QNode(_q_node, self.dev, interface='torch')

    def forward(self, x: torch.Tensor):
        outs = []
        for sample in x:
            res = self.qnode(sample, self.weights, self.n_class)
            outs.append(torch.stack(res))
        return torch.stack(outs)


class GateBasedQuantumLSTMCell(nn.Module):
    """QLSTM cell where LSTM gates are implemented by variational quantum circuits (VQCs).

    Modes:
      - Default (4 VQC): gates i,f,g,o consume concat([x, h]).
      - Pre-encoders (6 VQC): two extra VQC encode x and h separately into R^{hidden_size},
        then combined as z (concat or sum) and fed to the 4 gate VQC.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, vqc_depth: int, device_name: str = 'default.qubit', *, use_preencoders: bool = False, shots: int | None = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_preencoders = use_preencoders

        if use_preencoders:
            # Two encoders: map x (R^{input_size}) -> R^{hidden_size}, and h (R^{hidden_size}) -> R^{hidden_size}
            # Use hidden_size qubits so we can measure hidden_size expectation values.
            # Only the first input_size qubits receive feature RY rotations; the rest stay at |+>.
            self.x_encoder = VQC(vqc_depth, n_qubits=hidden_size, n_class=hidden_size, device_name=device_name, shots=shots)
            self.h_encoder = VQC(vqc_depth, n_qubits=hidden_size, n_class=hidden_size, device_name=device_name, shots=shots)
            # Gate VQCs consume concatenated encodings of size 2*hidden_size
            gate_n_qubits = 2 * hidden_size
        else:
            # Gates consume concat([x, h]) directly
            gate_n_qubits = input_size + hidden_size

        self.input_gate = VQC(vqc_depth, gate_n_qubits, hidden_size, device_name, shots=shots)
        self.forget_gate = VQC(vqc_depth, gate_n_qubits, hidden_size, device_name, shots=shots)
        self.cell_gate = VQC(vqc_depth, gate_n_qubits, hidden_size, device_name, shots=shots)
        self.output_gate = VQC(vqc_depth, gate_n_qubits, hidden_size, device_name, shots=shots)
        self.output_proj = torch.nn.Linear(hidden_size, output_size)

        # --- Logging: architecture summary ---
        logger = logging.getLogger(__name__)
        gate_n_qubits = (2 * hidden_size) if use_preencoders else (input_size + hidden_size)
        def _params(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if getattr(p, 'requires_grad', False))
        total_params = _params(self)
        parts = [
            "GateBasedQuantumLSTMCell initialized:",
            f"  - device={device_name} shots={'analytic' if (shots is None or int(shots) <= 0) else int(shots)}",
            f"  - VQC depth per circuit={vqc_depth}",
            f"  - qubits per gate-VQC={gate_n_qubits}",
            f"  - pre-encoders={'enabled' if use_preencoders else 'disabled'}",
            f"  - hidden_size={hidden_size}, output_size={output_size}",
            f"  - total trainable parameters ≈ {total_params}",
        ]
        logger.info("\n".join(parts))

    def forward(self, x, state):
        h_prev, c_prev = state
        if self.use_preencoders:
            # Encode separately then concat
            x_enc = self.x_encoder(x)
            h_enc = self.h_encoder(h_prev)
            comb = torch.cat([x_enc, h_enc], dim=1)
        else:
            comb = torch.cat([x, h_prev], dim=1)
        i = torch.sigmoid(self.input_gate(comb))
        f = torch.sigmoid(self.forget_gate(comb))
        g = torch.tanh(self.cell_gate(comb))
        o = torch.sigmoid(self.output_gate(comb))
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        out = self.output_proj(h_t)
        return out, (h_t, c_t)
