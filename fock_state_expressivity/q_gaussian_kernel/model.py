"""
Quantum Gaussian kernel model implementations using photonic circuits.

This module provides various quantum circuit architectures for implementing
quantum Gaussian kernels, including different interferometer configurations
and quantum layer constructions for kernel methods.
"""

import random

import numpy as np
import perceval as pcvl
import torch
import torch.nn as nn
from merlin import OutputMappingStrategy, QuantumLayer


def create_circuit(args):
    """
    Create a quantum photonic circuit for Gaussian kernel implementation.

    Args:
        args: Configuration object containing circuit parameters including:
            - circuit (str): Circuit type ("mzi", "general", "spiral", "ps_based", "bs_based", "general_all_angles")
            - train_circuit (bool): Whether the circuit parameters are trainable

    Returns:
        pcvl.Circuit: The constructed quantum photonic circuit
    """
    circuit = pcvl.Circuit(2)
    if args.circuit == "mzi":
        if args.train_circuit:
            left_side = pcvl.BS(
                theta=pcvl.P("theta_l_1"),
                phi_bl=pcvl.P("theta_l_2"),
                phi_tl=pcvl.P("theta_l_3"),
                phi_br=pcvl.P("theta_l_4"),
                phi_tr=pcvl.P("theta_l"),
            )
            right_side = pcvl.BS(
                theta=pcvl.P("theta_r_1"),
                phi_bl=pcvl.P("theta_r_2"),
                phi_tl=pcvl.P("theta_r_3"),
                phi_br=pcvl.P("theta_r_4"),
                phi_tr=pcvl.P("theta_r"),
            )
        else:
            left_side = pcvl.BS()
            right_side = pcvl.BS()
    elif args.circuit == "general_all_angles":
        if args.train_circuit:
            left_side = pcvl.GenericInterferometer(
                2,
                lambda i: pcvl.BS(
                    theta=pcvl.P(f"theta_l1{i}_bs"),
                    phi_tr=pcvl.P(f"theta_l2{i}_bs"),
                    phi_br=pcvl.P(f"theta_l3{i}_bs"),
                    phi_tl=pcvl.P(f"theta_l4{i}_bs"),
                    phi_bl=pcvl.P(f"theta_l5{i}_bs"),
                )
                // pcvl.PS(phi=pcvl.P(f"theta_psl1{i}"))
                // pcvl.BS(
                    theta=pcvl.P(f"theta_l6{i}_bs"),
                    phi_tr=pcvl.P(f"theta_l7{i}_bs"),
                    phi_br=pcvl.P(f"theta_l8{i}_bs"),
                    phi_tl=pcvl.P(f"theta_l9{i}_bs"),
                    phi_bl=pcvl.P(f"theta_l10{i}_bs"),
                )
                // pcvl.PS(phi=pcvl.P(f"theta_{i}")),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
            right_side = pcvl.GenericInterferometer(
                2,
                lambda i: pcvl.BS(
                    theta=pcvl.P(f"theta_r1{i}_bs"),
                    phi_tr=pcvl.P(f"theta_r2{i}_bs"),
                    phi_br=pcvl.P(f"theta_r3{i}_bs"),
                    phi_tl=pcvl.P(f"theta_r4{i}_bs"),
                    phi_bl=pcvl.P(f"theta_r5{i}_bs"),
                )
                // pcvl.PS(phi=pcvl.P(f"theta_psr1{i}"))
                // pcvl.BS(
                    theta=pcvl.P(f"theta_r6{i}_bs"),
                    phi_tr=pcvl.P(f"theta_r7{i}_bs"),
                    phi_br=pcvl.P(f"theta_r8{i}_bs"),
                    phi_tl=pcvl.P(f"theta_r9{i}_bs"),
                    phi_bl=pcvl.P(f"theta_r10{i}_bs"),
                )
                // pcvl.PS(phi=pcvl.P(f"theta_psr2{i}")),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
        else:
            left_side = pcvl.GenericInterferometer(
                2,
                lambda i: pcvl.BS() // pcvl.PS(phi=random.random() * np.pi),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
            right_side = pcvl.GenericInterferometer(
                2,
                lambda i: pcvl.BS() // pcvl.PS(phi=random.random() * np.pi),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
    elif args.circuit == "general":
        if args.train_circuit:
            left_side = pcvl.GenericInterferometer(
                2,
                lambda i: pcvl.BS(theta=pcvl.P(f"theta_l1{i}_bs"))
                // pcvl.PS(phi=pcvl.P(f"theta_psl1{i}"))
                // pcvl.BS(theta=pcvl.P(f"theta_l2{i}_bs"))
                // pcvl.PS(phi=pcvl.P(f"theta_{i}")),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
            right_side = pcvl.GenericInterferometer(
                2,
                lambda i: pcvl.BS(theta=pcvl.P(f"theta_r1{i}_bs"))
                // pcvl.PS(phi=pcvl.P(f"theta_psr1{i}"))
                // pcvl.BS(theta=pcvl.P(f"theta_r2{i}_bs"))
                // pcvl.PS(phi=pcvl.P(f"theta_psr2{i}")),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
        else:
            left_side = pcvl.GenericInterferometer(
                2,
                lambda i: pcvl.BS() // pcvl.PS(phi=random.random() * np.pi),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
            right_side = pcvl.GenericInterferometer(
                2,
                lambda i: pcvl.BS() // pcvl.PS(phi=random.random() * np.pi),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
    elif args.circuit == "spiral":
        if args.train_circuit:
            left_side = pcvl.GenericInterferometer(
                2,
                lambda i: pcvl.BS(theta=pcvl.P(f"theta_l1{i}_bs"))
                // pcvl.PS(phi=pcvl.P(f"theta_psl1{i}"))
                // pcvl.BS(theta=pcvl.P(f"theta_l2{i}_bs"))
                // pcvl.PS(phi=pcvl.P(f"theta_{i}")),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
            right_side = pcvl.GenericInterferometer(
                2,
                lambda i: pcvl.BS()
                // pcvl.PS(phi=pcvl.P(f"theta_psr1{i}"))
                // pcvl.BS()
                // pcvl.PS(phi=pcvl.P(f"theta_psr2{i}")),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
        else:
            left_side = pcvl.GenericInterferometer(
                2,
                lambda i: pcvl.BS() // pcvl.PS(phi=random.random() * np.pi),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
            right_side = pcvl.GenericInterferometer(
                2,
                lambda i: pcvl.BS() // pcvl.PS(phi=random.random() * np.pi),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
    elif args.circuit == "ps_based":
        if args.train_circuit:
            left_side = pcvl.GenericInterferometer(
                2,
                lambda i: pcvl.BS()
                // pcvl.PS(phi=pcvl.P(f"theta_l1{i}"))
                // pcvl.BS()
                // pcvl.PS(phi=pcvl.P(f"theta_{i}")),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
            right_side = pcvl.GenericInterferometer(
                2,
                lambda i: pcvl.BS()
                // pcvl.PS(phi=pcvl.P(f"theta_r1{i}"))
                // pcvl.BS()
                // pcvl.PS(phi=pcvl.P(f"theta_r2{i}")),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
        else:
            left_side = pcvl.GenericInterferometer(
                2,
                lambda i: pcvl.BS() // pcvl.PS(phi=random.random() * np.pi),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
            right_side = pcvl.GenericInterferometer(
                2,
                lambda i: pcvl.BS() // pcvl.PS(phi=random.random() * np.pi),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
    elif args.circuit == "bs_based":
        if args.train_circuit:
            left_side = pcvl.GenericInterferometer(
                2,
                lambda i: pcvl.BS(theta=pcvl.P(f"theta_l1{i}_bs"))
                // pcvl.PS(phi=random.random() * np.pi)
                // pcvl.BS(theta=pcvl.P(f"theta_l2{i}_bs"))
                // pcvl.PS(phi=random.random() * np.pi),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
            right_side = pcvl.GenericInterferometer(
                2,
                lambda i: pcvl.BS(theta=pcvl.P(f"theta_r1{i}_bs"))
                // pcvl.PS(phi=random.random() * np.pi)
                // pcvl.BS(theta=pcvl.P(f"theta_r2{i}_bs"))
                // pcvl.PS(phi=random.random() * np.pi),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
        else:
            left_side = pcvl.GenericInterferometer(
                2,
                lambda i: pcvl.BS() // pcvl.PS(phi=random.random() * np.pi),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
            right_side = pcvl.GenericInterferometer(
                2,
                lambda i: pcvl.BS() // pcvl.PS(phi=random.random() * np.pi),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )

    circuit.add(0, left_side, merge=True)
    circuit.add(0, pcvl.PS(pcvl.P("\u03b4")), merge=True)
    circuit.add(0, right_side, merge=True)
    return circuit


class ScaleLayer(nn.Module):
    """
    Multiply the input tensor by a learned or fixed factor for quantum encoding.

    Args:
        dim (int): Dimension of the input data to be encoded
        scale_type (str): Type of scaling method. Options:
            - "learned": Learnable parameter initialized to 1/(2π)
            - "2pi", "pi", "/pi", "1": Fixed scaling factors
            - "/2pi", "/3pi", "/4pi": Inverse scaling factors
            - "0.1": Decimal scaling factor
            - "paper": Special scaling as used in reference paper

    Returns:
        nn.Module that applies scaling transformation to input tensor
    """

    def __init__(self, dim, scale_type="learned"):
        self.scale_type = scale_type
        super().__init__()
        # Create a single learnable parameter (initialized to 0.1 by default)
        if scale_type == "learned":
            self.scale = nn.Parameter(torch.full((dim,), 1 / (2 * torch.pi)))
        elif scale_type == "2pi":
            self.scale = torch.full((dim,), 2 * torch.pi)
        elif scale_type == "pi":
            self.scale = torch.full((dim,), torch.pi)
        elif scale_type == "/pi":
            self.scale = torch.full((dim,), 1 / torch.pi)
        elif scale_type == "1":
            self.scale = torch.full((dim,), 1)
        elif scale_type == "0.1":
            self.scale = torch.full((dim,), 0.1)
        elif scale_type == "/2pi":
            self.scale = torch.full((dim,), 1 / (2 * torch.pi))
        elif scale_type == "/3pi":
            self.scale = torch.full((dim,), 1 / (3 * torch.pi))
        elif scale_type == "/4pi":
            self.scale = torch.full((dim,), 1 / (4 * torch.pi))
        elif scale_type == "paper":
            self.scale = torch.full((dim,), 1 / torch.pi)

    def forward(self, x):
        """Apply scaling transformation to input tensor."""
        if self.scale_type == "paper":
            result = x * self.scale - np.pi / 2
        else:
            result = x * self.scale
        return result


def create_quantum_layer(num_photons, args):
    """
    Create a quantum layer for Gaussian kernel implementation.

    Args:
        num_photons (int): Total number of photons in the initial state
        args: Configuration object containing model parameters

    Returns:
        nn.Sequential: Complete quantum model with scaling, quantum circuit, and linear layers
    """
    if num_photons % 2 == 0:
        input_state = [num_photons // 2, num_photons // 2]
    else:
        input_state = [(num_photons // 2) + 1, num_photons // 2]

    scale_layer = ScaleLayer(1, args.scale_type)

    circuit = create_circuit(args)

    if args.train_circuit:
        train_params = ["theta"]
    else:
        train_params = []

    qc = QuantumLayer(
        input_size=1,
        circuit=circuit,
        trainable_parameters=train_params,
        input_parameters=["\u03b4"],
        input_state=input_state,
        no_bunching=args.no_bunching,
        output_mapping_strategy=OutputMappingStrategy.NONE,
    )

    linear_layer = nn.Linear(qc.output_size, 1)

    # init.kaiming_uniform_(linear_layer.weight, nonlinearity='relu')
    # init.zeros_(linear_layer.bias)

    model = nn.Sequential(scale_layer, qc, linear_layer)

    return model
