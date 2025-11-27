from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from merlin import QuantumLayer, ComputationSpace
from torch import nn
from utils.quantum import create_quantum_circuit


@dataclass(frozen=True)
class ArchitectureSpec:
    """Configuration of a single HQNN photon architecture."""

    modes: int
    photons: int
    no_bunching: bool
    param_count: int


class ScaleLayer(nn.Module):
    """Element-wise learnable scaling layer used ahead of the quantum layer."""

    def __init__(self, dim: int, scale_type: str = "learned") -> None:
        super().__init__()
        if scale_type == "learned":
            self.scale = nn.Parameter(torch.rand(dim))
        elif scale_type == "2pi":
            self.register_buffer("scale", torch.full((dim,), 2 * torch.pi))
        elif scale_type == "pi":
            self.register_buffer("scale", torch.full((dim,), torch.pi))
        elif scale_type == "1":
            self.register_buffer("scale", torch.ones(dim))
        else:
            raise ValueError(f"Unsupported scale_type: {scale_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def _candidate_modes() -> Sequence[int]:
    return tuple(range(2, 30, 2))


def enumerate_architectures(
    num_features: int, num_classes: int
) -> list[ArchitectureSpec]:
    """Enumerate HQNN architectures sorted by parameter count."""
    specs: list[ArchitectureSpec] = []
    for modes in _candidate_modes():
        max_photons = modes // 2
        for photons in range(1, max_photons + 1):
            for no_bunching in (True, False):
                if no_bunching:
                    output_size = math.comb(modes, photons)
                else:
                    output_size = math.comb(modes + photons - 1, photons)

                circuit = create_quantum_circuit(modes, size=num_features)
                trainable_parameters = sum(
                    1 for p in circuit.get_parameters() if not p.name.startswith("px")
                )

                param_count = (
                    num_features + trainable_parameters + output_size * num_classes
                )
                specs.append(
                    ArchitectureSpec(
                        modes=modes,
                        photons=photons,
                        no_bunching=no_bunching,
                        param_count=param_count,
                    )
                )

    specs.sort(key=lambda spec: spec.param_count)
    return specs


def build_hqnn_model(
    num_features: int,
    num_classes: int,
    architecture: ArchitectureSpec,
    device: torch.device | str,
) -> tuple[nn.Module, list[int], int]:
    """Instantiate the HQNN model for a given architecture."""

    input_state = [0] * architecture.modes
    for index in range(architecture.photons):
        input_state[2 * index] = 1

    circuit = create_quantum_circuit(architecture.modes, size=num_features)
    computation_space = ComputationSpace.UNBUNCHED if architecture.no_bunching else ComputationSpace.FOCK
    quantum_layer = QuantumLayer(
        input_size=num_features,
        circuit=circuit,
        trainable_parameters=[
            p.name for p in circuit.get_parameters() if not p.name.startswith("px")
        ],
        input_parameters=["px"],
        input_state=input_state,
        computation_space=computation_space,
        device=device,
    )
    output_size = quantum_layer.output_size
    classification_layer = nn.Linear(
        in_features=output_size,
        out_features=num_classes,
        bias=True,
    )
    nn.init.constant_(classification_layer.bias, 0.0)

    model = nn.Sequential(
        ScaleLayer(num_features, scale_type="learned"),
        quantum_layer,
        classification_layer,
    ).to(device)

    return model, input_state, output_size
