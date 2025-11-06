"""Model definitions for the QCNN data classification reproduction."""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn

from utils.circuit import build_single_gi_layer


class SingleGI(nn.Module):
    """Photonic layer applying a single Gaussian interferometer."""

    def __init__(
        self,
        n_modes: int,
        n_features: int,
        n_photons: int,
        reservoir_mode: bool,
        state_pattern: str,
        required_inputs: int,
        input_dim: int,
    ) -> None:
        super().__init__()
        if input_dim != required_inputs:
            raise ValueError(
                f"Quantum layer expects {required_inputs} features but received {input_dim}. "
                "Adjust preprocessing or circuit parameters so they match exactly."
            )
        self.K = required_inputs
        self.q_layer = build_single_gi_layer(
            n_modes=n_modes,
            n_features=n_features,
            n_photons=n_photons,
            reservoir_mode=reservoir_mode,
            state_pattern=state_pattern,
        )

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        if angles.dim() == 1:
            angles = angles.unsqueeze(0)
        if angles.shape[-1] != self.K:
            raise ValueError(
                f"Expected input with last dimension {self.K}, got {angles.shape[-1]}."
            )
        return self.q_layer(angles)


class QuantumPatchKernel(nn.Module):
    """Wrapper that applies a QuantumLayer to sliding PCA patches."""

    def __init__(self, q_layer: nn.Module, patch_dim: int) -> None:
        super().__init__()
        self.q_layer = q_layer
        self.output_size = getattr(q_layer, "output_size", None)
        if self.output_size is None:
            raise ValueError("QuantumLayer must expose an output_size attribute.")
        self.patch_dim = patch_dim

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        if patch.dim() == 1:
            patch = patch.unsqueeze(0)
        if patch.shape[-1] != self.patch_dim:
            raise ValueError(
                f"Expected patch with last dimension {self.patch_dim}, got {patch.shape[-1]}."
            )
        return self.q_layer(patch)


class QConvModel(nn.Module):
    """Pseudo-convolutional model mixing classical or quantum kernels."""

    def __init__(
        self,
        input_dim: int,
        n_kernels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        kernel_modules: List[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if n_kernels <= 0:
            raise ValueError("n_kernels must be a positive integer.")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be a positive integer.")
        if stride <= 0:
            raise ValueError("stride must be a positive integer.")

        self.input_dim = input_dim
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_quantum = kernel_modules is not None

        if self.use_quantum:
            if kernel_modules is None or len(kernel_modules) != n_kernels:
                raise ValueError("kernel_modules must contain one module per kernel.")
            self.kernel_modules = nn.ModuleList(kernel_modules)
            sample_output = getattr(self.kernel_modules[0], "output_size", None)
            if sample_output is None:
                raise ValueError("Quantum kernels must expose an output_size attribute.")
            self.kernel_output_dim = sample_output
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        else:
            weight = torch.empty(n_kernels, kernel_size)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            self.weight = nn.Parameter(weight)
            if bias:
                bound = 1 / kernel_size**0.5
                self.bias = nn.Parameter(torch.empty(n_kernels).uniform_(-bound, bound))
            else:
                self.register_parameter("bias", None)
            self.kernel_modules = None
            self.kernel_output_dim = 1

        self.output_features = self.output_dim(input_dim)
        self.head = nn.Linear(self.output_features, 2)

    def _compute_num_windows(self, input_len: int) -> int:
        if input_len < self.kernel_size:
            raise ValueError(
                f"Input length {input_len} is smaller than kernel size {self.kernel_size}."
            )
        return 1 + (input_len - self.kernel_size) // self.stride

    def output_dim(self, input_len: int) -> int:
        num_windows = self._compute_num_windows(input_len)
        return num_windows * self.n_kernels * self.kernel_output_dim

    def _apply_classical(self, patches: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("bpk,nk->bpn", patches, self.weight)
        if self.bias is not None:
            out = out + self.bias.view(1, 1, -1)
        out = out.permute(0, 2, 1).contiguous()
        return out.view(out.size(0), -1)

    def _apply_quantum(self, patches: torch.Tensor, num_windows: int, batch_size: int) -> torch.Tensor:
        patches_flat = patches.contiguous().view(-1, patches.size(-1))
        outputs = []
        for kernel in self.kernel_modules:
            y = kernel(patches_flat)
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            y = y.view(batch_size, num_windows, -1)
            outputs.append(y)
        out = torch.stack(outputs, dim=1)
        return out.view(out.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError(
                "QConvModel expects inputs of shape (batch, features) or a single feature vector."
            )

        patches = x.unfold(dimension=-1, size=self.kernel_size, step=self.stride)
        num_windows = patches.size(1)
        if num_windows == 0:
            raise ValueError(
                "Unfold produced zero windows. Check kernel_size and stride values."
            )

        if self.use_quantum:
            features = self._apply_quantum(patches, num_windows, x.size(0))
        else:
            features = self._apply_classical(patches)
        return self.head(features)
