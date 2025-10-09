#!/usr/bin/env python3
# merlin_reproduction.py
# Parallel-columns only. Always use PCA-8; adapt to circuit-required input count K with a selectable adapter.

from __future__ import annotations

import math, random, statistics, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import numpy as np
import perceval as pcvl
import merlin as ML
from enum import Enum

# ======================= State Generator =======================

class StatePattern(Enum):
    DEFAULT = "default"
    SPACED = "spaced"
    SEQUENTIAL = "sequential"
    PERIODIC = "periodic"

class StateGenerator:
    @staticmethod
    def generate_state(n_modes, n_photons, state_pattern):
        if n_photons < 0 or n_photons > n_modes:
            raise ValueError(f"Cannot place {n_photons} photons into {n_modes} modes.")
        if state_pattern == StatePattern.SPACED:
            return StateGenerator._generate_spaced_state(n_modes, n_photons)
        elif state_pattern == StatePattern.SEQUENTIAL:
            return StateGenerator._generate_sequential_state(n_modes, n_photons)
        elif state_pattern in [StatePattern.PERIODIC, StatePattern.DEFAULT]:
            return StateGenerator._generate_periodic_state(n_modes, n_photons)
        else:
            return StateGenerator._generate_periodic_state(n_modes, n_photons)

    @staticmethod
    def _generate_spaced_state(n_modes, n_photons):
        if n_photons == 0:
            return [0] * n_modes
        if n_photons == 1:
            pos = n_modes // 2
            return [1 if i == pos else 0 for i in range(n_modes)]
        positions = [int(i * n_modes / n_photons) for i in range(n_photons)]
        positions = [min(pos, n_modes - 1) for pos in positions]
        occ = [0] * n_modes
        for pos in positions:
            occ[pos] += 1
        return occ

    @staticmethod
    def _generate_periodic_state(n_modes, n_photons):
        bits = [1 if i % 2 == 0 else 0 for i in range(min(n_photons * 2, n_modes))]
        count = sum(bits)
        i = 0
        while count < n_photons and i < n_modes:
            if i >= len(bits):
                bits.append(0)
            if bits[i] == 0:
                bits[i] = 1
                count += 1
            i += 1
        padding = [0] * (n_modes - len(bits))
        return bits + padding

    @staticmethod
    def _generate_sequential_state(n_modes, n_photons):
        return [1 if i < n_photons else 0 for i in range(n_modes)]

# ======================= Parallel-Columns Circuit Only =======================

def _generate_interferometer(n_modes, stage_idx, reservoir_mode=False):
    if reservoir_mode:
        return pcvl.GenericInterferometer(
            n_modes,
            lambda idx: pcvl.BS(theta=np.pi * 2 * random.random())
            // (0, pcvl.PS(phi=np.pi * 2 * random.random())),
            shape=pcvl.InterferometerShape.RECTANGLE,
            depth=2 * n_modes,
            phase_shifter_fun_gen=lambda idx: pcvl.PS(
                phi=np.pi * 2 * random.random()
            ),
        )
    else:
        def mzi(P1, P2):
            return (
                pcvl.Circuit(2)
                .add((0, 1), pcvl.BS())
                .add(0, pcvl.PS(P1))
                .add((0, 1), pcvl.BS())
                .add(0, pcvl.PS(P2))
            )

    offset = stage_idx * (n_modes * (n_modes - 1) // 2)
    shape = pcvl.InterferometerShape.RECTANGLE
    return pcvl.GenericInterferometer(
        n_modes,
        fun_gen=lambda idx: mzi(
            pcvl.P(f"phi_0{offset + idx}"), pcvl.P(f"phi_1{offset + idx}")
        ),
        shape=shape,
        phase_shifter_fun_gen=lambda idx: pcvl.PS(
            phi=pcvl.P(f"phi_02{stage_idx}_{idx}")
        ),
    )

def build_parallel_columns_circuit(n_modes: int, n_features: int, reservoir_mode: bool=False):
    """
    Columns of: [Interferometer] -> [per-mode PS (trainable inputs)] repeated n_features times,
    then a final [Interferometer].
    """
    circuit = pcvl.Circuit(n_modes)
    ps_idx = 0
    for stage in range(n_features + 1):
        circuit.add(0, _generate_interferometer(n_modes, stage, reservoir_mode))
        if stage < n_features:
            for m_idx in range(n_modes):
                circuit.add(m_idx, pcvl.PS(pcvl.P(f"pl{ps_idx}x")))
                ps_idx += 1
    return circuit

def create_quantum_circuit(n_modes: int, n_features: int):
    # 1. Left interferometer - trainable transformation
    wl = pcvl.GenericInterferometer(
        n_modes,
        lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_li{i}")) //
                 pcvl.BS() // pcvl.PS(pcvl.P(f"theta_lo{i}")),
        shape=pcvl.InterferometerShape.RECTANGLE
    )

    # 2. Input encoding - maps classical data to quantum parameters
    c_var = pcvl.Circuit(n_modes)
    for i in range(n_features):  # 4 input features
        px = pcvl.P(f"px{i + 1}")
        c_var.add(i, pcvl.PS(px))

    # 3. Right interferometer - trainable transformation
    wr = pcvl.GenericInterferometer(
        n_modes,
        lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_ri{i}")) //
                 pcvl.BS() // pcvl.PS(pcvl.P(f"theta_ro{i}")),
        shape=pcvl.InterferometerShape.RECTANGLE
    )

    # Combine all components
    return wl // c_var // wr

def required_input_params(n_modes: int, n_features: int) -> int:
    # Parallel-columns requires K = n_features * n_modes input parameters ("pl*")
    return n_features * n_modes

# ======================= Data / Utils =======================

def make_pca(k: int):
    to_t = transforms.Compose([transforms.ToTensor()])
    base_tr = datasets.MNIST("./data", train=True,  download=True, transform=to_t)
    base_te = datasets.MNIST("./data", train=False, download=True, transform=to_t)

    def filt(base):
        Xs, Ys = [], []
        for img, lab in base:
            if int(lab) in (0, 1):
                Xs.append(img.view(-1).float())
                Ys.append(0 if int(lab) == 0 else 1)
        return torch.stack(Xs, 0), torch.tensor(Ys, dtype=torch.long)

    Xtr, ytr = filt(base_tr)
    Xte, yte = filt(base_te)
    pca = PCA(n_components=k, svd_solver="full", whiten=False, random_state=0)
    Ztr_raw = torch.from_numpy(pca.fit_transform(Xtr.numpy())).float()
    Zte_raw = torch.from_numpy(pca.transform(Xte.numpy())).float()
    mins = Ztr_raw.min(0, keepdim=True).values
    maxs = Ztr_raw.max(0, keepdim=True).values
    Ztr = torch.clamp((Ztr_raw - mins) / (maxs - mins + 1e-8), 0.0, 1.0)
    Zte = torch.clamp((Zte_raw - mins) / (maxs - mins + 1e-8), 0.0, 1.0)
    return (Ztr, ytr), (Zte, yte)

def set_seed(s: int):
    random.seed(s); torch.manual_seed(s)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, angle_factor: float) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        logits = model(xb * angle_factor)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / max(1, total)

###############################
### Naive approach using GI ###
###############################

# ======================= Build Q Layer (parallel-columns only) =======================

def build_single_gi_layer(
    n_modes: int,
    n_features: int,
    n_photons: int = 4,
    reservoir_mode: bool = False,
    state_pattern: str = "default"
):
    #circ = build_parallel_columns_circuit(n_modes, n_features, reservoir_mode)
    circ = create_quantum_circuit(n_modes, n_features)
    #pcvl.pdisplay(circ)
    trainable_prefixes = [] if reservoir_mode else ["theta"]
    input_prefixes     = ["px"]

    print(f"N_modes = {n_modes} and n_photons = {n_photons} and n_features = {n_features}")
    pat = StatePattern[state_pattern.upper()] if state_pattern else StatePattern.DEFAULT
    input_state = StateGenerator.generate_state(n_modes, n_photons, pat)

    return ML.QuantumLayer(
        input_size=n_modes,
        output_size=2,
        circuit=circ,
        trainable_parameters=trainable_prefixes,
        input_parameters=input_prefixes,
        input_state=input_state,
        output_mapping_strategy=ML.OutputMappingStrategy.GROUPING,
    )

# ======================= Model (PCA-8 → K adapter) =======================

class SingleGI(nn.Module):
    def __init__(
        self,
        n_modes,
        n_features,
        n_photons: int,
        reservoir_mode: bool,
        state_pattern: str,
        required_inputs: int,
        adapter: str,
        input_dim: int,
    ):
        super().__init__()
        self.K = required_inputs
        self.adapter = adapter
        self.input_dim = input_dim
        self.q = build_single_gi_layer(
            n_modes, n_features, n_photons=n_photons,
            reservoir_mode=reservoir_mode, state_pattern=state_pattern
        )
        # Optional learnable adapter
        if self.adapter == "linear":
            self.lin = nn.Linear(self.input_dim, self.K)

    def _adapt(self, features: torch.Tensor) -> torch.Tensor:
        """Remap features to the K inputs expected by the quantum layer."""
        if features.dim() == 1:
            features = features.unsqueeze(0)
        current_dim = features.shape[-1]
        if current_dim == self.K and self.adapter not in ("repeat", "linear"):
            return features
        if self.adapter in ("auto", "slice"):
            if current_dim >= self.K:
                return features[..., :self.K]
            return F.pad(features, (0, self.K - current_dim), value=0.0)
        if self.adapter == "zero_pad":
            if current_dim >= self.K:
                return features[..., :self.K]
            return F.pad(features, (0, self.K - current_dim), value=0.0)
        if self.adapter == "repeat":
            if current_dim == 0:
                raise ValueError("Cannot repeat features with zero width.")
            reps = (self.K + current_dim - 1) // current_dim
            if features.dim() == 1:
                expanded = features.repeat(reps)
            elif features.dim() == 2:
                expanded = features.repeat(1, reps)
            else:
                repeat_dims = [1] * (features.dim() - 1) + [reps]
                expanded = features.repeat(*repeat_dims)
            return expanded[..., :self.K]
        if self.adapter == "linear":
            return self.lin(features)
        # Fallback: slice or pad as needed
        if current_dim >= self.K:
            return features[..., :self.K]
        return F.pad(features, (0, self.K - current_dim), value=0.0)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        adapted = self._adapt(angles)  # [..., K]
        return self.q(adapted)

###################################
### Pseudo-convolution approach ###
###################################

class QuantumPatchKernel(nn.Module):
    """Wrap a QuantumLayer so it can process sliding PCA patches."""

    def __init__(
        self,
        q_layer: ML.QuantumLayer,
        required_inputs: int,
        patch_dim: int,
        adapter: str = "auto",
    ):
        super().__init__()
        self.q = q_layer
        self.K = required_inputs
        self.patch_dim = patch_dim
        self.adapter = adapter
        if self.adapter == "linear":
            self.lin = nn.Linear(patch_dim, self.K)

    def _adapt(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 1:
            features = features.unsqueeze(0)
        current_dim = features.shape[-1]
        if current_dim == self.K and self.adapter not in ("repeat", "linear"):
            return features
        if self.adapter in ("auto", "slice"):
            if current_dim >= self.K:
                return features[..., :self.K]
            return F.pad(features, (0, self.K - current_dim), value=0.0)
        if self.adapter == "zero_pad":
            if current_dim >= self.K:
                return features[..., :self.K]
            return F.pad(features, (0, self.K - current_dim), value=0.0)
        if self.adapter == "repeat":
            if current_dim == 0:
                raise ValueError("Cannot repeat features with zero width.")
            reps = (self.K + current_dim - 1) // current_dim
            if features.dim() == 1:
                expanded = features.repeat(reps)
            else:
                expanded = features.repeat(1, reps)
            return expanded[..., :self.K]
        if self.adapter == "linear":
            return self.lin(features)
        if current_dim >= self.K:
            return features[..., :self.K]
        return F.pad(features, (0, self.K - current_dim), value=0.0)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        adapted = self._adapt(patch)
        return self.q(adapted)

    @property
    def output_size(self) -> int:
        if getattr(self.q, "output_size", None) is None:
            raise ValueError("QuantumLayer does not expose output_size attribute.")
        return self.q.output_size

class qconv1d(nn.Module):
    """
    Lightweight 1D convolution on top of PCA components.

    The layer treats each sample as a 1D signal whose length is the number of PCA
    coefficients.  It slides `n_kernels` learnable kernels of length
    `kernel_size` across the feature axis with the provided `stride` and
    produces a tensor of shape ``(batch, n_kernels * n_windows)`` where
    ``n_windows`` is the number of extracted patches.
    """

    def __init__(
        self,
        n_kernels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        flatten: bool = True,
        kernel_modules: list[nn.Module] | None = None,
    ):
        super().__init__()
        if n_kernels <= 0:
            raise ValueError("n_kernels must be a positive integer.")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be a positive integer.")
        if stride <= 0:
            raise ValueError("stride must be a positive integer.")

        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.flatten = flatten
        print(f"The conv1d is made of \n - {n_kernels} kernels of size {kernel_size} with a stride of {stride}")
        self.use_quantum = kernel_modules is not None
        if self.use_quantum:
            if kernel_modules is None or len(kernel_modules) != n_kernels:
                raise ValueError("kernel_modules must contain n_kernels modules.")
            self.kernel_modules = nn.ModuleList(kernel_modules)
            self.kernel_output_dim = getattr(self.kernel_modules[0], "output_size", None)
            if self.kernel_output_dim is None:
                raise ValueError("Quantum kernels must expose an output_size attribute.")
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        else:
            weight = torch.empty(n_kernels, kernel_size)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            self.weight = nn.Parameter(weight)
            self.kernel_modules = None
            self.kernel_output_dim = 1
            if bias:
                bound = 1 / math.sqrt(kernel_size)
                self.bias = nn.Parameter(torch.empty(n_kernels).uniform_(-bound, bound))
            else:
                self.register_parameter("bias", None)

    def _compute_num_windows(self, input_len: int) -> int:
        if input_len < self.kernel_size:
            raise ValueError(
                f"Input length {input_len} is smaller than kernel size {self.kernel_size}."
            )
        # Mirrors torch.nn.Conv1d behaviour (floor division).
        return 1 + (input_len - self.kernel_size) // self.stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(
                "qconv1d expects inputs of shape (batch, n_components) "
                "or (n_components,) optionally with a singleton channel dimension."
            )

        patches = x.unfold(dimension=-1, size=self.kernel_size, step=self.stride)
        num_windows = patches.size(1)
        if num_windows == 0:
            raise ValueError(
                "Unfold produced zero windows. Check kernel_size and stride values."
            )

        if self.use_quantum:
            patches_flat = patches.contiguous().view(-1, patches.size(-1))
            outputs = []
            for kernel in self.kernel_modules:
                y = kernel(patches_flat)
                if y.dim() == 1:
                    y = y.unsqueeze(-1)
                y = y.view(x.size(0), num_windows, -1)
                outputs.append(y)
            out = torch.stack(outputs, dim=1)  # (batch, n_kernels, num_windows, out_dim)
            if self.flatten:
                return out.view(out.size(0), -1)
            return out

        out = torch.einsum("bpk,nk->bpn", patches, self.weight)
        if self.bias is not None:
            out = out + self.bias.view(1, 1, -1)
        out = out.permute(0, 2, 1).contiguous()

        if self.flatten:
            return out.view(out.size(0), -1)
        return out

    def output_dim(self, input_len: int) -> int:
        """Number of features produced for a given PCA dimension."""
        num_windows = self._compute_num_windows(input_len)
        return num_windows * self.n_kernels * self.kernel_output_dim


class QConvModel(nn.Module):
    """Apply quantum convolutional kernels over PCA components followed by a linear head."""

    def __init__(self, conv_layer: qconv1d, input_dim: int):
        super().__init__()
        self.conv = conv_layer
        self.input_dim = input_dim
        self.conv_output_dim = self.conv.output_dim(input_dim)
        if not getattr(self.conv, "flatten", True):
            raise ValueError("qconv1d layer must output flattened features.")
        self.head = nn.Linear(self.conv_output_dim, 2)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        features = self.conv(angles)
        return self.head(features)


# ======================= Train / Eval =======================

def train_once(Ztr, ytr, Zte, yte,
               n_modes, n_features, steps: int, batch: int,
               opt_name: str, lr: float, momentum: float,
               angle_factor: float,
               n_photons: int, reservoir_mode: bool, state_pattern: str,
               required_inputs: int, adapter: str,
               seed: int,
               model_type: str,
               conv_params: dict | None = None) -> float:
    set_seed(seed)

    input_dim = Ztr.shape[-1]
    if model_type == "single":
        model = SingleGI(
            n_modes=n_modes, n_features=n_features,
            n_photons=n_photons, reservoir_mode=reservoir_mode, state_pattern=state_pattern,
            required_inputs=required_inputs, adapter=adapter,
            input_dim=input_dim,
        )
    elif model_type == "qconv":
        if conv_params is None:
            raise ValueError("conv_params must be provided for the qconv model.")
        params = dict(conv_params)
        factory = params.pop("kernel_modules_factory", None)
        if factory is not None:
            params["kernel_modules"] = factory()
        conv_layer = qconv1d(**params)
        model = QConvModel(conv_layer=conv_layer, input_dim=input_dim)
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Expected 'single' or 'qconv'.")
    if opt_name == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    lossf = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(Ztr, ytr), batch_size=batch, shuffle=True, drop_last=True)
    test_loader  = DataLoader(TensorDataset(Zte, yte), batch_size=512, shuffle=False)

    model.train()
    it = 0
    while it < steps:
        for xb, yb in train_loader:
            optim.zero_grad(set_to_none=True)
            logits = model(xb * angle_factor)
            loss = lossf(logits, yb)
            loss.backward()
            optim.step()
            it += 1
            if it >= steps:
                break

    return evaluate(model, test_loader, angle_factor)

# ======================= CLI =======================

def main():
    ap = argparse.ArgumentParser(description="Parallel-columns GI on MNIST 0vs1 (PCA-8 everywhere)")
    ap.add_argument("--steps", type=int, default=200, help="optimizer steps (not epochs)")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--opt", choices=["adam","sgd"], default="adam")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.9)

    # Angle scaling
    ap.add_argument("--angle_scale", choices=["none","pi","2pi"], default="none",
                    help="map [0,1] → unchanged, [0,π], or [0,2π]")

    ap.add_argument("--n_modes", type=int, default=8)
    ap.add_argument("--n_features", type=int, default=8, help="feature count used by the circuit topology")
    ap.add_argument("--reservoir_mode", action="store_true", help="freeze interferometer params (randomized)")
    ap.add_argument("--state_pattern", choices=["default","spaced","sequential","periodic"], default="default")
    ap.add_argument("--n_photons", type=int, default=4)

    # Always PCA-8 by default
    ap.add_argument("--pca_dim", type=int, default=8)

    ap.add_argument("--model", choices=["qconv", "single"], default="qconv",
                    help="choose between quantum convolution (default) or single GI model")

    # Adapter from 8 → K
    ap.add_argument("--adapter", choices=["auto","slice","zero_pad","repeat","linear"], default="auto",
                    help="auto: slice if K≤8 else zero-pad; linear adds a learnable 8→K map")
    ap.add_argument("--qconv_kernels", type=int, default=4,
                    help="number of kernels for the pseudo-convolution")
    ap.add_argument("--qconv_kernel_size", type=int, default=2,
                    help="kernel size for the pseudo-convolution")
    ap.add_argument("--qconv_stride", type=int, default=1,
                    help="stride for the pseudo-convolution")
    ap.add_argument("--qconv_classical", action="store_true",
                    help="use classical learnable kernels instead of QuantumLayer kernels")
    ap.add_argument("--qconv_adapter", choices=["auto","slice","zero_pad","repeat","linear"], default="auto",
                    help="adapter strategy from PCA patches to quantum kernel inputs")
    ap.add_argument("--qconv_kernel_modes", type=int, default=8,
                    help="number of optical modes per quantum kernel (default: qconv_kernel_size)")
    ap.add_argument("--qconv_kernel_features", type=int, default=2,
                    help="feature depth (stages) for quantum kernels")

    args = ap.parse_args()

    # Angle scale factor
    angle_factor = 1.0 if args.angle_scale == "none" else (math.pi if args.angle_scale == "pi" else 2 * math.pi)

    # Required K for parallel-columns
    K = required_input_params(args.n_modes, args.n_features)

    conv_params = None
    conv_output_dim = None
    conv_mode_str = "disabled"
    if args.model == "qconv":
        if args.qconv_kernel_size > args.pca_dim:
            raise ValueError("qconv kernel_size cannot exceed the PCA dimension.")
        if args.qconv_stride <= 0:
            raise ValueError("qconv stride must be a positive integer.")
        num_windows = 1 + (args.pca_dim - args.qconv_kernel_size) // args.qconv_stride
        if num_windows <= 0:
            raise ValueError("qconv configuration results in zero sliding windows.")
        use_quantum = not args.qconv_classical
        conv_params = {
            "n_kernels": args.qconv_kernels,
            "kernel_size": args.qconv_kernel_size,
            "stride": args.qconv_stride,
            "bias": args.qconv_classical,
            "flatten": True,
        }
        conv_output_dim = num_windows * args.qconv_kernels
        conv_mode_str = "classical"
        if use_quantum:
            kernel_modes = args.qconv_kernel_modes or args.qconv_kernel_size
            if kernel_modes <= 0:
                raise ValueError("qconv_kernel_modes must be a positive integer.")
            if args.qconv_kernel_features <= 0:
                raise ValueError("qconv_kernel_features must be a positive integer.")
            kernel_required_inputs = required_input_params(kernel_modes, args.qconv_kernel_features)
            print(f"Building a convolution with kernels on {kernel_modes} modes to catch {args.qconv_kernel_features} features")
            def _make_kernel() -> QuantumPatchKernel:
                """q_layer = build_single_gi_layer(
                    n_modes=kernel_modes,
                    n_features=args.qconv_kernel_features,
                    n_photons=args.n_photons,
                    reservoir_mode=args.reservoir_mode,
                    state_pattern=args.state_pattern,
                )"""
                q_layer = ML.QuantumLayer(
                        input_size=kernel_modes,
                        output_size=2,
                        circuit=create_quantum_circuit(kernel_modes, args.qconv_kernel_features),
                        trainable_parameters=["theta"],
                        input_parameters=["px"],
                        input_state=[1,0,1,0,1,0,1,0],
                        output_mapping_strategy=ML.OutputMappingStrategy.GROUPING,
                    )
                return QuantumPatchKernel(
                    q_layer,
                    required_inputs=kernel_required_inputs,
                    patch_dim=args.qconv_kernel_size,
                    adapter=args.qconv_adapter,
                )

            sample_kernel = _make_kernel()
            sample_out_dim = sample_kernel.output_size
            del sample_kernel

            def _factory():
                return [_make_kernel() for _ in range(args.qconv_kernels)]

            conv_params["bias"] = False
            conv_params["kernel_modules_factory"] = _factory
            conv_output_dim = num_windows * args.qconv_kernels * sample_out_dim
            conv_mode_str = f"quantum(out={sample_out_dim})"

    # Load PCA-8 (or user-override)
    (Ztr, ytr), (Zte, yte) = make_pca(args.pca_dim)
    print(f"PCA-{args.pca_dim} ready: train {Ztr.shape}, test {Zte.shape} | angle={args.angle_scale}")
    if args.model == "qconv":
        conv_str = (f"{conv_params['n_kernels']}x{conv_params['kernel_size']} stride={conv_params['stride']} "
                    f"| mode={conv_mode_str} -> {conv_output_dim} dims")
    else:
        conv_str = "n/a (single GI)"
    print(f"Model: {args.model} | circuit=parallel_columns | n_modes={args.n_modes} n_features={args.n_features} "
          f"| required inputs K={K} | adapter={args.adapter} | reservoir={args.reservoir_mode} | conv={conv_str}")

    accs = []
    for s in range(args.seeds):
        print(f"[Seed {s+1}/{args.seeds}]")
        acc = train_once(
            Ztr, ytr, Zte, yte,
            args.n_modes, args.n_features,
            steps=args.steps, batch=args.batch,
            opt_name=args.opt, lr=args.lr, momentum=args.momentum,
            angle_factor=angle_factor,
            n_photons=args.n_photons, reservoir_mode=args.reservoir_mode, state_pattern=args.state_pattern,
            required_inputs=K, adapter=args.adapter,
            seed=1235 + s,
            model_type=args.model,
            conv_params=conv_params
        )
        print(f"  Test accuracy: {acc*100:.2f}%")
        accs.append(acc)

    mean = statistics.mean(accs)
    std = statistics.stdev(accs) if len(accs) > 1 else 0.0
    print("\n=== Summary ===")
    print("Accuracies:", ", ".join(f"{a*100:.2f}%" for a in accs))
    print(f"Mean ± Std: {mean*100:.2f}% ± {std*100:.2f}%")

if __name__ == "__main__":
    main()
