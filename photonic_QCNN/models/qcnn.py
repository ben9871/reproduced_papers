import io
import math
import random
import re
import sys
from collections.abc import Generator
from typing import Union

import numpy as np
import sympy as sp
import torch
from merlin import CircuitConverter
from merlin import build_slos_distribution_computegraph as build_slos_graph
from perceval import (
    BS,
    PS,
    Circuit,
    GenericInterferometer,
    InterferometerShape,
    Matrix,
    P,
    Unitary,
    catalog,
)
from torch import Tensor, nn


def get_circuit(m: int, type: str = "MZI"):
    """
    Generate different types of quantum circuits for photonic quantum computing.

    Args:
        m: Number of modes in the circuit
        type: Type of circuit to generate. Options are:
            - 'MZI': Mach-Zehnder interferometer with phase shifts
            - 'BS': Beam splitter based interferometer
            - 'BS_random_PS': Beam splitter with random phase shifts
            - 'paper_6modes': Specific 6-mode circuit from the paper

    Returns:
        Circuit: The generated quantum circuit

    Raises:
        ValueError: If circuit type is not recognized
    """
    if type == "MZI":
        return GenericInterferometer(m, catalog["mzi phase first"].generate)
    elif type == "BS":
        return GenericInterferometer(
            m,
            lambda i: BS.Ry(theta=-2 * P(f"phi_{i}")),
            shape=InterferometerShape.TRIANGLE,
        )
    elif type == "BS_random_PS":
        return GenericInterferometer(
            m,
            lambda i: BS.Ry(theta=-2 * P(f"phi_{i}"))
            // PS(phi=2 * np.pi * random.random()),
            shape=InterferometerShape.TRIANGLE,
        )
    elif type == "paper_6modes":
        c = Circuit(6)
        c.add(2, BS.Ry(theta=-2 * P("phi_0")))
        c.add(1, BS.Ry(theta=-2 * P("phi_1")))
        c.add(3, BS.Ry(theta=-2 * P("phi_2")))
        c.add(0, BS.Ry(theta=-2 * P("phi_3")))
        c.add(2, BS.Ry(theta=-2 * P("phi_4")))
        c.add(4, BS.Ry(theta=-2 * P("phi_5")))
        c.add(1, BS.Ry(theta=-2 * P("phi_6")))
        c.add(3, BS.Ry(theta=-2 * P("phi_7")))
        return c
    else:
        raise ValueError(f"Unknown circuit type {type}")


def generate_all_fock_states(m, n) -> Generator:
    """
    Generate all possible Fock states for n photons and m modes.

    Args:
        m: Number of modes
        n: Number of photons

    Returns:
        Generator of tuples of each Fock state.
    """
    if n == 0:
        yield (0,) * m
        return
    if m == 1:
        yield (n,)
        return

    for i in range(n + 1):
        for state in generate_all_fock_states(m - 1, n - i):
            yield (i,) + state


class OneHotEncoder(nn.Module):
    def __init__(self):
        """
        One Hot Encoder

        Converts an image X to density matrix in the One Hot Amplitude
        basis. For a given d x d image, the density matrix will be of
        size d^2 x d^2.
        """
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        norm = torch.sqrt(torch.square(torch.abs(x)).sum(dim=(1, 2, 3)))
        x = x / norm.view(-1, 1, 1, 1)

        # Flatten each image and multiply by transpose to get density matrix
        x_flat = x.reshape(x.shape[0], -1)
        rho = x_flat.unsqueeze(2) @ x_flat.unsqueeze(1)
        rho = rho.to(torch.complex64)

        return rho

    def __repr__(self):
        return "OneHotEncoder()"


class OneHotDecoder(torch.nn.Module):
    """
    Measures a state rho in the one hot basis and re-constructs the
    image.

    Args:
        channels (int): Number of channels within the encoding.
    """

    def __init__(self, channels=None):
        super().__init__()
        self.channels = channels

    def forward(self, rho):
        b = len(rho)
        c = 1 if self.channels is None else self.channels

        dim = int(np.sqrt(len(rho[0]) // c))

        diagonals = rho.diagonal(dim1=1, dim2=2)
        diagonals = torch.abs(diagonals)

        x = diagonals.reshape(b, c, dim, dim)

        norm = torch.sqrt(torch.square(torch.abs(x)).sum(dim=(1, 2, 3)))
        norm = norm.view(-1, 1, 1, 1)
        return x

    def __repr__(self):
        c = 0 if self.channels is None else self.channels
        return f"OneHotDecoder(channels={c})"


class AParametrizedLayer(nn.Module):
    """
    Abstract parametrized layer.

    Base class layer for inheriting functionality methods.

    Args:
        dims (tuple): Input dimensions into a parametrized layer.
    """

    def __init__(self, dims: tuple[int]):
        super().__init__()
        self.dims = dims
        self._training_params = []

        if dims[0] != dims[1]:
            raise NotImplementedError("Non-square images not supported yet.")

    def _set_param_names(self, circuit):
        """
        Ensures that two different parametrized circuits have different
        perceval parameter names.
        """
        param_list = list(circuit.get_parameters())

        if not self._training_params:
            param_start_idx = 0
        else:
            # Take index from last parameter name
            param_start_idx = int(
                re.search(r"\d+", self._training_params[-1].name).group()
            )

        for i, p in enumerate(param_list):
            p.name = f"phi{i + param_start_idx + 1}"

        for _, comp in circuit:
            if hasattr(comp, "_phi"):
                param = comp.get_parameters()[0]
                param._symbol = sp.S(param.name)

        self._training_params.extend(param_list)

    def _modify_channels(self, rho, u, uc):
        """
        Prepares shapes for rho, main register U, channel unitary U
        """
        b = rho.shape[0]
        c_out = self.out_channels
        c_in = self.in_channels

        # Evaluate unitary on channels register
        if c_out is not None:
            uc = self._circuit_graph_c.to_tensor(self.phi[-self._n_params_c :])

        # No channel register
        if c_in is None and c_out is None:
            pass

        # Channel register but apply no operations to channels
        elif c_in is not None and c_out is None:
            i = torch.eye(c_in)
            u = torch.kron(i, u)

        # Create channel register
        elif c_in is None and c_out is not None:
            i_rho = torch.zeros((c_out, c_out))
            i_rho[0][0] = 1.0
            i_rho = i_rho.expand(b, -1, -1)

            rho = torch.vmap(torch.kron)(i_rho, rho)

        # Expand the size of the channel register.
        elif c_out > c_in:
            rho = self._mode_insertion(rho)

        # Apply channels to unchanged size register
        elif c_out == c_in:
            pass

        # Apply channels unitary to subset of channels - Transpose convolutions only.
        elif c_out < c_in:
            if c_in - c_out == 1:
                uc = nn.functional.pad(uc, (0, 1, 0, 1))
                uc[c_out][c_out] = 1.0
            else:
                uc = torch.kron(uc, torch.eye(c_in - c_out))
        else:
            raise NotImplementedError("I might have forgotten something.")

        return rho, u, uc


class QConv2d(AParametrizedLayer):
    def __init__(
        self,
        dims,
        kernel_size: int,
        stride: int = None,
        in_channels: int = None,
        out_channels: int = None,
        circuit: str = "MZI",
    ):
        """
        Quantum 2D Convolutional layer to processor

        Args:
            dims: Input dimensions.
            kernel_size: Size of universal interferometer
            stride: Stride of the universal interferometer across the
                modes.
        """
        if in_channels is not None and out_channels is None:
            raise ValueError("Please specify a number of output channels")

        super().__init__(dims)
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Define filters
        filters = []
        for _ in range(2):
            filter = get_circuit(kernel_size, circuit)
            self._set_param_names(filter)
            filters.append(filter)

        # Create x and y and channel registers
        self._reg_x = Circuit(dims[0], name="Conv X")
        self._reg_y = Circuit(dims[1], name="Conv Y")

        # Add filters with specified stride
        for i in range((dims[0] - kernel_size) // self.stride + 1):
            self._reg_x.add(self.stride * i, filters[0])

        for i in range((dims[1] - kernel_size) // self.stride + 1):
            self._reg_y.add(self.stride * i, filters[1])

        self._n_params_x = len(self._reg_x.get_parameters())
        self._n_params_y = len(self._reg_y.get_parameters())
        num_params = self._n_params_x + self._n_params_y

        if out_channels is not None:
            self._reg_c = get_circuit(out_channels, circuit)
            self._set_param_names(self._reg_c)
            self._n_params_c = len(self._reg_c.get_parameters())
            num_params += self._n_params_c

        # Suppress unnecessary print statements from pcvl_pytorch
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            # Build circuit graphs for the two registers separately.
            self._circuit_graph_x = CircuitConverter(
                self._reg_x, ["phi"], torch.float32
            )
            self._circuit_graph_y = CircuitConverter(
                self._reg_y, ["phi"], torch.float32
            )
            if out_channels is not None:
                self._circuit_graph_c = CircuitConverter(
                    self._reg_c, ["phi"], torch.float32
                )
        finally:
            sys.stdout = original_stdout

        # Create model parameters
        self.phi = nn.Parameter(2 * np.pi * torch.rand(num_params))

        if out_channels is not None and in_channels is not None:
            if in_channels < out_channels:
                self._mode_insertion = _InsertChannels(dims, in_channels, out_channels)

    def forward(self, rho, adjoint=False):
        b = len(rho)

        phi_x = self.phi[: self._n_params_x]
        phi_y = self.phi[self._n_params_x : self._n_params_x + self._n_params_y]

        ux = self._circuit_graph_x.to_tensor(phi_x)
        uy = self._circuit_graph_y.to_tensor(phi_y)
        u = torch.kron(ux, uy)

        # Evaluate unitary on channels register
        if self.out_channels is not None:
            uc = self._circuit_graph_c.to_tensor(self.phi[-self._n_params_c :])
        else:
            uc = None

        rho, u, uc = self._modify_channels(rho, u, uc)

        if self.out_channels is not None:
            u = torch.kron(uc, u)

        u = u.unsqueeze(0).expand(b, -1, -1)
        udag = u.transpose(1, 2).conj()

        # There is only one photon in each register, can apply the U directly.
        if not adjoint:
            u_rho = torch.bmm(u, rho)
            new_rho = torch.bmm(u_rho, udag)
        else:
            # Apply adjoint to rho
            udag_rho = torch.bmm(udag, rho)
            new_rho = torch.bmm(udag_rho, u)

        return new_rho

    def __repr__(self):
        c_in = 0 if self.in_channels is None else self.in_channels
        c_out = 0 if self.out_channels is None else self.out_channels
        return f"QConv2d({self.dims}, kernel_size={self.kernel_size}, in_channels={c_in}, out_channels={c_out})"


class QCycleConv2d(AParametrizedLayer):
    def __init__(self, dims, in_channels=None, out_channels=None):
        """
        Quantum 2D Cyclic Convolutional layer

        :param dims: Dimension of input image.
        """
        super().__init__()
        self.dims = dims
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.m = sum(dims)

        # Define the registers
        self._build_registers()

        num_params = self.m
        if out_channels is not None:
            self._reg_c = GenericInterferometer(
                out_channels, catalog["mzi phase first"].generate
            )
            self._n_params_c = len(self._reg_c.get_parameters())
            num_params += self._n_params_c

        # Suppress unnecessary print statements
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            # Build circuit graphs for the two registers separately.
            self._circuit_graph_x = CircuitConverter(
                self._reg_x, ["phi"], torch.float32
            )
            self._circuit_graph_y = CircuitConverter(
                self._reg_y, ["phi"], torch.float32
            )
            if out_channels is not None:
                self._circuit_graph_c = CircuitConverter(
                    self._reg_c, ["phi"], torch.float32
                )
        finally:
            sys.stdout = original_stdout

        # Create and register model parameters
        self.phi = torch.nn.Parameter(2 * np.pi * torch.rand(num_params))

        if out_channels is not None and in_channels is not None:
            if out_channels > in_channels:
                self._mode_insertion = _InsertChannels(dims, in_channels, out_channels)

    def forward(self, rho, adjoint=False):
        b = len(rho)

        phi_x = self.phi[: self.m // 2]
        phi_y = self.phi[self.m // 2 : self.m]

        ux = self._circuit_graph_x.compute(phi_x)
        uy = self._circuit_graph_y.compute(phi_y)
        u = torch.kron(ux, uy)

        # Evaluate unitary on channels register
        if self.out_channels is not None:
            phi_c = self.phi[-self._n_params_c :]
            uc = self._circuit_graph_c.compute(phi_c)
        else:
            uc = None

        rho, u, uc = self._modify_channels(rho, u, uc)

        if self.out_channels is not None:
            u = torch.kron(uc, u)

        u = u.unsqueeze(0).expand(b, -1, -1)
        udag = u.transpose(1, 2).conj()

        # Since there is only one photon in each register, can apply the U directly.
        if not adjoint:
            u_rho = torch.bmm(u, rho)
            new_rho = torch.bmm(u_rho, udag)
        else:
            # Apply adjoint to rho
            udag_rho = torch.bmm(udag, rho)
            new_rho = torch.bmm(udag_rho, u)

        return new_rho

    def _build_registers(
        self,
    ) -> Circuit:
        """Build a register for a given size."""
        m_reg = self.m // 2

        # Build QFT interferometer unitaries.
        k = np.arange(m_reg).reshape((1, m_reg))
        l = np.arange(m_reg).reshape((m_reg, 1))  # noqa: E741
        exponent = -2j * np.pi * k * l / m_reg
        matrix = np.exp(exponent) / np.sqrt(m_reg)
        matrix_dag = matrix.T.conj()

        uft = Unitary(Matrix(matrix))
        uft_dag = Unitary(Matrix(matrix_dag))

        # Build x register
        self._reg_x = Circuit(m_reg, name="QCycleConv2d").add(0, uft)
        for mode in range(m_reg):
            self._reg_x.add(mode, PS(P(f"phi{mode}")))
        self._reg_x.add(0, uft_dag)

        # Build y register
        self._reg_y = Circuit(m_reg, name="QCycleConv2d").add(0, uft)
        for mode in range(m_reg):
            self._reg_x.add(mode, PS(P(f"phi{mode + m_reg}")))
        self._reg_x.add(0, uft_dag)

    def __repr__(self):
        c_in = 0 if self.in_channels is None else self.in_channels
        c_out = 0 if self.out_channels is None else self.out_channels

        return f"QCycleConv2d({self.dims}, in_channels={c_in}, out_channels={c_out})"


class QPooling(torch.nn.Module):
    """
    Quantum pooling layer

    :param dims: Input image dimensions
    :param kernel_size: Dimension by which the image is reduced.
    :param channels: Number of input channels.
    """

    def __init__(self, dims, kernel_size, channels=None):
        d = dims[0]
        k = kernel_size
        c = 1 if channels is None else channels
        new_d = d // kernel_size

        super().__init__()
        self.dims = d
        self._new_d = new_d
        self.kernel_size = k
        self.channels = channels

        # Create all index combinations at once
        x = torch.arange(c * d**2)
        y = torch.arange(c * d**2)

        # Let f, h represent the channel indices.
        # In basis: |e_f>|e_i>|e_j><e_h|<e_l|<e_m|
        f = x // (d**2)
        h = y // (d**2)

        # Let i, j, l, m represent the one hot indices
        i = (x % (d**2)) // d
        j = (x % (d**2)) % d
        l = (y % (d**2)) // d  # noqa: E741
        m = (y % (d**2)) % d

        # Create meshgrids for pairwise operations
        f_grid, h_grid = torch.meshgrid(f, h, indexing="ij")
        i_grid, l_grid = torch.meshgrid(i, l, indexing="ij")
        j_grid, m_grid = torch.meshgrid(j, m, indexing="ij")

        # Ensure that odd mode photon numbers match
        inject_condition = (i_grid % k == l_grid % k) & (j_grid % k == m_grid % k)
        match_odd1 = ((i_grid % k != 0) & (i_grid == l_grid)) | (i_grid % k == 0)
        match_odd2 = ((j_grid % k != 0) & (j_grid == m_grid)) | (j_grid % k == 0)

        mask = inject_condition & match_odd1 & match_odd2
        mask_coords = torch.nonzero(mask, as_tuple=False)
        self._mask_coords = (mask_coords[:, 0], mask_coords[:, 1])

        new_i = i_grid[mask] // k
        new_j = j_grid[mask] // k
        new_l = l_grid[mask] // k
        new_m = m_grid[mask] // k

        self._new_x = new_i * new_d + new_j + f_grid[mask] * new_d**2
        self._new_y = new_l * new_d + new_m + h_grid[mask] * new_d**2

    def forward(self, rho):
        b = len(rho)
        c = 1 if self.channels is None else self.channels

        b_indices = torch.arange(b).unsqueeze(1).expand(-1, len(self._new_x))
        b_indices = b_indices.reshape(-1)

        new_x = self._new_x.expand(b, -1).reshape(-1)
        new_y = self._new_y.expand(b, -1).reshape(-1)

        new_rho = torch.zeros(
            b,
            c * self._new_d**2,
            c * self._new_d**2,
            dtype=rho.dtype,
            device=rho.device,
        )

        values = rho[:, self._mask_coords[0], self._mask_coords[1]].reshape(-1)
        new_rho.index_put_((b_indices, new_x, new_y), values, accumulate=True)

        return new_rho


# For QDense layer, patch merlin.SLOSGraph to add method to return amplitudes
def compute_amplitudes(self, unitary: Tensor, input_state: list[int]) -> torch.Tensor:
    """
    Compute the amplitudes using the pre-built graph.

    Args:
        unitary (torch.Tensor): Single unitary matrix [m x m] or batch
            of unitaries [b x m x m]. The unitary should be provided in
            the complex dtype corresponding to the graph's dtype.
            For example: for torch.float32, use torch.cfloat;
            for torch.float64, use torch.cdouble.
        input_state (list[int]): Input_state of length self.m with
            self.n_photons in the input state

    Returns:
        Tensor: Output amplitudes associated with each Fock state.
    """
    # Add batch dimension
    if len(unitary.shape) == 2:
        unitary = unitary.unsqueeze(0)

    if any(n < 0 for n in input_state) or sum(input_state) == 0:
        raise ValueError("Photon numbers cannot be negative or all zeros")

    if self.no_bunching and not all(x in [0, 1] for x in input_state):
        raise ValueError(
            "Input state must be binary (0s and 1s only) in non-bunching mode"
        )

    batch_size, m, m2 = unitary.shape
    if m != m2 or m != self.m:
        raise ValueError(
            f"Unitary matrix must be square with dimension {self.m}x{self.m}"
        )

    # Check dtype to match the complex dtype used for the graph building
    if unitary.dtype != self.complex_dtype:
        raise ValueError(
            f"Unitary dtype {unitary.dtype} doesn't match the expected complex"
            f" dtype {self.complex_dtype} for the graph built with dtype"
            f" {self.dtype}. Please provide a unitary with the correct dtype "
            f"or rebuild the graph with a compatible dtype."
        )
    idx_n = []
    self.norm_factor_input = 1
    for i, count in enumerate(input_state):
        for c in range(count):
            self.norm_factor_input *= c + 1
            idx_n.append(i)

            bounds1 = self.index_photons[len(idx_n) - 1][1]
            bounds2 = self.index_photons[len(idx_n) - 1][0]
            if (i > bounds1) or (i < bounds2):
                raise ValueError(
                    f"Input state photons must be bounded by {self.index_photons}"
                )

    # Get device from unitary
    device = unitary.device

    # Initial amplitude
    amplitudes = torch.ones((batch_size, 1), dtype=self.complex_dtype, device=device)

    # Apply each layer
    for layer_idx, layer_fn in enumerate(self.layer_functions):
        p = idx_n[layer_idx]
        amplitudes, self.contributions = layer_fn(
            unitary, amplitudes, p, return_contributions=True
        )

    self.prev_amplitudes = amplitudes

    # Normalize the amplitudes
    self.norm_factor_output = self.norm_factor_output.to(device=device)
    amplitudes = amplitudes * torch.sqrt(self.norm_factor_output)
    amplitudes = amplitudes / math.sqrt(self.norm_factor_input)

    return amplitudes


class QDense(AParametrizedLayer):
    """
    Dense layer

    Expects an input density matrix in the One Hot Amplitude basis and
    performs SLOS to return the output density matrix in the whole fock
    space.

    Args:
        dims: Input image dimensions.
        sizes: Size of the dense layers placed in succession.
        channels: Size of the channel register received into dense layer.

    """

    def __init__(
        self,
        dims,
        m: Union[int, list] = None,
        channels: int = None,
        circuit: str = "MZI",
        add_modes: int = 0,
        device=None,
    ):
        super().__init__(dims)

        self.dims = dims
        self.channels = channels
        self.add_modes = add_modes

        if add_modes % 2 == 0 and add_modes > 0:
            insertion_x = [-i for i in range(int(add_modes / 2))]
            insertion_y = [i - 1 + dims[0] * 2 for i in range(int(add_modes / 2))]
            insertion = insertion_x + insertion_y
            self.add_modes_layer = _InsertMainModes(dims, insertion)
            self.dims = (dims[0] + len(insertion_x), dims[1] + len(insertion_y))
        elif add_modes != 0:
            raise NotImplementedError("Asymmetric insertions not supported yet.")

        self.device = device

        n = 2 if channels is None else 3
        m = m if m is not None else sum(self.dims)
        if channels is None:
            self.m = [m]
        else:
            self.m = [m + channels]

        # Construct circuit and circuit graph
        self._training_params = []

        self.circuit = Circuit(max(self.m))
        for m in self.m:
            # Implement the same circuit from the paper
            if circuit == "BS" and add_modes == 2 and m == 6:
                circuit = "paper_6modes"
            gi = get_circuit(m, circuit)
            self._set_param_names(gi)
            self.circuit.add(0, gi)

        # Suppress unnecessary print statements
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            self._circuit_graph = CircuitConverter(self.circuit, ["phi"], torch.float32)
        finally:
            sys.stdout = original_stdout

        # Set up input states & SLOS graphs
        self._input_states = [
            tuple(int(i == x) for i in range(self.dims[0]))
            + tuple(int(i == y) for i in range(self.dims[1]))
            for x in range(self.dims[1])
            for y in range(self.dims[0])
        ]

        if channels is not None:
            channel_states = list(generate_all_fock_states(channels, 1))

            self._input_states = [
                c + m for c in channel_states for m in self._input_states
            ]

        self._slos_graph = build_slos_graph(
            m=max(self.m),
            n_photons=n,
            device=self.device,
        )
        self._slos_graph.__class__.compute_amplitudes = compute_amplitudes

        # Create and register model parameters
        num_params = len(self._training_params)
        self.phi = nn.Parameter(2 * np.pi * torch.rand(num_params))

    def forward(self, rho):
        # Add modes
        if self.add_modes == 2:
            rho = self.add_modes_layer(rho)

        b = len(rho)

        # Run SLOS & extract amplitudes
        unitary = self._circuit_graph.to_tensor(self.phi)

        amplitudes = torch.stack(
            [
                self._slos_graph.compute_amplitudes(unitary, basis_state)
                for basis_state in self._input_states
            ]
        ).squeeze()

        u_evolve = amplitudes.transpose(0, 1)

        # Amplitudes constitute evolution operator
        u_evolve = u_evolve.expand(b, -1, -1)
        u_evolve_dag = u_evolve.transpose(1, 2).conj()

        # Extract upper triangular & divide diagonal by 2
        upper_rho = torch.triu(rho)
        diagonal_mask = torch.eye(rho.size(-1), dtype=torch.bool)
        upper_rho[..., diagonal_mask] /= 2

        # U rho U dagger for hermitian rho
        inter_rho1 = torch.bmm(u_evolve, upper_rho)
        inter_rho = torch.bmm(inter_rho1, u_evolve_dag)

        new_rho = inter_rho + inter_rho.transpose(1, 2).conj()
        return new_rho

    def __repr__(self):
        c = 0 if self.channels is None else self.channels
        m = self.m[0] if len(self.m) == 1 else self.m

        return f"QDense({self.dims}, m={m}, channels={c})"


class Measure(nn.Module):
    """
    Measurement operator.

    Assumes input is written in Fock basis and extracts diagonal.

    If one would like to perform a partial measurement, the following
    params can be specified.

    Args:
        m: Total number of modes in-device. Default: None
        n: Number of photons in-device. Default: 2.
        subset: Number of modes being measured. Default: None.
    """

    def __init__(self, m: int = None, n: int = 2, subset: int = None):
        super().__init__()
        self.m = m
        self.subset = subset

        if subset is not None:
            all_states = list(generate_all_fock_states(m, 2))
            reduced_states = []
            for i in range(max(0, subset - m + n), n + 1):
                reduced_states += list(generate_all_fock_states(subset, i))
            self.reduced_states_len = len(reduced_states)

            # To reproduce paper, measure from center if 6 modes and measure from start otherwise
            if m == 6:
                self.indices = torch.tensor(
                    [
                        reduced_states.index(state[2 : 2 + subset])
                        for state in all_states
                    ]
                )
            else:
                self.indices = torch.tensor(
                    [reduced_states.index(state[:subset]) for state in all_states]
                )

    def forward(self, rho):
        b = len(rho)
        probs = torch.abs(rho.diagonal(dim1=1, dim2=2))

        if self.subset is not None:
            indices = self.indices.unsqueeze(0).expand(b, -1)
            # Change size of the QCNN output accordingly
            probs_output = torch.zeros(
                (b, self.reduced_states_len), device=probs.device, dtype=probs.dtype
            )
            probs_output.scatter_add_(dim=1, index=indices, src=probs)
            return probs_output

        return probs

    def __repr__(self):
        if self.subset is not None:
            return f"Measure(m={self.m}, n={self.n}, subset={self.subset})"
        else:
            return "Measure"


class QTransposeConv2d(AParametrizedLayer):
    """
    Quantum Transpose Convolutional layer.

    :param dims: Input image dimensions.
    :param kernel_size: Size of universal interferometer and number of
        modes injected.
    :param qconvolution: State is passed through the adjoint of the
        given qconvolution.
    :param scatter_first: If True, trainable beam-splitters will be used
        to mix the modes a little before convolution.
    """

    def __init__(
        self,
        dims,
        kernel_size: int,
        qconv: QConv2d,
        in_channels=None,
        scatter_first=False,
    ):
        super().__init__()
        self.in_dims = dims
        self.kernel_size = kernel_size
        self.qconv = qconv
        self.in_channels = in_channels
        self.scatter_first = scatter_first

        out_dim_x = dims[0] + (kernel_size - 1) * dims[0]
        out_dim_y = dims[1] + (kernel_size - 1) * dims[1]

        self.out_dims = (out_dim_x, out_dim_y)

        self._training_params = []

        # Pad the image to desired dimensions
        insertions = [i for i in range(sum(dims)) for _ in range(kernel_size - 1)]
        self._mode_insertions = _InsertMainModes(
            dims, insertions, channels=self.in_channels
        )

        if scatter_first:
            self._build_scatter_graphs()

        self._new_conv = QConv2d(
            qconv.dims,
            qconv.kernel_size,
            in_channels=in_channels,
            out_channels=qconv.in_channels,
        )

    def forward(self, rho):
        # Force TQConv params match QConv params
        self._new_conv.phi = self.qconv.phi

        rho = self._mode_insertions(rho)

        if self.scatter_first:
            u_scatter_x = self._scatter_graph_x.to_tensor(self.theta_x)
            u_scatter_y = self._scatter_graph_y.to_tensor(self.theta_y)

            u_scatter = torch.kron(u_scatter_x, u_scatter_y)
            u_scatter = u_scatter.unsqueeze(0).expand(len(rho), -1, -1)

            if self.in_channels is not None:
                ic = torch.eye(self.in_channels)
                u_scatter = torch.kron(ic, u_scatter)
            rho = torch.bmm(u_scatter, rho)

        rho = self._new_conv(rho, adjoint=True)
        return rho

    def __repr__(self):
        c_in = 0 if self.in_channels is None else self.in_channels
        c_out = 0 if self.qconv.in_channels is None else self.qconv.in_channels

        return f"QTransposeConv2d({self.in_dims}, kernel_size={self.kernel_size}, in_channels={c_in}, out_channels={c_out})"

    def _build_scatter_graphs(self):
        """Applies beam-splitters to mix inserted and pre-existing modes."""
        scatter_x = Circuit(self.out_dims[0])

        for j in range(self.out_dims[0] // self.kernel_size):
            gi = GenericInterferometer(
                self.kernel_size,
                lambda i: BS.Ry(P(f"theta{i + j * self.kernel_size}")),  # noqa: B023
            )
            self._set_param_names(gi)
            scatter_x.add(j * self.kernel_size, gi)

        num_theta_x = len(scatter_x.get_parameters())

        scatter_y = Circuit(self.out_dims[1])
        for j in range(self.out_dims[0] // self.kernel_size):
            gi = GenericInterferometer(
                self.kernel_size,
                lambda i: BS.Ry(P(f"theta{i + j * self.kernel_size + num_theta_x}")),  # noqa: B023
            )
            self._set_param_names(gi)
            scatter_y.add(j * self.kernel_size, gi)

        num_theta_y = len(scatter_y.get_parameters())

        self.theta_x = torch.nn.Parameter(2 * np.pi * torch.rand(num_theta_x))
        self.theta_y = torch.nn.Parameter(2 * np.pi * torch.rand(num_theta_y))

        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            # Build circuit graphs for the two registers separately.
            self._scatter_graph_x = CircuitConverter(
                scatter_x, input_specs=["theta"], dtype=torch.float32
            )
            self._scatter_graph_y = CircuitConverter(
                scatter_y, input_specs=["theta"], dtype=torch.float32
            )
        finally:
            sys.stdout = original_stdout


class _InsertChannels(torch.nn.Module):
    """
    Helper class for adding channels to bottom of channel register
    within convolutional layers.

    Args:
        dims (tuple): Input dimensions
        in_channels (int): Initial size of channel register
        out_channels (int): Output size of channel register
    """

    def __init__(self, dims, in_channels, out_channels):
        super().__init__()
        d = dims[0]
        c_in = in_channels
        c_out = out_channels

        super().__init__()
        self.dims = dims
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Create all index combinations at once
        x = torch.arange(c_in * d**2)
        y = torch.arange(c_in * d**2)

        # f, h represent the channel indices.
        # In basis: |e_f>|e_i>|e_j><e_h|<e_l|<e_m|
        f = x // (d**2)
        h = y // (d**2)

        # i, j, l, m represent the one hot indices
        i = (x % (d**2)) // d
        j = (x % (d**2)) % d
        l = (y % (d**2)) // d  # noqa: E741
        m = (y % (d**2)) % d

        new_f = f + (c_out - c_in)
        new_h = h + (c_out - c_in)

        new_x = i * d + j + new_f * d**2
        new_y = l * d + m + new_h * d**2

        x_coords, y_coords = torch.meshgrid(new_x, new_y, indexing="ij")
        self._new_x = x_coords.flatten()
        self._new_y = y_coords.flatten()

    def forward(self, rho):
        b = rho.shape[0]
        c_out = self.out_channels
        d1, d2 = self.dims[0] ** 2, self.dims[1] ** 2

        new_rho = torch.zeros(
            (b, c_out * d1, c_out * d2), dtype=rho.dtype, device=rho.device
        )
        rho_flat = rho.view(b, -1)

        x_coords = self._new_x.unsqueeze(0).expand(b, -1)
        y_coords = self._new_y.unsqueeze(0).expand(b, -1)
        batch_indices = (
            torch.arange(b, device=rho.device)
            .unsqueeze(-1)
            .expand(-1, self._new_x.size(0))
        )

        # Broadcast values to new density matrix indices
        new_rho.index_put_(
            (batch_indices, x_coords, y_coords), rho_flat, accumulate=True
        )
        return new_rho


class _InsertMainModes(torch.nn.Module):
    """
    Inserts empty modes into specified locations in main register.

    Args:
        dims (tuple): Input image dimensions
        insertions (list): List of mode indices to insert empty modes
            after.
    """

    def __init__(self, dims, insertions: list[int], channels=None):
        c = 1 if channels is None else channels
        d = dims[0]

        super().__init__()
        if max(insertions) > 2 * d:
            raise ValueError(f"Insertions {insertions} exceed input dimensions {d}")

        self.insertions = insertions
        self.in_dims = dims
        self.channels = c

        self._new_d = d + len(insertions) // 2

        # Insertion coordinates within each register
        insertions_x = torch.tensor(
            [insert for insert in insertions if insert < dims[0]]
        )
        insertions_y = torch.tensor(
            [insert - dims[0] for insert in insertions if insert >= dims[0]]
        )
        if len(insertions_x) != len(insertions_y):
            raise NotImplementedError("Asymmetric insertions not supported yet.")

        x = torch.arange(c * d**2)
        y = torch.arange(c * d**2)

        # f, h represent the channel indices.
        # In basis: |e_f>|e_i>|e_j><e_h|<e_l|<e_m|
        f = x // (d**2)
        h = y // (d**2)

        # Let i, j, l, m represent the one hot indices
        i = (x % (d**2)) // d
        j = (x % (d**2)) % d
        l = (y % (d**2)) // d  # noqa: E741
        m = (y % (d**2)) % d

        # Apply insertions to shift indices
        if insertions_x.numel():
            i += (insertions_x[None, :] <= i[:, None]).sum(dim=1)
            l += (insertions_x[None, :] <= l[:, None]).sum(dim=1)  # noqa: E741

        if insertions_y.numel():
            j += (insertions_y[None, :] <= j[:, None]).sum(dim=1)
            m += (insertions_y[None, :] <= m[:, None]).sum(dim=1)

        # Create & flatten meshgrids for channel indices
        f_grid, h_grid = torch.meshgrid(f, h, indexing="ij")
        f_flat = f_grid.flatten()
        h_flat = h_grid.flatten()

        # Repeat i, j, l, m to match the flattened channel meshgrid size
        i = i.repeat(len(h))
        j = j.repeat(len(h))
        l = l.repeat_interleave(len(f))  # noqa: E741
        m = m.repeat_interleave(len(f))

        # Calculate new density matrix coordinates
        self._new_x = i * self._new_d + j + f_flat * self._new_d**2
        self._new_y = l * self._new_d + m + h_flat * self._new_d**2

        x_flat = x.repeat(len(y))
        y_flat = y.repeat_interleave(len(x))
        self._mask_coords = (x_flat, y_flat)

    def forward(self, rho):
        b = len(rho)
        c = self.channels

        # Create batch indices
        b_indices = torch.arange(b).repeat_interleave(len(self._new_x))

        # Expand coordinate arrays for all batches
        new_x = self._new_x.unsqueeze(0).expand(b, -1).reshape(-1)
        new_y = self._new_y.unsqueeze(0).expand(b, -1).reshape(-1)

        new_rho = torch.zeros(
            b,
            c * self._new_d**2,
            c * self._new_d**2,
            dtype=rho.dtype,
            device=rho.device,
        )
        values = rho[:, self._mask_coords[0], self._mask_coords[1]].reshape(-1)

        new_rho.index_put_((b_indices, new_x, new_y), values, accumulate=True)

        return new_rho

    def __repr__(self):
        c = 1 if self.channels is None else self.channels
        return f"_InsertMainModes({self.in_dims}, insertions={self.insertions}, channels={c})"


"""
Symbolic cheat sheet:
d = dimensions
c = channels
b = batches
k = kernel size

x, y = plain coordinates within density matrix
i, j, m, l = Main register one hot indices
f, h = Channel register one hot indices

Matrix is written in basis |e_f>|e_i>|e_j><e_h|<e_m|<e_l|
where |e_i> = |0>^{i} |1> |0> ^{d - i - 1}

Here, i, m are encoding the "X"-axis of the image.
j, l are encoding the "Y"-axis of the image.
"""
