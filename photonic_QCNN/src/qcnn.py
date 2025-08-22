import sys
import io
import re
from typing import Union, Generator

import math
import numpy as np
import sympy as sp
import torch
from torch import Tensor, nn

from merlin import CircuitConverter
from merlin import build_slos_distribution_computegraph as build_slos_graph

from perceval import catalog
from perceval.components import GenericInterferometer, Circuit


def generate_all_fock_states(m, n) -> Generator:
    """
    Generate all possible Fock states for n photons and m modes.
    
    Args:
        m: Number of modes.
        n: Number of photons.
    
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
        for state in generate_all_fock_states(m-1, n-i):
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
        
    def forward(self, X: Tensor) -> Tensor:
        if X.dim() == 3:
            X = X.unsqueeze(1)
        
        norm = torch.sqrt(torch.square(torch.abs(X)).sum(dim=(1, 2, 3)))
        X = X / norm.view(-1, 1, 1, 1)
        
        # Flatten each image and multiply by transpose to get density matrix
        X_flat = X.reshape(X.shape[0], -1)
        rho = X_flat.unsqueeze(2) @ X_flat.unsqueeze(1)
        rho = rho.to(torch.complex64)
        
        return rho
    
    def __repr__(self):
        return f"OneHotEncoder()"


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
            raise NotImplementedError('Non-square images not supported yet.')
        
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
                re.search(r'\d+', self._training_params[-1].name).group()
            )
        
        for i, p in enumerate(param_list):
            p.name = f"phi{i + param_start_idx + 1}"
        
        for _, comp in circuit:
            if hasattr(comp, "_phi"):
                param = comp.get_parameters()[0]
                param._symbol = sp.S(param.name)
        
        self._training_params.extend(param_list)


class QConv2d(AParametrizedLayer):
    def __init__(self,
        dims,
        kernel_size: int,
        stride: int = None,
    ):
        """
        Quantum 2D Convolutional layer
        
        Args:
            dims: Input dimensions.
            kernel_size: Size of universal interferometer.
            stride: Stride of the universal interferometer across the 
                modes.
        """
        
        super().__init__(dims)
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        
        # Define filters
        filters = []
        for _ in range(2):
            filter = GenericInterferometer(
                kernel_size, catalog['mzi phase first'].generate
            )
            self._set_param_names(filter)
            filters.append(filter)
        
        # Create x and y registers
        self._reg_x = Circuit(dims[0], name='Conv X')
        self._reg_y = Circuit(dims[1], name='Conv Y')
        
        # Add filters with specified stride
        for i in range((dims[0] - kernel_size) // self.stride + 1):
            self._reg_x.add(self.stride * i, filters[0])
        
        for i in range((dims[1] - kernel_size) // self.stride + 1):
            self._reg_y.add(self.stride * i, filters[1])
        
        num_params_x = len(self._reg_x.get_parameters())
        num_params_y = len(self._reg_y.get_parameters())
        
        # Suppress unnecessary print statements from pcvl_pytorch
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            # Build circuit graphs for the two registers separately.
            self._circuit_graph_x = CircuitConverter(
                self._reg_x,
                ['phi'],
                torch.float32
            )
            self._circuit_graph_y = CircuitConverter(
                self._reg_y,
                ['phi'],
                torch.float32
            )
        finally:
            sys.stdout = original_stdout
        
        # Create model parameters
        self.phi_x = 2*np.pi * nn.Parameter(torch.rand(num_params_x))
        self.phi_y = 2*np.pi * nn.Parameter(torch.rand(num_params_y))
    
    def forward(self, rho, adjoint = False):
        b = len(rho)
        
        Ux = self._circuit_graph_x.to_tensor(self.phi_x)
        Uy = self._circuit_graph_y.to_tensor(self.phi_y)
        U = torch.kron(Ux, Uy)
        
        U = U.unsqueeze(0).expand(b, -1, -1)
        Udag = U.transpose(1, 2).conj()
        
        # There is only one photon in each register, can apply the U directly.
        if not adjoint:
            U_rho = torch.bmm(U, rho)
            new_rho = torch.bmm(U_rho, Udag)
        else:
            # Apply adjoint to rho
            Udag_rho = torch.bmm(Udag, rho)
            new_rho = torch.bmm(Udag_rho, U)
        
        return new_rho
    
    def __repr__(self):
        return f"QConv2d({self.dims}, kernel_size={self.kernel_size}), stride={self.stride}"


class QPooling(torch.nn.Module):
    """
    Quantum pooling layer.

    Reduce the size of the encoded image by the given kernel size.
    
    Args:
        dims: Input image dimensions.
        kernel_size: Dimension by which the image is reduced.
    """
    def __init__(self, dims: tuple[int], kernel_size: int):
        d = dims[0]
        k = kernel_size
        new_d = d // kernel_size
        
        super().__init__()
        self.dims = d
        self._new_d = new_d
        self.kernel_size = k
        
        # Create all index combinations at once
        x = torch.arange(d ** 2)
        y = torch.arange(d ** 2)
        
        # Let f, h represent the channel indices.
        # Channels not included in current script.
        # In basis: |e_f>|e_i>|e_j><e_h|<e_l|<e_m|
        f = x // (d ** 2)
        h = y // (d ** 2)
        
        # Let i, j, l, m represent the one hot indices
        i = (x % (d ** 2)) // d
        j = (x % (d ** 2)) % d
        l = (y % (d ** 2)) // d
        m = (y % (d ** 2)) % d
        
        f_grid, h_grid = torch.meshgrid(f, h, indexing='ij')
        i_grid, l_grid = torch.meshgrid(i, l, indexing='ij')
        j_grid, m_grid = torch.meshgrid(j, m, indexing='ij')
        
        # Ensure that odd mode photon numbers match.
        match_odd1 = ((i_grid % k != 0) & (i_grid == l_grid)) | (i_grid % k == 0)
        match_odd2 = ((j_grid % k != 0) & (j_grid == m_grid)) | (j_grid % k == 0)
        
        # Ensure photon number in ancillae used for photon injection match
        inject_condition = (i_grid % k == l_grid % k) & (j_grid % k == m_grid % k)
        
        mask = inject_condition & match_odd1 & match_odd2
        mask_coords = torch.nonzero(mask, as_tuple=False)
        self._mask_coords = (mask_coords[:, 0], mask_coords[:, 1])
        
        # New one hot indices
        new_i = i_grid[mask] // k
        new_j = j_grid[mask] // k
        new_l = l_grid[mask] // k
        new_m = m_grid[mask] // k
        
        # New matrix coordinates
        self._new_x = new_i * new_d + new_j + f_grid[mask] * new_d ** 2
        self._new_y = new_l * new_d + new_m + h_grid[mask] * new_d ** 2
    
    def forward(self, rho):
        b = len(rho)
        
        b_indices = torch.arange(b).unsqueeze(1).expand(-1, len(self._new_x))
        b_indices = b_indices.reshape(-1)
        
        new_x = self._new_x.expand(b, -1).reshape(-1)
        new_y = self._new_y.expand(b, -1).reshape(-1)
        
        new_rho = torch.zeros(b, self._new_d ** 2, self._new_d ** 2,
                            dtype=rho.dtype, device=rho.device)
        
        values = rho[:, self._mask_coords[0], self._mask_coords[1]].reshape(-1)
        new_rho.index_put_((b_indices, new_x, new_y), values, accumulate=True)
        
        return new_rho
    
    def __repr__(self):
        return f"QPooling({self.dims}, kernel_size={self.kernel_size})"


# For QDense layer, patch merlin.SLOSGraph to add method to return amplitudes
def compute_amplitudes(self,
    unitary: Tensor,
    input_state: list[int]
) -> torch.Tensor:
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
            "Input state must be binary (0s and 1s only) in non-bunching mode")
    
    batch_size, m, m2 = unitary.shape
    if m != m2 or m != self.m:
        raise ValueError(
            f"Unitary matrix must be square with dimension {self.m}x{self.m}")
    
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
            
            bounds1 = self.index_photons[len(idx_n)-1][1]
            bounds2 = self.index_photons[len(idx_n)-1][0]
            if (i > bounds1) or (i < bounds2):
                raise ValueError(
                    f"Input state photons must be bounded by {self.index_photons}")
    
    # Get device from unitary
    device = unitary.device
    
    # Initial amplitude 
    amplitudes = torch.ones(
        (batch_size, 1), dtype=self.complex_dtype, device=device
    )
    
    # Apply each layer
    for layer_idx, layer_fn in enumerate(self.layer_functions):
        p = idx_n[layer_idx]
        amplitudes, self.contributions = layer_fn(
            unitary, amplitudes, p, return_contributions=True)
    
    self.prev_amplitudes = amplitudes
    
    # Normalize the amplitudes
    self.norm_factor_output = self.norm_factor_output.to(device=device)
    amplitudes = amplitudes * torch.sqrt(self.norm_factor_output)
    amplitudes = amplitudes / math.sqrt(self.norm_factor_input)
    
    return amplitudes


class QDense(AParametrizedLayer):
    """
    Quantum Dense layer 
    
    Expects an input density matrix in the One Hot Amplitude basis and 
    performs SLOS to return the output density matrix in the whole Fock 
    space.
    
    Args:
        dims (tuple[int]): Input image dimensions.
        m (int | list[int]): Size of the dense layers placed in 
            succession. If `None`, a single universal dense layer is 
            applied.
    """
    def __init__(
        self,
        dims,
        m: Union[int, list[int]] = None,
        device = None
    ):
        super().__init__(dims)
        
        self.dims = dims
        self.device = device
        
        m = m if m is not None else sum(dims)
        self.m = [m]
        
        # Construct circuit and circuit graph
        self._training_params = []
        
        self.circuit = Circuit(max(self.m))
        for m in self.m:
            gi = GenericInterferometer(m, catalog['mzi phase first'].generate)
            self._set_param_names(gi)
            self.circuit.add(0, gi)
        
        # Suppress unnecessary print statements
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            self._circuit_graph = CircuitConverter(
                self.circuit,
                ['phi'],
                torch.float32
            )
        finally:
            sys.stdout = original_stdout
        
        # Set up input states & SLOS graphs
        self._input_states = [
            tuple(int(i==x) for i in range(dims[0])) + \
            tuple(int(i==y) for i in range(dims[1]))
            for x in range(dims[1])
            for y in range(dims[0])
        ]
        
        self._slos_graph = build_slos_graph(
            m=max(self.m),
            n_photons=2,
            device=self.device,
        )
        self._slos_graph.__class__.compute_amplitudes = compute_amplitudes
        
        # Create and register model parameters
        num_params = len(self._training_params)
        self.phi = nn.Parameter(2*np.pi * torch.rand(num_params))
        
    def forward(self, rho):
        b = len(rho)
        
        # Run SLOS & extract amplitudes.
        unitary = self._circuit_graph.to_tensor(self.phi)
        
        amplitudes = torch.stack([
            self._slos_graph.compute_amplitudes(unitary, basis_state)
            for basis_state in self._input_states
        ]).squeeze()
        
        u_evolve = amplitudes.T
        
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
        m = self.m[0] if len(self.m) == 1 else self.m
        return f"QDense({self.dims}, m={m})"


class Measure(nn.Module):
    """
    Measurement operator.
    
    Assumes input is written in Fock basis and extracts diagonal.
    
    If one would like to perform a partial measurement, the following 
    params can be specified.
    
    Args:
        m (int): Total number of modes in-device. Default: None.
        n (int): Number of photons in-device. Default: 2.
        subset (int): Number of modes being measured. Default: None.
    """
    def __init__(self, m: int = None, n: int = 2, subset: int = None):
        super().__init__()
        self.m = m
        self.subset = subset
        
        if subset is not None:
            all_states = list(generate_all_fock_states(m, 2))
            reduced_states = []
            for i in range(n + 1):
                reduced_states += list(generate_all_fock_states(subset, i))
            
            self.indices = torch.tensor([
                reduced_states.index(state[:subset])
                for state in all_states
            ])
    
    def forward(self, rho):
        b = len(rho)
        probs = torch.abs(rho.diagonal(dim1=1, dim2=2))
        
        if self.subset is not None:
            indices = self.indices.unsqueeze(0).expand(b, -1)
            probs_output = torch.zeros(
                indices.shape, device=probs.device, dtype=probs.dtype
            )
            probs_output.scatter_add_(dim=1, index=indices, src=probs)
            return probs_output
        
        return probs
    
    def __repr__(self):
        if self.subset is not None:
            return f'Measure(m={self.m}, n={self.n}, subset={self.subset})'
        else:
            return f'Measure()'



""" 
Symbolic cheat sheet:
d = dimensions 
b = batches 
k = kernel size

x, y = plain coordinates within density matrix
f, h = Channel register one hot indices (Channels not included in this script.)
i, j, m, l = Main register one hot indices 

Matrix is written in basis |e_f>|e_i>|e_j><e_h|<e_m|<e_l| 
where |e_i> = |0>^{i} |1> |0> ^{d - i - 1}

Here, i, m are encoding the "X"-axis of the image.
j, l are encoding the "Y"-axis of the image.
"""