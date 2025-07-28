"""
Quantum random kitchen sinks implementation for kernel approximation.

This module implements quantum-enhanced random kitchen sinks for approximating
Gaussian kernels using photonic quantum circuits. The approach combines classical
random feature methods with quantum circuits to compute kernel approximations.
"""

import perceval as pcvl
import numpy as np
from merlin import QuantumLayer, OutputMappingStrategy
import torch

def get_random_w_b(r, random_state):
    """
    Generate random weights and biases for random Fourier features.
    
    Args:
        r (int): Number of random features
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (w, b) where w is weight matrix and b is bias vector
    """
    np.random.seed(random_state)
    w = np.random.normal(size=(r, 2))
    b = np.random.uniform(low=0.0, high=2.0 * np.pi, size=(r,))
    return w, b

def get_x_r_i_s(x_s, w, b, r, gamma):
    """
    Given input data points x_s, of size [num_points, num_features],
    Return the x_{r, i}_s of size [num_points, r] such that
    x_{r, i} = gamma * (w_r * x_i + b_r)
    """
    num_points, num_features = x_s.shape

    x_r_i_s = gamma * (np.matmul(x_s, w.T) + np.tile(b, (num_points, 1)))
    assert x_r_i_s.shape == (num_points, r), f'Wrong shape for x_r_i_s: {x_r_i_s.shape}'

    return x_r_i_s

def get_z_s_classically(x_r_i_s):
    """
    Compute classical random kitchen sinks features.
    
    Args:
        x_r_i_s (np.array): Transformed input features
        
    Returns:
        np.array: Classical random features z(x) = sqrt(2/r) * cos(gamma * (w * x + b))
    """
    n, r = x_r_i_s.shape
    z_s = np.sqrt(2) * np.cos(x_r_i_s)
    z_s = z_s / np.sqrt(r)
    return z_s

def get_approx_kernel_train(z_s):
    """
    Compute approximate kernel matrix for training data.
    
    Args:
        z_s (np.array): Random features for training data
        
    Returns:
        np.array: Approximate kernel matrix K ≈ z(x) * z(x)^T
    """
    result_matrix = np.matmul(z_s, z_s.T)
    assert result_matrix.shape == (z_s.shape[0], z_s.shape[0]), f'Wrong shape for result_matrix: {result_matrix.shape}'
    return result_matrix

def get_approx_kernel_predict(z_s_test, z_s_train):
    """
    Compute approximate kernel matrix between test and training data.
    
    Args:
        z_s_test (np.array): Random features for test data
        z_s_train (np.array): Random features for training data
        
    Returns:
        np.array: Approximate kernel matrix K ≈ z(x_test) * z(x_train)^T
    """
    result_matrix = np.matmul(z_s_test, z_s_train.T)
    assert result_matrix.shape == (z_s_test.shape[0], z_s_train.shape[0]), f'Wrong shape for result_matrix: {result_matrix.shape}'
    return result_matrix

def get_mzi():
    """
    Create a Mach-Zehnder interferometer quantum circuit.
    
    Returns:
        pcvl.Circuit: MZI circuit with beam splitters and phase shifter
    """
    circuit = pcvl.Circuit(2)
    circuit.add(0, pcvl.BS())
    circuit.add(0, pcvl.PS(pcvl.P('data')))
    circuit.add(0, pcvl.BS())
    return circuit

def get_general():
    """
    Create a general interferometer quantum circuit with trainable parameters.
    
    Returns:
        pcvl.Circuit: General interferometer circuit with variable phase shifters
    """
    left_side = pcvl.GenericInterferometer(2,
                                           lambda i: pcvl.BS() // pcvl.PS(phi=pcvl.P(f"theta_psl1{i}")) // \
                                                     pcvl.BS() // pcvl.PS(phi=pcvl.P(f"theta_{i}")),
                                           shape=pcvl.InterferometerShape.RECTANGLE)
    right_side = pcvl.GenericInterferometer(2,
                                            lambda i: pcvl.BS() // pcvl.PS(phi=pcvl.P(f"theta_psr1{i}")) // \
                                                      pcvl.BS() // pcvl.PS(phi=pcvl.P(f"theta_psr2{i}")),
                                            shape=pcvl.InterferometerShape.RECTANGLE)

    circuit = pcvl.Circuit(2)
    circuit.add(0, left_side)
    circuit.add(0, pcvl.PS(pcvl.P('data')))
    circuit.add(0, right_side)
    return circuit

def get_circuit(args):
    """
    Get quantum circuit based on configuration.
    
    Args:
        args: Configuration object with circuit type specification
        
    Returns:
        tuple: (circuit, trainable_parameters) for the specified architecture
        
    Raises:
        ValueError: If circuit type is not supported
    """
    if args.circuit == "mzi":
        return get_mzi(), []
    elif args.circuit == "general":
        return get_general(), ['theta']
    else:
        raise ValueError(f'Wrong circuit type: {args.circuit}')

def save_circuit_locally(circuit, path):
    """
    Save quantum circuit visualization to file.
    
    Args:
        circuit (pcvl.Circuit): Circuit to visualize
        path (str): Output file path
    """
    pcvl.pdisplay_to_file(circuit, path)

def get_input_fock_state(num_photons):
    """
    Generate input Fock state for given number of photons.
    
    Args:
        num_photons (int): Total number of photons
        
    Returns:
        list: Fock state distribution across 2 modes
    """
    if num_photons % 2 == 0:
        return [int(num_photons / 2), int(num_photons / 2)]
    else:
        return [int(1 + (num_photons // 2)), int(num_photons // 2)]

def get_output_mapping(strategy):
    """
    Configure output mapping strategy for quantum layer.
    
    Args:
        strategy (str): Output mapping strategy ('NONE', 'LINEAR', 'GROUPING')
        
    Returns:
        tuple: (mapping_strategy, output_size, add_external_layer)
        
    Raises:
        ValueError: If strategy is not supported
    """
    if strategy == 'NONE':
        return OutputMappingStrategy.NONE, None, True
    elif strategy == 'LINEAR':
        return OutputMappingStrategy.LINEAR, 1, False
    elif strategy == 'GROUPING':
        return OutputMappingStrategy.GROUPING, 1, False
    else:
        raise ValueError(f'Unknown strategy {strategy}')

def get_q_model(args):
    """
    Create quantum model for random kitchen sinks implementation.
    
    Args:
        args: Configuration object containing model parameters
        
    Returns:
        torch.nn.Module: Complete quantum model (with optional linear layer)
    """
    torch.manual_seed(args.random_state)

    input_fock_state = get_input_fock_state(int(args.num_photon))
    circuit, trainable_params = get_circuit(args)
    save_circuit_locally(circuit, './results/circuit.png')
    output_mapping_strategy, output_size, add_external_layer = get_output_mapping(args.output_mapping_strategy)

    q_layer = QuantumLayer(
                input_size=1,
                output_size=output_size,
                circuit=circuit,
                trainable_parameters=trainable_params,
                input_parameters=["data"],
                input_state=input_fock_state,
                no_bunching=args.no_bunching,
                output_mapping_strategy=output_mapping_strategy
            )

    if add_external_layer:
        linear_layer = torch.nn.Linear(int(q_layer.output_size), 1)
        return torch.nn.Sequential(q_layer, linear_layer)

    return q_layer