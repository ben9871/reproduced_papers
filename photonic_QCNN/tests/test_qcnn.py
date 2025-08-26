import pytest
import torch
from perceval import BasicState, BSDistribution
from perceval.algorithm import Sampler

from photonic_QCNN.src.qcnn import OneHotEncoder, QConv2d, QDense, QPooling
from photonic_QCNN.tests._processor_builder import ProcessorBuilder


def _create_input_statevector(x):
    """Create OneHot encoded Perceval statevector from square image, x.
    """
    input_state = 0 * BasicState([])
    for i in range(len(x)):
        for j in range(len(x[i])):
            state_i = [0] * i + [1] + [0] * (len(x) - i - 1)
            state_j = [0] * j + [1] + [0] * (len(x) - j - 1)
            state = x[i][j].item() * BasicState(state_i + state_j + [1, 1])
            input_state += state
    return input_state

def _trim_results(distribution: BSDistribution, m: int):
    """Truncate a given Perceval BSDistribution to the first `m` modes.
    """
    trimmed_results = {}
    for key, val in distribution.items():
        new_key = key[:m]
        if new_key in trimmed_results:
            trimmed_results[new_key] += val
        else:
            trimmed_results[new_key] = val
    return trimmed_results


@pytest.mark.parametrize("dims", [(4, 4), (6, 6)])
def test_pooling(dims):
    """
    Test the qcnn.QPooling class by cross referencing the measured probs
    with a Perceval processor consisting of the following structure:
    >> ENCODING >> QCONV2D >> QPOOLING >> QCONV2D
    """
    # ---- CHECK QCNN w/ Sampler ----
    qcnn = ProcessorBuilder(dims)
    qcnn.add_conv(kernel_size=2)

    # Num params in first Conv layer
    num_params_conv = len(qcnn.free_parameters)

    qcnn.add_pooling(kernel_size=2)
    qcnn.add_conv(kernel_size=2)
    qcnn.fix_parameters()

    # Parameters in each conv
    conv1_params = [float(p) for p in qcnn.fixed_parameters[:num_params_conv]]
    conv2_params = [float(p) for p in qcnn.fixed_parameters[num_params_conv:]]

    x = torch.rand(dims)

    # Create input state
    input_state = _create_input_statevector(x)

    # Run processor and obtain results
    processor = qcnn.processor
    processor.min_detected_photons_filter(0)
    processor.with_input(input_state)
    sampler = Sampler(processor)
    results = sampler.probs()['results']

    # Group results based off state in first sum(dims) // 2 modes
    trimmed_results = _trim_results(results, sum(dims) // 2)

    results_processor = list(trimmed_results.values())
    results_processor.sort()
    results_processor = torch.tensor(results_processor)

    # ---- CHECK QCNN SUBMODULES ----
    encoder = OneHotEncoder()
    conv1 = QConv2d(dims=dims, kernel_size=2, stride=2)
    pool = QPooling(dims, kernel_size=2)
    conv2 = QConv2d(dims=(dims[0] // 2, dims[1] // 2), kernel_size=2, stride=2)

    num_params1 = len(conv1_params)
    num_params2 = len(conv2_params)

    conv1.phi_x.data = torch.tensor(conv1_params)[:num_params1 // 2]
    conv1.phi_y.data = torch.tensor(conv1_params)[num_params1 // 2:]

    conv2.phi_x.data = torch.tensor(conv2_params)[:num_params2 // 2]
    conv2.phi_y.data = torch.tensor(conv2_params)[num_params2 // 2:]

    x = x.unsqueeze(0)
    rho = encoder(x)
    rho = conv1(rho)
    rho = pool(rho)
    rho = conv2(rho)

    results_submodules = torch.abs(rho[0]).diagonal(dim1=0, dim2=1)
    results_submodules = torch.sort(results_submodules)[0]

    assert torch.allclose(results_submodules, results_processor)


@pytest.mark.parametrize("dims", [(4, 4), (6, 6)])
def test_dense(dims):
    """
    Test the qcnn.QDense class by cross referencing the measured probs
    with a Perceval processor consisting of the following structure:
    >> ENCODING >> QCONV2D >> QPOOLING >> QDENSE
    """
    # ---- CHECK QCNN w/ perceval processor & sampler ----
    qcnn = ProcessorBuilder(dims)
    qcnn.add_conv(kernel_size=2)

    # Num params in first Conv layer
    len_params_conv1 = len(qcnn.free_parameters)

    qcnn.add_pooling(kernel_size=2)
    qcnn.add_dense()
    qcnn.fix_parameters()

    # Parameters in each conv
    conv1_params = [float(p) for p in qcnn.fixed_parameters[:len_params_conv1]]
    dense_params = [float(p) for p in qcnn.fixed_parameters[len_params_conv1:]]

    x = torch.rand(dims)

    # Create input statevector for Perceval processor
    input_state = _create_input_statevector(x)

    # Run processor and obtain results
    processor = qcnn.processor
    processor.min_detected_photons_filter(0)
    processor.with_input(input_state)
    sampler = Sampler(processor)
    results = sampler.probs()['results']

    # Group results based off state in first m_remaining modes
    trimmed_results = _trim_results(results, sum(dims) // 2)

    results_processor = list(trimmed_results.values())
    results_processor.sort()
    results_processor = torch.tensor(results_processor)

    # ---- CHECK QCNN SUBMODULES ----
    encoder = OneHotEncoder()
    conv1 = QConv2d(dims=dims, kernel_size=2, stride=2)
    pool = QPooling(dims, kernel_size=2)
    dense = QDense(dims=(dims[0] // 2, dims[1] // 2))

    # Assign phi parameters in Conv & Dense torch layers
    num_conv_params = len(conv1_params)
    conv1.phi_x.data = torch.tensor(conv1_params)[:num_conv_params // 2]
    conv1.phi_y.data = torch.tensor(conv1_params)[num_conv_params // 2:]

    dense.phi.data = torch.tensor(dense_params)

    x = x.unsqueeze(0)
    rho = encoder(x)
    rho = conv1(rho)
    rho = pool(rho)
    rho = dense(rho)

    results_submodules = torch.abs(rho[0]).diagonal(dim1=0, dim2=1)
    results_submodules = torch.sort(results_submodules)[0]

    assert torch.allclose(results_submodules, results_processor, rtol=5e-3)
