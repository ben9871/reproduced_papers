"""
Processor Builder.

Here one can build the QCNN as a `Processor` in Perceval.
Used for visualizing the QCNN & testing.
"""

import re
from collections.abc import Sequence
from itertools import product
from typing import Union

import numpy as np
import sympy as sp
from perceval import BasicState, Parameter, Processor, StateVector
from perceval.components import (
    Detector,
    FFCircuitProvider,
    GenericInterferometer,
    catalog,
)
from perceval.components.unitary_components import PERM, Barrier, Circuit


class ProcessorBuilder:
    """
    QCNN Processor Builder.

    Builds the Perceval processor as each layer in the QCNN is called.

    Args:
        dims (tuple[int]): Dimension of data embedded in the QCNN
    """

    def __init__(self, dims: tuple):
        self.dims = dims

        if len(dims) != 2:
            raise ValueError("QCNN processes 2D data only.")

        self.m = sum(self.dims)
        self._processor = Processor("SLOS", self.m)

        self.free_parameters = []

        # For keeping track of no. of modes across different pooling layers.
        self._m_history = [self.m]
        self._pool_history = []

        # No. of qubit modes after pooling layers
        self._m_remaining = self.m

        # Dimension of image after pooling
        self._dims_remaining = self.dims

        self._num_ancillae = 0
        self._num_conv = 0
        self._num_dense = 0
        self._num_pools = 0

    def add_conv(
        self,
        kernel_size: int,
        stride: int = None,
    ) -> None:
        """
        2D Quantum Convolution.

        Applies an interferometer of a given kernel size and across each
        register repeatedly.

        Args:
            kernel_size (int): Size of universal interferometer
            stride (int): Stride of the convolutional filter across
                the processor. Default: equal to the `kernel_size`.
        """
        if self._num_dense != 0:
            raise ValueError("Cannot add a convolutional layer after a dense layer.")

        stride = kernel_size if stride is None else stride

        filters = []
        for _ in range(2):
            filter = GenericInterferometer(
                kernel_size, catalog["mzi phase first"].generate
            )
            self._set_param_names(filter)

            filter.name = "Conv2D_Filter"
            filters.append(filter)

        conv_layer = Circuit(self._m_remaining, name=f"Conv2D{self._num_conv + 1}")

        # Apply convolutional filters across the processor.
        num_filters = (self._m_remaining // 2 - kernel_size) // stride + 1

        for i in range(num_filters):
            conv_layer.add(stride * i, filters[0])
            conv_layer.add(stride * i + self._m_remaining // 2, filters[1])

        self._processor.add(0, conv_layer)
        self._num_conv += 1

    def add_pooling(self, kernel_size: int = 2) -> None:
        """
        Quantum Pooling Layer.

        Introduces ancillary modes containing a single photon. When a
        photon is detected on the control modes, a swap is performed
        with the ancillary photon.

        Args:
            kernel_size: Dimension by which to decrease the size of the
                input state.
        """
        if self._m_remaining % kernel_size != 0:
            raise ValueError(
                "Number of modes remaining must be divisible by "
                "the kernel size before Pooling."
            )

        # Reduce dimensions of image.
        self._m_remaining //= kernel_size

        dim_x = self._dims_remaining[0] // kernel_size
        dim_y = self._dims_remaining[1] // kernel_size
        self._dims_remaining = (dim_x, dim_y)

        # Expand size of processor
        self._num_ancillae += 2
        self.m += 2

        self._m_history.append(self._m_remaining)
        self._pool_history.append(kernel_size)

        old_processor = self._processor
        self._processor = Processor("SLOS", self.m)
        pool_coord = 0

        # Add previous components to new processor
        for idx, pos_comp in enumerate(old_processor.components):
            past_m_remaining = self._m_history[pool_coord + 1]
            past_kernel_size = self._pool_history[pool_coord]
            position, component = pos_comp

            # Change feedfoward configurator position
            if self._num_pools and isinstance(component, PERM):
                if idx + 1 != len(old_processor.components):
                    # When a Barrier is the next component, we place a new
                    # pooling layer at new position.
                    if isinstance(old_processor.components[idx + 1][1], Barrier):
                        self._add_feedforward_components(
                            past_m_remaining, past_kernel_size
                        )
                        pool_coord += 1

            elif not isinstance(
                component, (PERM, FFCircuitProvider, Barrier, Detector)
            ):
                self._processor.add(position, component)

        self._add_feedforward_components(self._m_remaining, kernel_size)
        self._num_pools += 1

    def add_dense(self, m: int = None) -> None:
        """
        Quantum Dense Layer.

        Merge the two registers and apply a global interferometer
        across the processor.

        Args:
            m (int): Size of the dense layer.
        """
        if m is None:
            m = self._m_remaining

        if m > self._m_remaining:
            raise ValueError(
                "Dense layer cannot have more modes than the number of remaining modes."
            )

        dense_layer = GenericInterferometer(m, catalog["mzi phase first"].generate)
        self._set_param_names(dense_layer)

        dense_layer.name = "Dense" + str(self._num_dense + 1)

        self._processor.add(0, dense_layer)
        self._num_dense += 1

    def fix_parameters(
        self,
        params: Union[Sequence[Parameter], Parameter] = None,
        values: Union[Sequence[float], float] = None,
    ) -> None:
        """
        Fix circuit parameters and remove them as a trainable parameter.
        Default values are randomly assigned.

        Args:
            params: Sequence of parameters to be removed. If `None`, all
                parameters are fixed. Default: `None`.
            values: Values to set the parameters. If `None`, random
                values are selected.
        """
        if not hasattr(self, "fixed_parameters"):
            self.fixed_parameters = []

        self.fixed_parameters += self.free_parameters

        if params is None:
            params = self.free_parameters

        if values is None:
            values = np.random.uniform(0, 2 * np.pi, len(params))

        if isinstance(params, Parameter):
            params.set_value(values)
            self.free_parameters.remove(params)

        else:
            for i, param in enumerate(params):
                param.set_value(values[i])

            self.free_parameters = [p for p in self.free_parameters if p not in params]

    @property
    def n(self) -> int:
        active_photons = 2
        return active_photons + self._num_ancillae

    @property
    def processor(self) -> Processor:
        return self._processor

    def _input_state(self, datapoint: np.ndarray) -> StateVector:
        """Generate data-embedded initial state"""
        datapoint = np.asarray(datapoint)
        norm = np.sum(datapoint**2)
        amplitudes = np.float32(datapoint / norm)

        state_vec = StateVector()
        for x in range(self.dims[0]):
            for y in range(self.dims[1]):
                x_state = [int(i == x) for i in range(self.dims[0])]
                y_state = [int(i == y) for i in range(self.dims[1])]
                ancillae = [1] * self._num_ancillae

                # Tensor product the X, y & ancilla states
                input_state = x_state + y_state + ancillae

                state_vec += float(amplitudes[x][y]) * BasicState(input_state)
        return state_vec

    def _set_param_names(self, circuit: Circuit) -> None:
        """Assigns the names of a parametrized circuit"""
        param_list = circuit.get_parameters()

        if not self.free_parameters:
            param_start_idx = 0
        else:
            # Take index from last parameter name
            param_start_idx = int(
                re.search(r"\d+", self.free_parameters[-1].name).group()
            )

        for i, p in enumerate(param_list):
            p.name = f"phi{i + param_start_idx + 1}"

        for _, comp in circuit:
            if hasattr(comp, "_phi"):
                param = comp.get_parameters()[0]
                param._symbol = sp.S(param.name)

        self.free_parameters.extend(param_list)

    def _add_feedforward_components(self, m_reduced, kernel_size):
        """Adds photonic state injection components in pooling layer to
        processor.
        """
        ## Step 1. Bring ancillae next to target modes and drop control modes
        perm1 = [0] * (kernel_size * m_reduced + self._num_ancillae)

        # Move active modes
        for i in range(kernel_size * m_reduced):
            if i % kernel_size == 0:
                perm1[i] = i // kernel_size

        # Move in fresh ancillae next to active modes
        for i in range(self._num_ancillae):
            perm1[m_reduced * kernel_size + i] = m_reduced + i

        # Measured modes
        for i in range(kernel_size * m_reduced):
            if i % kernel_size != 0:
                perm1[i] = max(perm1) + 1

        self._processor.add(0, PERM(perm1))

        ## Step. 2: Add feedforward configurator
        for i in range(m_reduced * (kernel_size - 1)):
            detector_position = m_reduced + self._num_ancillae + i
            self._processor.add(detector_position, Detector.threshold())

        ## All possible measurements
        num_measured_modes = m_reduced * (kernel_size - 1) // 2
        measurements_register = [
            [1 if j == i else 0 for j in range(num_measured_modes)]
            for i in range(num_measured_modes)
        ]
        measurements_register += [[0] * (m_reduced // 2 * (kernel_size - 1))]

        measurements = [
            x + y for x, y in product(measurements_register, measurements_register)
        ]
        feedforward = FFCircuitProvider(
            m=m_reduced * (kernel_size - 1),
            offset=-self._num_ancillae + 1,
            default_circuit=Circuit(m_reduced + 2),
        )

        # Add configurations to FFCircuitProvider for each measurement
        for measurement in measurements:
            register_x = measurement[: len(measurement) // 2]
            register_y = measurement[len(measurement) // 2 :]

            # Group the measured modes into their corresponding pools
            register_x_grouped = [
                register_x[i : i + kernel_size - 1]
                for i in range(0, len(register_x), kernel_size - 1)
            ]
            register_y_grouped = [
                register_y[i : i + kernel_size - 1]
                for i in range(0, len(register_y), kernel_size - 1)
            ]
            swap_circuit = Circuit(m_reduced + 2)

            for i, group in enumerate(register_x_grouped):
                # If a photon is detected in a pooling filter, switch with the
                # ancilla for x register
                if sum(group) == 1:
                    injection_permutation = self._swap(
                        swap_circuit.m, i, swap_circuit.m - 2
                    )
                    swap_circuit.add(0, injection_permutation)
                    break

            for i, group in enumerate(register_y_grouped):
                # If a photon is detected in a pooling filter, switch with the
                # ancilla for x register
                if sum(group) == 1:
                    injection_permutation = self._swap(
                        swap_circuit.m, m_reduced // 2 + i, swap_circuit.m - 1
                    )
                    swap_circuit.add(0, injection_permutation)
                    break

            feedforward.add_configuration(measurement, swap_circuit)

        self._processor.add(m_reduced + self._num_ancillae, feedforward)

        # Switch out recently used ancillae
        permutation = [self._num_ancillae - 2, self._num_ancillae - 1]
        for i in range(self._num_ancillae - 2):
            permutation.append(i)

        self._processor.add(m_reduced, PERM(permutation))

    @staticmethod
    def _swap(m, mode1: int, mode2: int) -> PERM:
        """Returns a permutation of size m, swapping mode1 and mode2"""
        perm = [*range(m)]
        perm[mode1], perm[mode2] = mode2, mode1
        return PERM(perm)
