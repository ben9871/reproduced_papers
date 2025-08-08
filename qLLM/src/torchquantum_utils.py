"""
TorchQuantum model training utilities for qLLM experiments using MerLin.
"""

# to install torchquantum: clone the original repo https://github.com/mit-han-lab/torchquantum/tree/main
# pip install --editable . (I think pip install torchquantum is not up to date with latest qiskit versions)
# some guidance can be found in the original GitHub repo: https://github.com/mit-han-lab/torchquantum/tree/main
# This code was written mostly by Claude and reviewed by me

# typing import dict, list, tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle


class QuantumCircuitVisualizer:
    """
    Visualizer for quantum circuits in the hybrid model
    """

    def __init__(self):
        self.colors = {
            "ry": "#FF6B6B",
            "cnot": "#4ECDC4",
            "measurement": "#45B7D1",
            "wire": "#2C3E50",
            "amplitude_encoding": "#9B59B6",
            "angle_encoding": "#F39C12",
        }

    def draw_circuit(
        self,
        circuit_config: dict,
        title: str = "Quantum Circuit",
        figsize: tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """
        Draw a quantum circuit based on configuration

        Args:
            circuit_config: dictionary containing circuit structure
            title: Title for the plot
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        n_qubits = circuit_config["n_qubits"]
        n_layers = circuit_config.get("n_layers", 1)
        connectivity = circuit_config.get("connectivity", 1)
        encoding_type = circuit_config.get("encoding_type", "amplitude")

        # Set up the plot
        ax.set_xlim(0, 10 + n_layers * 4)  # More space for sequential blocks
        ax.set_ylim(-0.5, n_qubits - 0.5)
        ax.set_aspect("equal")

        # Draw qubit wires
        for i in range(n_qubits):
            ax.plot(
                [0, 10 + n_layers * 4],
                [n_qubits - 1 - i, n_qubits - 1 - i],
                color=self.colors["wire"],
                linewidth=2,
            )
            ax.text(
                -0.5,
                n_qubits - 1 - i,
                f"|q{i}⟩",
                verticalalignment="center",
                fontsize=12,
                fontweight="bold",
            )

        # Draw encoding block
        encoding_color = (
            self.colors["amplitude_encoding"]
            if encoding_type == "amplitude"
            else self.colors["angle_encoding"]
        )

        for i in range(n_qubits):
            y_pos = n_qubits - 1 - i
            rect = FancyBboxPatch(
                (0.5, y_pos - 0.3),
                2,
                0.6,
                boxstyle="round,pad=0.1",
                facecolor=encoding_color,
                edgecolor="black",
                linewidth=1.5,
            )
            ax.add_patch(rect)

            encoding_text = "Amp" if encoding_type == "amplitude" else "Ang"
            ax.text(
                1.5,
                y_pos,
                encoding_text,
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=10,
                color="white",
            )

        # Draw parameterized layers as sequential Ui blocks
        x_offset = 3
        for layer in range(n_layers):
            layer_x = x_offset + layer * 4  # More space for sequential blocks

            # Track which qubits get RY gates (only targets of CNOTs)
            _target_qubits = set()
            drawn_connections = set()

            # First, determine all CNOT pairs and their targets
            cnot_pairs = []
            for i in range(n_qubits):
                for j in range(
                    max(0, i - connectivity), min(n_qubits, i + connectivity + 1)
                ):
                    if (
                        i != j
                        and (i, j) not in drawn_connections
                        and (j, i) not in drawn_connections
                    ):
                        drawn_connections.add((i, j))
                        cnot_pairs.append((i, j))
                        _target_qubits.add(j)  # Only targets get RY gates

            # Draw each Ui block: CNOT first, then RY on target
            block_spacing = 0.8 if cnot_pairs else 0
            for block_idx, (control, target) in enumerate(cnot_pairs):
                cnot_x = layer_x + block_idx * block_spacing
                ry_x = cnot_x + 0.4

                control_y = n_qubits - 1 - control
                target_y = n_qubits - 1 - target

                # Draw CNOT gate
                # Control qubit (filled circle)
                control_circle = Circle(
                    (cnot_x, control_y), 0.15, facecolor="black", edgecolor="black"
                )
                ax.add_patch(control_circle)

                # Target qubit (circle with cross)
                target_circle = Circle(
                    (cnot_x, target_y),
                    0.25,
                    facecolor="white",
                    edgecolor=self.colors["cnot"],
                    linewidth=2,
                )
                ax.add_patch(target_circle)

                # Draw cross inside target
                ax.plot(
                    [cnot_x - 0.15, cnot_x + 0.15],
                    [target_y, target_y],
                    color=self.colors["cnot"],
                    linewidth=2,
                )
                ax.plot(
                    [cnot_x, cnot_x],
                    [target_y - 0.15, target_y + 0.15],
                    color=self.colors["cnot"],
                    linewidth=2,
                )

                # Connection line
                ax.plot(
                    [cnot_x, cnot_x],
                    [min(control_y, target_y), max(control_y, target_y)],
                    color=self.colors["cnot"],
                    linewidth=2,
                )

                # Draw RY gate on target qubit only
                ry_circle = Circle(
                    (ry_x, target_y),
                    0.3,
                    facecolor=self.colors["ry"],
                    edgecolor="black",
                    linewidth=1.5,
                )
                ax.add_patch(ry_circle)
                ax.text(
                    ry_x,
                    target_y,
                    "RY",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    fontsize=8,
                    color="white",
                )

                # Add parameter label
                ax.text(
                    ry_x,
                    target_y - 0.6,
                    f"θ{layer},{block_idx}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    style="italic",
                )

        # Draw measurement
        meas_x = x_offset + n_layers * 3 + 1
        for i in range(n_qubits):
            y_pos = n_qubits - 1 - i
            rect = FancyBboxPatch(
                (meas_x, y_pos - 0.25),
                1.5,
                0.5,
                boxstyle="round,pad=0.05",
                facecolor=self.colors["measurement"],
                edgecolor="black",
                linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.text(
                meas_x + 0.75,
                y_pos,
                "⟨Z⟩",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=12,
                color="white",
            )

        # Add title and labels
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Circuit Depth", fontsize=12)
        ax.set_ylabel("Qubits", fontsize=12)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add legend
        legend_elements = [
            mpatches.Patch(
                color=encoding_color, label=f"{encoding_type.title()} Encoding"
            ),
            mpatches.Patch(color=self.colors["ry"], label="RY Rotation"),
            mpatches.Patch(color=self.colors["cnot"], label="CNOT Gate"),
            mpatches.Patch(
                color=self.colors["measurement"], label="Pauli-Z Measurement"
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, 1))

        plt.tight_layout()
        return fig

    def draw_multi_encoder_architecture(
        self,
        encoder_configs: list[dict],
        quantum_config: dict,
        fusion_method: str = "concatenate",
        figsize: tuple[int, int] = (16, 10),
    ) -> plt.Figure:
        """
        Draw the complete multi-encoder architecture
        """
        fig = plt.figure(figsize=figsize)

        n_encoders = len(encoder_configs)
        n_rows = n_encoders + 2  # encoders + fusion + quantum PQC

        # Draw individual encoders
        for i, config in enumerate(encoder_configs):
            ax = plt.subplot(n_rows, 1, i + 1)

            # Create mini circuit for each encoder
            self._draw_mini_circuit(
                ax, config, f"sQE {i + 1}: {config['n_qubits']} qubits"
            )

        # Draw fusion layer
        fusion_ax = plt.subplot(n_rows, 1, n_encoders + 1)
        self._draw_fusion_layer(fusion_ax, encoder_configs, fusion_method)

        # Draw quantum PQC
        pqc_ax = plt.subplot(n_rows, 1, n_encoders + 2)
        self._draw_mini_circuit(
            pqc_ax, quantum_config, "Quantum PQC (Actual QPU)", encoding_type="angle"
        )

        plt.suptitle(
            "Multi-Encoder Hybrid Quantum Neural Network",
            fontsize=18,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig

    def _draw_mini_circuit(
        self, ax, config: dict, title: str, encoding_type: str = "amplitude"
    ):
        """Draw a simplified circuit representation"""
        n_qubits = config["n_qubits"]
        n_layers = config.get("n_layers", 1)

        ax.set_xlim(0, 8)
        ax.set_ylim(-0.5, n_qubits - 0.5)
        ax.set_aspect("equal")

        # Draw wires
        for i in range(n_qubits):
            ax.plot(
                [0, 8],
                [n_qubits - 1 - i, n_qubits - 1 - i],
                color=self.colors["wire"],
                linewidth=1.5,
            )

        # Encoding
        encoding_color = (
            self.colors["amplitude_encoding"]
            if encoding_type == "amplitude"
            else self.colors["angle_encoding"]
        )

        for i in range(n_qubits):
            y_pos = n_qubits - 1 - i
            rect = Rectangle(
                (0.5, y_pos - 0.2), 1, 0.4, facecolor=encoding_color, edgecolor="black"
            )
            ax.add_patch(rect)

        # Parameterized layers (simplified)
        for layer in range(min(n_layers, 3)):  # Show max 3 layers
            x_pos = 2 + layer * 1.5
            for i in range(n_qubits):
                y_pos = n_qubits - 1 - i
                circle = Circle(
                    (x_pos, y_pos), 0.15, facecolor=self.colors["ry"], edgecolor="black"
                )
                ax.add_patch(circle)

        # Measurement
        for i in range(n_qubits):
            y_pos = n_qubits - 1 - i
            rect = Rectangle(
                (6.5, y_pos - 0.2),
                1,
                0.4,
                facecolor=self.colors["measurement"],
                edgecolor="black",
            )
            ax.add_patch(rect)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    def _draw_fusion_layer(self, ax, encoder_configs: list[dict], fusion_method: str):
        """Draw the fusion layer representation"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 2)

        # Input representations
        n_encoders = len(encoder_configs)
        input_width = 8 / n_encoders

        for i, config in enumerate(encoder_configs):
            x_start = 1 + i * input_width
            rect = Rectangle(
                (x_start, 1.2),
                input_width * 0.8,
                0.3,
                facecolor="lightblue",
                edgecolor="black",
            )
            ax.add_patch(rect)
            ax.text(
                x_start + input_width * 0.4,
                1.35,
                f"{config['n_qubits']}D",
                ha="center",
                va="center",
                fontsize=10,
            )

        # Fusion operation
        fusion_rect = Rectangle((4, 0.5), 2, 0.4, facecolor="orange", edgecolor="black")
        ax.add_patch(fusion_rect)
        ax.text(
            5,
            0.7,
            fusion_method.upper(),
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=10,
        )

        # Arrows
        for i in range(n_encoders):
            x_pos = 1 + i * input_width + input_width * 0.4
            ax.arrow(
                x_pos,
                1.2,
                0,
                -0.2,
                head_width=0.1,
                head_length=0.05,
                fc="black",
                ec="black",
            )

        ax.arrow(
            5, 0.5, 0, -0.2, head_width=0.1, head_length=0.05, fc="black", ec="black"
        )

        ax.set_title(f"Fusion Layer ({fusion_method})", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)


class AmplitudeEncodingModule(tq.QuantumModule):
    """
    Amplitude encoding module for embedding classical feature vectors
    Implements |ψ(x)⟩ = Σᵢ xᵢ |i⟩ encoding from classical data
    """

    def __init__(self, n_qubits: int, normalize: bool = True):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_amplitudes = 2**n_qubits
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode classical vector into quantum amplitudes
        Args:
            x: Classical feature vector (batch_size, feature_dim)
        Returns:
            Processed feature vector for quantum device
        """
        batch_size, feature_dim = x.shape

        # Ensure feature dimension matches number of amplitudes
        if feature_dim > self.n_amplitudes:
            # Truncate if too many features
            amplitudes = x[:, : self.n_amplitudes]
        elif feature_dim < self.n_amplitudes:
            # Pad with zeros if too few features
            padding = torch.zeros(
                batch_size,
                self.n_amplitudes - feature_dim,
                device=x.device,
                dtype=x.dtype,
            )
            amplitudes = torch.cat([x, padding], dim=1)
        else:
            amplitudes = x

        # Normalize to ensure valid quantum state (|ψ|² = 1)
        if self.normalize:
            amplitudes = torch.nn.functional.normalize(amplitudes, p=2, dim=1)
        #  the state can be "loaded" directly by setting the amplitudes, bypassing
        #  the need for explicit quantum circuit construction with initialization gates (quantum simulation)
        return amplitudes


class SimulatedQuantumEncoder(tq.QuantumModule):
    """
    Single simulated quantum encoder (sQE) block
    Implements noiseless state vector simulation with parameterized gates
    """

    def __init__(self, n_qubits: int, n_layers: int, connectivity: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.connectivity = connectivity

        # Amplitude encoding module
        self.amplitude_encoder = AmplitudeEncodingModule(n_qubits)

        # Parameterized quantum layers
        self.quantum_layers = nn.ModuleList()
        self.cnot_connection_patterns = []  # Store CNOT patterns separately

        for layer_idx in range(n_layers):
            # RY rotation gates for each qubit
            ry_gates = tq.QuantumModuleList()
            for _qubit_idx in range(n_qubits):
                ry_gates.append(tq.RY(has_params=True, trainable=True))

            # CNOT gates based on connectivity pattern
            cnot_gates = tq.QuantumModuleList()
            cnot_pairs = []

            # Generate connectivity pattern (increasing connectivity as in Figure 2)
            effective_connectivity = min(connectivity * (layer_idx + 1), n_qubits - 1)

            for i in range(n_qubits):
                for j in range(
                    max(0, i - effective_connectivity),
                    min(n_qubits, i + effective_connectivity + 1),
                ):
                    if i != j and (i, j) not in cnot_pairs and (j, i) not in cnot_pairs:
                        cnot_pairs.append((i, j))
                        cnot_gates.append(tq.CNOT())

            # Store this layer's gates and connections
            layer_dict = {"ry_gates": ry_gates, "cnot_gates": cnot_gates}
            self.quantum_layers.append(nn.ModuleDict(layer_dict))
            self.cnot_connection_patterns.append(cnot_pairs)

        # Exact measurement module (noiseless simulation)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through simulated quantum encoder
        Args:
            x: Classical feature vector from LLM layer
        Returns:
            Exact expectation values (latent representation)
        """
        batch_size = x.shape[0]

        # Step 1: Amplitude encoding (preprocess features)
        amplitudes = self.amplitude_encoder(x)

        # Step 2: Create quantum device and set initial state
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=batch_size, device=x.device)

        # For amplitude encoding, manually set the quantum state
        # TODO: StatePreparation gates
        qdev.reset_states(bsz=batch_size)

        # Apply amplitude encoding through parameterized gates
        # This is a workaround since direct state setting may not be available
        for i in range(self.n_qubits):
            if i < amplitudes.shape[1]:
                # Use RY gates to encode amplitude information
                tq.RY()(qdev, wires=i, params=amplitudes[:, i] * np.pi)

        # Step 3: Apply parameterized quantum layers
        for layer_idx, layer_gates in enumerate(self.quantum_layers):
            ry_gates = layer_gates["ry_gates"]
            cnot_gates = layer_gates["cnot_gates"]
            cnot_pairs = self.cnot_connection_patterns[layer_idx]

            # Apply RY rotations
            for qubit_idx, ry_gate in enumerate(ry_gates):
                ry_gate(qdev, wires=qubit_idx)

            # Apply CNOT gates
            for (control, target), cnot_gate in zip(cnot_pairs, cnot_gates):
                cnot_gate(qdev, wires=[control, target])

        # Step 4: Exact expectation value computation (noiseless)
        expectation_values = self.measure(qdev)

        return expectation_values

    def visualize_circuit(self) -> plt.Figure:
        """Visualize this encoder's circuit"""
        visualizer = QuantumCircuitVisualizer()
        config = {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "connectivity": self.connectivity,
            "encoding_type": "amplitude",
        }
        return visualizer.draw_circuit(
            config, f"Simulated Quantum Encoder ({self.n_qubits} qubits)"
        )


class MultiEncoderSimulatedQuantumBlock(nn.Module):
    """
    Multi-encoder design for studying scaling behavior of sPQC
    Allows manageable increase in sPQC complexity as PQC grows
    """

    def __init__(
        self,
        encoder_configs: list[dict],
        fusion_method: str = "concatenate",
        feature_dim: int = None,
    ):
        """
        Args:
            encoder_configs: list of encoder configurations
                Each config: {"n_qubits": int, "n_layers": int, "connectivity": int}
            fusion_method: How to combine encoder outputs ("concatenate", "average", "weighted")
            feature_dim: Input feature dimension (for splitting across encoders)
        """
        super().__init__()

        self.encoder_configs = encoder_configs
        self.fusion_method = fusion_method
        self.feature_dim = feature_dim
        self.n_encoders = len(encoder_configs)

        # Create multiple simulated quantum encoders
        self.encoders = nn.ModuleList()
        self.encoder_input_dims = []

        total_qubits = sum(config["n_qubits"] for config in encoder_configs)

        for _i, config in enumerate(encoder_configs):
            encoder = SimulatedQuantumEncoder(
                n_qubits=config["n_qubits"],
                n_layers=config["n_layers"],
                connectivity=config.get("connectivity", 1),
            )
            self.encoders.append(encoder)

            # Calculate input dimension for each encoder
            if feature_dim:
                # Distribute features across encoders based on their qubit count
                encoder_feature_dim = int(
                    feature_dim * config["n_qubits"] / total_qubits
                )
                self.encoder_input_dims.append(encoder_feature_dim)
            else:
                # Each encoder gets full input
                self.encoder_input_dims.append(2 ** config["n_qubits"])

        # Fusion parameters for weighted combination
        if fusion_method == "weighted":
            self.fusion_weights = nn.Parameter(torch.ones(self.n_encoders))

        # Calculate output dimension
        if fusion_method == "concatenate":
            self.output_dim = sum(config["n_qubits"] for config in encoder_configs)
        else:
            # For average/weighted, output dim is max of encoder outputs
            self.output_dim = max(config["n_qubits"] for config in encoder_configs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-encoder system
        Args:
            x: Input feature vector from LLM
        Returns:
            Fused latent representation
        """
        encoder_outputs = []

        if self.feature_dim and len(self.encoder_input_dims) > 1:
            # Split input across encoders
            start_idx = 0
            for _i, (encoder, input_dim) in enumerate(
                zip(self.encoders, self.encoder_input_dims)
            ):
                end_idx = start_idx + input_dim
                encoder_input = x[:, start_idx:end_idx]
                encoder_output = encoder(encoder_input)
                encoder_outputs.append(encoder_output)
                start_idx = end_idx
        else:
            # Each encoder processes full input
            for encoder in self.encoders:
                encoder_output = encoder(x)
                encoder_outputs.append(encoder_output)

        # Fuse encoder outputs
        if self.fusion_method == "concatenate":
            # Concatenate all encoder outputs
            fused_output = torch.cat(encoder_outputs, dim=1)

        elif self.fusion_method == "average":
            # Average encoder outputs (pad shorter ones with zeros)
            max_dim = max(output.shape[1] for output in encoder_outputs)
            padded_outputs = []

            for output in encoder_outputs:
                if output.shape[1] < max_dim:
                    padding = torch.zeros(
                        output.shape[0], max_dim - output.shape[1], device=output.device
                    )
                    padded_output = torch.cat([output, padding], dim=1)
                else:
                    padded_output = output[:, :max_dim]
                padded_outputs.append(padded_output)

            fused_output = torch.stack(padded_outputs, dim=0).mean(dim=0)

        elif self.fusion_method == "weighted":
            # Weighted combination of encoder outputs
            max_dim = max(output.shape[1] for output in encoder_outputs)
            padded_outputs = []

            for output in encoder_outputs:
                if output.shape[1] < max_dim:
                    padding = torch.zeros(
                        output.shape[0], max_dim - output.shape[1], device=output.device
                    )
                    padded_output = torch.cat([output, padding], dim=1)
                else:
                    padded_output = output[:, :max_dim]
                padded_outputs.append(padded_output)

            # Apply learned weights
            weights = torch.softmax(self.fusion_weights, dim=0)
            weighted_outputs = []
            for i, output in enumerate(padded_outputs):
                weighted_outputs.append(weights[i] * output)

            fused_output = torch.stack(weighted_outputs, dim=0).sum(dim=0)

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return fused_output


class ScalabilityStudyModel(nn.Module):
    """
    Complete model for studying scaling behavior with multi-encoder design
    """

    def __init__(
        self,
        llm_output_dim: int,
        encoder_configs: list[dict],
        quantum_pqc_config: dict,
        fusion_method: str = "concatenate",
        n_classes: int = 2,
    ):
        super().__init__()

        self.llm_output_dim = llm_output_dim
        self.n_classes = n_classes

        # Multi-encoder simulated quantum block
        self.multi_encoder_sqb = MultiEncoderSimulatedQuantumBlock(
            encoder_configs=encoder_configs,
            fusion_method=fusion_method,
            feature_dim=llm_output_dim,
        )

        # Angle encoding for quantum PQC
        self.angle_encoder = AngleEncodingModule(quantum_pqc_config["n_qubits"])

        # Actual quantum PQC
        self.quantum_pqc = QuantumPQC(
            n_qubits=quantum_pqc_config["n_qubits"],
            n_layers=quantum_pqc_config["n_layers"],
            connectivity=quantum_pqc_config.get("connectivity", 2),
        )

        # Output layer
        self.output_layer = nn.Linear(quantum_pqc_config["n_qubits"], n_classes)

        # Quantum device for PQC
        self.quantum_qdev = tq.QuantumDevice(n_wires=quantum_pqc_config["n_qubits"])

        # Store configuration for analysis
        self.encoder_configs = encoder_configs
        self.quantum_pqc_config = quantum_pqc_config

    def forward(self, llm_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete scalability study model
        Args:
            llm_features: Features from last layer of LLM
        Returns:
            Classification logits
        """
        batch_size = llm_features.shape[0]

        # Multi-encoder simulated quantum processing
        latent_representation = self.multi_encoder_sqb(llm_features)

        # Prepare for quantum PQC
        n_quantum_qubits = self.quantum_pqc_config["n_qubits"]
        if latent_representation.shape[1] > n_quantum_qubits:
            # Use first n_quantum_qubits components
            quantum_input = latent_representation[:, :n_quantum_qubits]
        elif latent_representation.shape[1] < n_quantum_qubits:
            # Pad with zeros
            padding = torch.zeros(
                batch_size,
                n_quantum_qubits - latent_representation.shape[1],
                device=latent_representation.device,
            )
            quantum_input = torch.cat([latent_representation, padding], dim=1)
        else:
            quantum_input = latent_representation

        # Reset quantum device
        self.quantum_qdev.reset_states(batch_size)

        # Angle encoding
        self.quantum_qdev = self.angle_encoder(self.quantum_qdev, quantum_input)

        # Quantum PQC
        quantum_output = self.quantum_pqc(self.quantum_qdev)

        # Final classification
        logits = self.output_layer(quantum_output)

        return logits

    def get_encoder_complexity_analysis(self) -> dict:
        """
        Analyze the complexity scaling of the multi-encoder system
        """
        analysis = {
            "total_encoders": len(self.encoder_configs),
            "encoder_details": [],
            "total_simulated_qubits": 0,
            "total_parameters": 0,
            "complexity_distribution": {},
        }

        for i, config in enumerate(self.encoder_configs):
            n_qubits = config["n_qubits"]
            n_layers = config["n_layers"]

            # Estimate parameters per encoder
            params_per_encoder = n_qubits * n_layers  # RY parameters

            encoder_info = {
                "encoder_id": i,
                "n_qubits": n_qubits,
                "n_layers": n_layers,
                "connectivity": config.get("connectivity", 1),
                "state_space_size": 2**n_qubits,
                "estimated_parameters": params_per_encoder,
            }

            analysis["encoder_details"].append(encoder_info)
            analysis["total_simulated_qubits"] += n_qubits
            analysis["total_parameters"] += params_per_encoder

        # Complexity distribution
        qubit_counts = [config["n_qubits"] for config in self.encoder_configs]
        unique_counts = list(set(qubit_counts))

        for count in unique_counts:
            analysis["complexity_distribution"][f"{count}_qubits"] = qubit_counts.count(
                count
            )

        return analysis

    def visualize_multi_encoder_architecture(self) -> plt.Figure:
        """
        Visualize the complete multi-encoder architecture
        """
        visualizer = QuantumCircuitVisualizer()
        return visualizer.draw_multi_encoder_architecture(
            encoder_configs=self.encoder_configs,
            quantum_config=self.quantum_pqc_config,
            fusion_method=self.multi_encoder_sqb.fusion_method,
        )


# Import previous classes that are still needed
class AngleEncodingModule(tq.QuantumModule):
    """Angle encoding module for loading latent vector into QPU"""

    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits

        # RY rotation gates for angle encoding
        self.ry_gates = tq.QuantumModuleList()
        for _ in range(n_qubits):
            self.ry_gates.append(tq.RY(has_params=False, trainable=False))

    def forward(
        self, qdev: tq.QuantumDevice, latent_vector: torch.Tensor
    ) -> tq.QuantumDevice:
        """Encode latent vector using angle encoding"""
        batch_size = latent_vector.shape[0]

        # Reset quantum device to |0...0> state
        qdev.reset_states(batch_size)

        # Apply RY rotations with angles from latent vector
        for i, ry_gate in enumerate(self.ry_gates):
            if i < latent_vector.shape[1]:
                # Use latent vector values as rotation angles
                angles = latent_vector[:, i]
                ry_gate(qdev, wires=i, params=angles)

        return qdev


class QuantumPQC(tq.QuantumModule):
    """Parameterized Quantum Circuit (PQC) for actual QPU execution"""

    def __init__(self, n_qubits: int, n_layers: int, connectivity: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.connectivity = connectivity

        # Create parameterized layers
        self.quantum_layers = tq.QuantumModuleList()
        self.cnot_connection_patterns = []  # Store CNOT patterns separately

        for _layer_idx in range(n_layers):
            # Create Ui blocks - each block contains CNOT + RY gates
            layer_blocks = tq.QuantumModuleList()
            cnot_pairs = []

            # Generate CNOT pairs based on connectivity pattern
            for i in range(n_qubits):
                for j in range(
                    max(0, i - connectivity), min(n_qubits, i + connectivity + 1)
                ):
                    if i != j and (i, j) not in cnot_pairs and (j, i) not in cnot_pairs:
                        # Create a Ui block: CNOT followed by RY rotation on target only
                        ui_block = tq.QuantumModuleList()
                        ui_block.append(tq.CNOT())  # CNOT gate
                        ui_block.append(
                            tq.RY(has_params=True, trainable=True)
                        )  # RY on target only

                        layer_blocks.append(ui_block)
                        cnot_pairs.append((i, j))

            self.quantum_layers.append(layer_blocks)
            self.cnot_connection_patterns.append(cnot_pairs)
        print(
            f"\n ############### \n Quantum Layers {self.quantum_layers} \n ###############"
        )
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        """Forward pass through quantum PQC"""

        # Apply parameterized layers as sequential Ui blocks
        for layer_idx, ui_blocks in enumerate(self.quantum_layers):
            cnot_pairs = self.cnot_connection_patterns[layer_idx]

            # Apply each Ui block sequentially: CNOT followed by RY rotation on target only
            for ui_block, (control, target) in zip(ui_blocks, cnot_pairs):
                cnot_gate, ry_target = ui_block

                # Apply CNOT gate first
                cnot_gate(qdev, wires=[control, target])

                # Then apply RY rotation to target qubit only
                ry_target(qdev, wires=target)

        # Measure
        measurement_output = self.measure(qdev)

        return measurement_output


# Utility functions for scaling studies
def create_scaling_study_configs() -> list[dict]:
    """
    Create encoder configurations for studying scaling behavior
    Optimized for 768-dimensional LLM features
    """
    configs = [
        # Small encoders (2^2=4, 2^3=8 amplitudes)
        {"n_qubits": 2, "n_layers": 2, "connectivity": 1},
        {"n_qubits": 3, "n_layers": 2, "connectivity": 1},
        # Medium encoders (2^4=16, 2^5=32 amplitudes)
        {"n_qubits": 4, "n_layers": 3, "connectivity": 2},
        {"n_qubits": 5, "n_layers": 3, "connectivity": 2},
        # Large encoders (2^6=64, 2^7=128 amplitudes)
        {"n_qubits": 6, "n_layers": 4, "connectivity": 3},
        {"n_qubits": 7, "n_layers": 4, "connectivity": 3},
        # Very large encoders (2^8=256, 2^9=512 amplitudes)
        # These can handle significant portions of the 768 features
        {"n_qubits": 8, "n_layers": 5, "connectivity": 4},
        {"n_qubits": 9, "n_layers": 5, "connectivity": 4},
    ]

    return configs


def run_scaling_experiment(
    llm_output_dim: int = 768, n_classes: int = 10, fusion_methods: list[str] = None
) -> dict:
    """
    Run scaling behavior experiments with different encoder configurations
    """
    if fusion_methods is None:
        fusion_methods = ["concatenate", "average", "weighted"]

    encoder_configs = create_scaling_study_configs()
    quantum_pqc_config = {"n_qubits": 6, "n_layers": 3, "connectivity": 2}

    results = {}

    for fusion_method in fusion_methods:
        print(f"\n=== Testing fusion method: {fusion_method} ===")

        # Create model
        model = ScalabilityStudyModel(
            llm_output_dim=llm_output_dim,
            encoder_configs=encoder_configs,
            quantum_pqc_config=quantum_pqc_config,
            fusion_method=fusion_method,
            n_classes=n_classes,
        )

        # Analyze complexity
        complexity_analysis = model.get_encoder_complexity_analysis()

        # Test with dummy data
        dummy_input = torch.randn(4, llm_output_dim)

        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        results[fusion_method] = {
            "complexity_analysis": complexity_analysis,
            "output_shape": output.shape,
            "model_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }

        print(
            f"Total simulated qubits: {complexity_analysis['total_simulated_qubits']}"
        )
        print(f"Output dimension: {model.multi_encoder_sqb.output_dim}")
        print(f"Model parameters: {results[fusion_method]['model_parameters']}")

    return results


# Example usage with visualization
if __name__ == "__main__":
    # Run scaling experiments
    scaling_results = run_scaling_experiment()

    # Print detailed results
    print("\n" + "=" * 60)
    print("SCALING EXPERIMENT RESULTS")
    print("=" * 60)

    for fusion_method, results in scaling_results.items():
        print(f"\nFusion Method: {fusion_method}")
        print("-" * 30)

        analysis = results["complexity_analysis"]
        print(f"Total Encoders: {analysis['total_encoders']}")
        print(f"Total Simulated Qubits: {analysis['total_simulated_qubits']}")
        print(f"Total Parameters: {analysis['total_parameters']}")
        print(f"Model Parameters: {results['model_parameters']}")
        print(f"Complexity Distribution: {analysis['complexity_distribution']}")

        for encoder_detail in analysis["encoder_details"]:
            print(
                f"  Encoder {encoder_detail['encoder_id']}: "
                f"{encoder_detail['n_qubits']} qubits, "
                f"{encoder_detail['n_layers']} layers, "
                f"state space: 2^{encoder_detail['n_qubits']} = {encoder_detail['state_space_size']}"
            )

    # =================================================================
    # VISUALIZATION EXAMPLES
    # =================================================================

    print("\n" + "=" * 60)
    print("CIRCUIT VISUALIZATION EXAMPLES")
    print("=" * 60)

    # 1. Visualize individual encoder
    print("\n1. Individual Simulated Quantum Encoder:")
    single_encoder = SimulatedQuantumEncoder(n_qubits=4, n_layers=3, connectivity=2)
    fig1 = single_encoder.visualize_circuit()
    plt.show()

    # 2. Visualize quantum PQC
    print("\n2. Quantum PQC Circuit:")
    visualizer = QuantumCircuitVisualizer()
    pqc_config = {
        "n_qubits": 6,
        "n_layers": 3,
        "connectivity": 2,
        "encoding_type": "angle",
    }
    fig2 = visualizer.draw_circuit(pqc_config, "Quantum PQC (Actual QPU)")
    plt.show()

    # 3. Visualize complete multi-encoder architecture
    print("\n3. Complete Multi-Encoder Architecture:")
    encoder_configs = create_scaling_study_configs()[
        :4
    ]  # Use first 4 for cleaner visualization
    quantum_config = {"n_qubits": 6, "n_layers": 3, "connectivity": 2}

    model = ScalabilityStudyModel(
        llm_output_dim=768,
        encoder_configs=encoder_configs,
        quantum_pqc_config=quantum_config,
        fusion_method="concatenate",
    )

    fig3 = model.visualize_multi_encoder_architecture()
    plt.show()

    # 4. Compare different connectivity patterns
    print("\n4. Connectivity Pattern Comparison:")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    connectivities = [1, 2, 3]
    for i, conn in enumerate(connectivities):
        config = {
            "n_qubits": 5,
            "n_layers": 3,
            "connectivity": conn,
            "encoding_type": "amplitude",
        }

        # Manual drawing for subplot
        ax = axes[i]
        vis = QuantumCircuitVisualizer()
        # Since draw_circuit creates its own figure, we'll create a simplified version
        vis._draw_mini_circuit(ax, config, f"Connectivity = {conn}")

    plt.suptitle("Connectivity Pattern Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # 5. Save circuit diagrams
    print("\n5. Saving Circuit Diagrams:")

    # Save individual encoder circuit
    fig1.savefig("simulated_quantum_encoder.png", dpi=300, bbox_inches="tight")
    print("   - Saved: simulated_quantum_encoder.png")

    # Save quantum PQC circuit
    fig2.savefig("quantum_pqc_circuit.png", dpi=300, bbox_inches="tight")
    print("   - Saved: quantum_pqc_circuit.png")

    # Save multi-encoder architecture
    fig3.savefig("multi_encoder_architecture.png", dpi=300, bbox_inches="tight")
    print("   - Saved: multi_encoder_architecture.png")

    # Example of individual encoder usage
    print("\n" + "=" * 60)
    print("INDIVIDUAL ENCODER EXAMPLE")
    print("=" * 60)

    # Create single encoder
    single_encoder = SimulatedQuantumEncoder(n_qubits=4, n_layers=3, connectivity=2)

    # Test with LLM features
    llm_features = torch.randn(2, 768)  # batch_size=2, feature_dim=768
    latent_repr = single_encoder(llm_features)

    print(f"Input shape: {llm_features.shape}")
    print(f"Latent representation shape: {latent_repr.shape}")
    print(f"Latent values: {latent_repr}")


def demonstrate_circuit_visualization():
    """
    Comprehensive demonstration of circuit visualization capabilities
    """
    print("=== Quantum Circuit Visualization Demo ===\n")

    # Initialize visualizer
    visualizer = QuantumCircuitVisualizer()

    # 1. Simple amplitude encoding circuit
    simple_config = {
        "n_qubits": 3,
        "n_layers": 2,
        "connectivity": 1,
        "encoding_type": "amplitude",
    }

    print("1. Drawing simple amplitude encoding circuit...")
    fig1 = visualizer.draw_circuit(simple_config, "Simple Amplitude Encoding Circuit")
    plt.show()

    # 2. Complex angle encoding circuit
    complex_config = {
        "n_qubits": 6,
        "n_layers": 4,
        "connectivity": 3,
        "encoding_type": "angle",
    }

    print("2. Drawing complex angle encoding circuit...")
    fig2 = visualizer.draw_circuit(complex_config, "Complex Angle Encoding Circuit")
    plt.show()

    # 3. Multi-encoder architecture
    encoder_configs = [
        {"n_qubits": 3, "n_layers": 2, "connectivity": 1},
        {"n_qubits": 4, "n_layers": 3, "connectivity": 2},
        {"n_qubits": 5, "n_layers": 3, "connectivity": 2},
    ]

    quantum_config = {"n_qubits": 6, "n_layers": 3, "connectivity": 2}

    print("3. Drawing multi-encoder architecture...")
    fig3 = visualizer.draw_multi_encoder_architecture(
        encoder_configs, quantum_config, fusion_method="concatenate"
    )
    plt.show()

    return fig1, fig2, fig3
