import math
import sys
import torch
import torch.nn as nn
import torchvision
import perceval as pcvl
from merlin import OutputMappingStrategy, QuantumLayer
from training_utils import InfoNCELoss

sys.path.append("./qnn")
from qnet import QNet


def create_quantum_circuit(modes=10, feature_size=10):
    # first trainable circuit
    pre_circuit = pcvl.GenericInterferometer(
        modes,
        lambda i: (
            pcvl.BS()  # theta=pcvl.P(f"bs_1_{i}")
            .add(0, pcvl.PS(pcvl.P(f"phase_train_1_{i}")))
            .add(0, pcvl.BS())  # theta=pcvl.P(f"bs_1_{i}")
            .add(0, pcvl.PS(pcvl.P(f"phase_train_2_{i}")))
        ),
    )
    # data encoding in phase shifters (sandwhich)
    var = pcvl.Circuit(modes)
    for k in range(0, feature_size):
        var.add(k % modes, pcvl.PS(pcvl.P(f"feature-{k}")))

    # second trainable circuit
    post_circuit = pcvl.GenericInterferometer(
        modes,
        lambda i: (
            pcvl.BS()  # theta=pcvl.P(f"bs_1_{i}")
            .add(0, pcvl.PS(pcvl.P(f"phase_train_3_{i}")))
            .add(0, pcvl.BS())  # theta=pcvl.P(f"bs_1_{i}")
            .add(0, pcvl.PS(pcvl.P(f"phase_train_4_{i}")))
        ),
    )

    circuit = pcvl.Circuit(modes)

    circuit.add(0, pre_circuit, merge=True)
    circuit.add(0, var, merge=True)
    circuit.add(0, post_circuit, merge=True)

    return circuit


def initialize_resnet_kaiming(model):
    """Apply Kaiming initialization to ResNet model"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)



class QSSL(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        framework_used = "Quantum (MerLin)" if args.merlin else "Classical ResNet18"
        framework_used = "Quantum (Qiskit)" if args.qiskit else framework_used
        print(f"\n Defining the SSL model with \n -{framework_used} \n - ")
        # backbone
        self.width = args.width
        # backbone with FC = Identity
        self.backbone = torchvision.models.resnet18(pretrained=False, zero_init_residual=True)
        # initialisation following Jaderberg et al.
        initialize_resnet_kaiming(self.backbone)

        backbone_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        # compressing layer to map to self.width dimension
        self.comp = nn.Linear(backbone_features, self.width)

        # building the representation network
        self.merlin = args.merlin
        self.qiskit = args.qiskit
        self.batch_norm = args.batch_norm
        self.bn = nn.BatchNorm2d(self.width)

        # photonic circuit
        if self.merlin:
            print("\n -> Building the quantum representation network with MerLin")
            self.modes = args.modes
            self.no_bunching = args.no_bunching
            self.circuit = create_quantum_circuit(
                modes=self.modes, feature_size=self.width
            )

            input_state = [(i + 1) % 2 for i in range(args.modes)]
            print(f"input state: {input_state} and no bunching: {self.no_bunching}")

            self.representation_network = QuantumLayer(
                input_size=self.width,
                output_size=None,  # math.comb(args.modes+photon_count-1,photon_count), # but we do not use it
                circuit=self.circuit,
                trainable_parameters=[
                    p.name
                    for p in self.circuit.get_parameters()
                    if not p.name.startswith("feature")
                ],
                input_parameters=["feature"],
                input_state=input_state,
                no_bunching=self.no_bunching,
                output_mapping_strategy=OutputMappingStrategy.NONE,
            )

            self.rep_net_output_size = self.representation_network.output_size

        elif self.qiskit:
            #from https://github.com/bjader/QSSL
            print("\n -> Building the quantum representation network with Qiskit")
            self.representation_network = QNet(n_qubits=self.width, encoding=args.encoding, ansatz_type=args.q_ansatz,
                                          layers=args.layers, sweeps_per_layer=args.q_sweeps,
                                          activation_function_type=args.activation, shots=args.shots,
                                          backend_type=args.q_backend, save_statevectors=args.save_dhs)
            self.rep_net_output_size = self.width
        else:
            mapping_paper = True
            print("\n -> Building the classical representation network ")
            if not mapping_paper:
                ### TO COMPARE TO MERLIN ###
                # we want to create a classical representation network with similar # of parameters to the QLayer with 10 modes, 5 photons
                # compute the number of parameters in a quantum network given 10 modes and 5 photons
                circuit = create_quantum_circuit(modes=10, feature_size=8)
                nb_trainable_params = len(
                    [
                        p.name
                        for p in circuit.get_parameters()
                        if not p.name.startswith("feature")
                    ]
                )
                output_size = (
                    math.comb(10, 5) if args.no_bunching else math.comb(10 + 5 - 1, 5)
                )
                total_parameters = (
                    nb_trainable_params + output_size * self.width
                )  # circuit + first layer of the proj
                # number of parameters in classical repnet
                classical_params = (
                    self.width * self.width + self.width
                ) * 3  # rep + first layer of the proj
                # difference
                diff = total_parameters - classical_params
                print(
                    f"--> Difference would be: {diff} (for {total_parameters} parameters for QNN)"
                )

                layers = []
                for _i in range(2):
                    layers.append(nn.Linear(self.width, self.width, bias=True))
                    layers.append(nn.LeakyReLU())
                # add another layer + activation to increase MLP size (TODO: have a more regular MLP)
                catching_output_size = int(diff / self.width - 1)
                layers.append(nn.Linear(self.width, catching_output_size, bias=True))
                layers.append(nn.LeakyReLU())

                self.representation_network = nn.Sequential(*layers)
                print(
                    f" Now in repnet = {sum(p.numel() for p in self.representation_network.parameters())}"
                )
                self.rep_net_output_size = catching_output_size

            else:
                ### TO MAP THE PAPER ###
                layers = []
                for i in range(args.layers):
                    layers.append(nn.Linear(args.width, args.width, bias=True))
                    layers.append(nn.LeakyReLU())
                self.representation_network = nn.Sequential(*layers)
                self.rep_net_output_size = args.width
        # self.fc = nn.Linear(self.rep_net_output_size, args.classes)

        self.loss_dim = args.loss_dim
        # projector to the loss space
        self.proj = nn.Sequential(
            nn.Linear(self.rep_net_output_size, self.width),
            nn.BatchNorm1d(self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.loss_dim),
        )

        self.normalize = nn.Sigmoid()
        self.temperature = args.temperature
        # contrastive loss
        self.criterion = InfoNCELoss(temperature=self.temperature)

    def forward(self, y1, y2):
        # Encoder and MLP layer for compression
        x1 = self.comp(self.backbone(y1))
        x2 = self.comp(self.backbone(y2))
        # print(f"\n After encoder x1 = {x1}")
        # BatchNorm if needed
        if self.batch_norm:
            x1 = self.bn(x1)
            x2 = self.bn(x2)
            # print(f"\n After Batch Norm x1 = {x1}")
        # Sigmoid before Representation Network
        x1 = torch.sigmoid(x1)
        x2 = torch.sigmoid(x2)
        # print(f"\n After sigmoid Norm x1 = {x1}")
        # in the original code they use x = x * np.pi
        z1 = self.representation_network(x1)
        z2 = self.representation_network(x2)
        # print(f"\n After representation network z1 = {z1}")
        # projection to loss space
        z1 = self.proj(z1)
        z2 = self.proj(z2)
        # print(f"\n After projection z1 = {z1}")

        # L2 normalize features before contrastive loss
        z1 = nn.functional.normalize(z1, p=2, dim=1)
        z2 = nn.functional.normalize(z2, p=2, dim=1)

        # Contrastive loss on the concatenated features (along batch dimension)
        z = torch.cat((z1, z2), dim=0)
        loss = self.criterion(z)
        # print(f"\n --- Loss = {loss} --- \n")
        return loss