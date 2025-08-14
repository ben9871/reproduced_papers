import numpy as np
import torch


class Hyperparameters:
    def __init__(
        self,
        n_samples=200,
        noise=0.2,
        random_state=42,
        scaling="Standard",
        test_prop=0.2,
        num_photon=10,
        output_mapping_strategy="NONE",
        no_bunching=False,
        circuit="mzi",
        batch_size=30,
        optimizer="adam",
        learning_rate=0.001,
        betas=(0.9, 0.999),
        weight_decay=0.02,
        num_epochs=50,
        c=1.0,
        r=1,
        gamma=1,
        train_hybrid_model=True,
        pre_encoding_scaling=1.0 / np.pi,
        z_q_matrix_scaling="1/sqrt(R)",
        hybrid_model_data="Default",
    ):
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state
        self.scaling = scaling
        self.test_prop = test_prop
        self.num_photon = num_photon
        self.output_mapping_strategy = output_mapping_strategy
        self.no_bunching = no_bunching
        self.circuit = circuit
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.c = c
        self.r = r
        self.gamma = gamma
        self.train_hybrid_model = train_hybrid_model
        self.pre_encoding_scaling = pre_encoding_scaling
        self.z_q_matrix_scaling = z_q_matrix_scaling
        self.set_z_q_matrix_scaling_value()
        self.hybrid_model_data = hybrid_model_data

        self.w = None
        self.b = None

    def set_z_q_matrix_scaling_value(self):
        if isinstance(self.z_q_matrix_scaling, str):
            if self.z_q_matrix_scaling == "1/sqrt(R)":
                self.z_q_matrix_scaling_value = torch.tensor(1.0 / np.sqrt(self.r))
            elif self.z_q_matrix_scaling == "sqrt(R)":
                self.z_q_matrix_scaling_value = torch.tensor(np.sqrt(self.r))
            elif self.z_q_matrix_scaling == "sqrt(R) + 3":
                self.z_q_matrix_scaling_value = torch.tensor(np.sqrt(self.r) + 3)
            else:
                raise ValueError('z_q_matrix_scaling must be "1/sqrt(R)" or "sqrt(R)"')
        else:
            self.z_q_matrix_scaling_value = torch.tensor(self.z_q_matrix_scaling)

    def set_random(self, w, b):
        """
        Set values for random weights and biases. That is to keep the same values for the quantum and classical
        methods in order to fairly compare the two.
        """
        self.w = w
        self.b = b
        return

    def set_gamma(self, gamma):
        self.gamma = gamma
        return

    def set_r(self, r):
        self.r = r
        self.set_z_q_matrix_scaling_value()
        return
