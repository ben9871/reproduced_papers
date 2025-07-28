

class Hyperparams:
    def __init__(self, num_runs=3, num_epochs=100, batch_size=32, lr=0.02, betas=[0.9, 0.999], weight_decay=0.0,
                 train_circuit=False, scale_type="learned", circuit="mzi", no_bunching=False, optimizer="adam",
                 shuffle_train=True):
        self.num_runs = num_runs
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.train_circuit = train_circuit
        self.scale_type = scale_type
        self.circuit = circuit
        self.no_bunching = no_bunching
        self.optimizer = optimizer
        self.shuffle_train = shuffle_train

