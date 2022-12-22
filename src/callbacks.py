import numpy as np
from src.network import Network
from src.layers import Layer


class Callback:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented in the child class")


# TODO: restore best weights
class EarlyStopping(Callback):
    def __init__(self, patience, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.best_loss = np.Inf
        self.counter = 0
        self.best_weights = []
        self.best_biases = []


    def __call__(self, network):

        if network.loss < self.best_loss:
            self.best_loss = network.loss
            self.counter = 0

            for layer in network.layers:
                self.best_weights.append(np.copy(layer.W))
                self.best_biases.append(np.copy(layer.bias))
            # todo save bias

        else:
            self.counter += 1

        if self.counter >= self.patience:
            network.training = False

            for i, layer in enumerate(network.layers):
                layer.W = self.best_weights[i]
                layer.bias = self.best_biases[i]
            print("Early stopping")

            return True


class RelativeEarlyStopping(Callback):

        def __init__(self, patience, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.patience = patience
            self.best_loss = np.Inf
            self.counter = 0
            self.best_weights = []

        def __call__(self, *args, **kwargs):
            raise NotImplementedError("This method should be implemented in the child class")