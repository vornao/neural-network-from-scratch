import numpy as np
from src.layers import Layer


class Callback:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented in the child class")


# TODO: restore best weights
class EarlyStopping(Callback):
    '''
    Early stopping callback
    patience: number of epochs to wait before stopping
    '''
    def __init__(self, patience, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.best_loss = np.Inf
        self.counter = 0
        self.best_weights = []
        self.best_biases = []


    def __call__(self, network):

        if network.get_loss_value() < self.best_loss:
            self.best_loss = network.get_loss_value()
            self.counter = 0

            del self.best_weights
            del self.best_biases
            self.best_weights = []
            self.best_biases = []
            
            for layer in network.layers[1:]:
                self.best_weights.append(np.copy(layer.W))
                self.best_biases.append(np.copy(layer.bias))

        else:
            self.counter += 1

        if self.counter >= self.patience:
            network.training = False

            for i, layer in enumerate(network.layers[1:]):
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