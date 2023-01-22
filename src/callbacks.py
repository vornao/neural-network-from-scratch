# Author: Giacomo Lagomarsini - Luca Miglior - Leonardo Stoppani
# Date: 2023-01-23
# License: MIT

import numpy as np
from src.layers import Layer


class Callback:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("-- I lay in the v o i d --")


class EarlyStopping(Callback):
    """
    Early stopping callback
    patience: number of epochs to wait before stopping
    """

    def __init__(self, patience, verbose=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.best_loss = np.Inf
        self.counter = 0
        self.best_weights = []
        self.best_biases = []
        self.verbose = verbose

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
            if self.verbose:
                print("Early stopping")

            return True


class ToleranceEarlyStopping(Callback):
    """
    Early stopping callback
    patience: number of epochs to wait before stopping
    tolerance: percentage of improvement to wait before stopping
    """

    def __init__(self, tolerance, patience, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tolerance = tolerance
        self.best_loss = np.Inf
        self.counter = 0
        self.patience = patience
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

        if (
            1 - (self.best_loss / network.get_loss_value()) < self.tolerance
            and self.counter >= self.patience
        ):
            network.training = False
            for i, layer in enumerate(network.layers[1:]):
                layer.W = self.best_weights[i]
                layer.bias = self.best_biases[i]
            return True
