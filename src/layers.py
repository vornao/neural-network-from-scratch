# Author: Giacomo Lagomarsini - Luca Miglior - Leonardo Stoppani
# Date: 2023-01-23
# License: MIT

import numpy as np
import src.activations as af
import src.regularizers as reg


class Layer:
    """Just a fully connected layer."""

    def __init__(
        self,
        units: int,
        input_shape: int,
        activation_function: af.Activation,
        bias=0.5,
        regularizer: reg.Regularizer = None,
        initializer: str = "uniform",
    ) -> None:

        # define shape of layer and number of units
        self.output_shape = units
        self.input_shape = input_shape
        self.units = units
        self.activation = activation_function
        self.regularizer = regularizer
        self.initializer = initializer

        self.init_layer()

    def net(self, x):
        """Compute layer's linear net input."""
        return (self.W.T @ x) + self.bias

    def output(self, x):
        """Compute layer's output and save it."""
        self.last_input = np.copy(x)
        self.last_net = self.net(x)
        self.last_output = self.activation.activation(self.last_net)

        return self.last_output

    def update_weights(self, deltas, eta, nesterov):
        """Update weights and biases using backpropagation."""

        # compute dl to update weights and deltas to propagate back.
        dl = deltas * self.activation.derivative(self.last_net)
        delta_w = self.last_input @ dl.T
        deltas_prop = self.W @ dl

        if nesterov:
            delta_w = delta_w + nesterov * self.last_delta
            self.last_delta = delta_w

        if self.regularizer:
            self.W -= eta * delta_w + self.regularizer.gradient(self.W)
        else:
            self.W -= eta * delta_w

        self.bias -= dl * eta
        return deltas_prop

    def init_layer(self):
        """Initialize weights and biases matrix."""

        input_shape = self.input_shape
        units = self.units

        if self.initializer == "uniform":
            # init weights and biases matrix with uniform distribution
            self.W = np.random.uniform(low=-0.5, high=0.5, size=(input_shape, units))
            self.bias = np.random.uniform(low=-0.1, high=0.1, size=(units, 1))

        elif self.initializer == "xavier":
            # init weights and biases matrix with Xavier initialization
            self.W = np.random.uniform(
                low=-np.sqrt(1 / (input_shape)),
                high=np.sqrt(1 / (input_shape)),
                size=(input_shape, units),
            )
            self.bias = np.random.uniform(
                low=-np.sqrt(1 / (input_shape)),
                high=np.sqrt(1 / (input_shape)),
                size=(units, 1),
            )

        elif self.initializer == "he":
            # init weights and biases matrix with He initialization
            self.W = np.random.uniform(
                low=-np.sqrt(2 / (input_shape)),
                high=np.sqrt(2 / (input_shape)),
                size=(input_shape, units),
            )
            self.bias = np.random.uniform(
                low=-np.sqrt(2 / (input_shape)),
                high=np.sqrt(2 / (input_shape)),
                size=(units, 1),
            )

        self.last_input = np.NaN
        self.last_net = np.NaN
        self.last_output = np.NaN
        # last delta used for momentum
        self.last_delta = np.zeros(self.W.shape)

        return self.W, self.bias

    def __str__(self) -> str:
        return f"Weights matrix = {self.W} \n, biases = {self.bias}"


class InputLayer(Layer):
    """Input layer."""

    def __init__(self, units: int) -> None:
        self.output_shape = units
        self.units = units

    def output(self, x):
        return x
