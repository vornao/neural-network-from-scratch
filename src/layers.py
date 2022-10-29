import numpy as np
import activation_functions as af


class Layer:
    """
    Just a fully connected layer.
    """

    def __init__(
        self, units: int, input_shape: int, activation_function: af.Activation, bias=0.5
    ) -> None:

        # define shape of layer and number of units
        self.output_shape = units
        self.units = units
        self.activation = activation_function

        # randomly init weight matrix and biases
        self.W = np.random.uniform(size=(units, input_shape))
        self.bias = np.ones(units) * bias


    def output(self, x):
        """
        Compute layer's output and save it.
        """
        self.input = np.expand_dims(x, axis=1)
        self.net = np.dot(self.W, x) + self.bias
        self.out = self.activation.activation(self.net)
        return self.out


    def update_weights(self, deltas, eta):

        # compute dl to update weights and deltas to propagate back.
        dl = np.multiply(deltas, self.activation.derivative(self.net))
        deltas_prop = np.dot(dl, self.W)
        delta_w = np.dot(np.expand_dims(dl, axis=1), self.input.T)
        # update weights and biases
        self.W = self.W - (eta * delta_w)
        self.bias = self.bias - dl * eta

        return deltas_prop


    def __str__(self) -> str:
        return f"Weights matrix = {self.W} \n, biases = {self.bias}"


class OutputLayer(Layer):
    def __init__(
        self, units: int, input_shape: int, activation_function: af.Activation, bias=0.5
    ) -> None:

        super().__init__(units, input_shape, activation_function, bias)

    def update_weights(self, deltas, eta: int):

        # compute dl to update weights and deltas to propagate back.
        dl = np.multiply(deltas, self.activation.derivative(self.net))
        deltas_prop = np.dot(dl, self.W)
        delta_w = np.dot(np.expand_dims(dl, axis=1), self.input.T)

        # update weights and biases
        self.W = self.W - (eta * delta_w)
        self.bias = self.bias - dl * eta

        return deltas_prop


class InputLayer(Layer):
    def __init__(self, units: int) -> None:
        self.units = units
        self.output_shape: int = units

    def output(self, x):
        return x
