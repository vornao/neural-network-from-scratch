import numpy as np
import activation_functions as af


class Layer:
    """
    Just a fully connected layer.
    """

    def __init__(
        self, units: int, input_shape: int, activation_function: af.Activation, bias=0.5
    ) -> None:

        # init bias vector
        self.bias = np.ones(units) * bias

        # set output shape
        self.output_shape = units
        self.units = units

        self.activation = activation_function

        # randomly init weight matrix
        self.W = np.random.uniform(size=(units, input_shape))

    def output(self, x):
        """
        Compute layer's output
        """
        self.lin_out = np.dot(self.W, x) + self.bias
        out = self.activation.activation(self.lin_out)
        self.last_input = x
        self.last_out = out
        return out

    def update_weights(self, deltas_prop, W_prop: np.ndarray, eta):

        # save last weights to propagate
        W_old = self.W

        # compute deltas for each unit in layer.
        # deltas is a vector with units length
        deltas = np.multiply(
            W_prop.T.dot(deltas_prop), self.activation.derivative(self.lin_out)
        )

        deltas = deltas.reshape(len(deltas), 1)  # TODO: how to not reshape everything?
        last_inputs = self.last_input.reshape(len(self.last_input), 1)
        self.W = self.W - eta * deltas.dot(last_inputs.T)

        # update bias
        self.bias = self.bias - deltas.flatten() * eta

        return W_old, deltas.flatten()

    def __str__(self) -> str:
        return f"Weights matrix = {self.W} \n, biases = {self.bias}"


class OutputLayer(Layer):
    def __init__(
        self, units: int, input_shape: int, activation_function: af.Activation, bias=0.5
    ) -> None:

        super().__init__(units, input_shape, activation_function, bias)

    def update_weights(self, deltas_prop, W_prop, eta):
        W_old = self.W

        # compute deltas for output layer:
        # in this case, we have the difference between labels and prediction
        deltas = np.multiply(deltas_prop, self.activation.derivative(self.lin_out))
        deltas = deltas.reshape(len(deltas), 1)
        last_inputs = self.last_input.reshape(len(self.last_input), 1)
        self.W = self.W - eta * deltas.dot(last_inputs.T)

        # update bias
        self.bias = self.bias - deltas.flatten() * eta

        return W_old, deltas.flatten()


class InputLayer(Layer):
    def __init__(self, units: int) -> None:
        self.units = units
        self.output_shape: int = units

    def output(self, x):
        return x
