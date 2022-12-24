import numpy as np
import src.activations as af
import src.regularizers as reg


class Layer:
    """Just a fully connected layer."""
 
    def __init__(
        self,
        units: int,
        input_shape: int,
        activation_function:
        af.Activation,
        bias=0.5,
        regularizer : reg.Regularizer = None
    ) -> None:

        # define shape of layer and number of units
        self.output_shape = units
        self.units = units
        self.activation = activation_function
        self.regularizer = regularizer

        # randomly init weight matrix and biases
        self.W = np.random.uniform(low=-0.05, high=0.05, size=(input_shape, units))
        self.bias = np.random.uniform(low=-0.01, high=0.01, size=(units, 1))
        self.last_input = np.NaN
        self.last_net = np.NaN
        self.last_output = np.NaN
        self.last_delta = np.zeros(self.W.shape) # last delta used for momentum
    def net(self, x):
        return (self.W.T @ x) + self.bias
  
    def output(self, x):
        """Compute layer's output and save it."""
        self.last_input = np.copy(x)
        self.last_net = self.net(x)
        self.last_output = self.activation.activation(self.last_net)

        return self.last_output

    def update_weights(self, deltas, eta, nesterov):
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

    def __str__(self) -> str:
        return f"Weights matrix = {self.W} \n, biases = {self.bias}"


class InputLayer(Layer):
    def __init__(self, units: int) -> None:
        self.output_shape = units
        self.units = units

    def output(self, x):
        return x


