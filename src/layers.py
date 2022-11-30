import numpy as np
import activation_functions as af


class Layer:
    """Just a fully connected layer."""
 
    def __init__(
        self, units: int, input_shape: int, activation_function: af.Activation, bias=0.5
    ) -> None:

        # define shape of layer and number of units
        self.output_shape = units
        self.units = units
        self.activation = activation_function

        # randomly init weight matrix and biases
        self.W = np.matrix(np.random.uniform(size=(input_shape, units)))
        self.bias = np.ones((units,1)) * bias

  
    def output(self, x):
        """Compute layer's output and save it."""

        self.input = x
        self.net = self.W.T @ x + self.bias
        self.out = self.activation.activation(self.net)
        return self.out


    def update_weights(self, deltas, eta):

        # compute dl to update weights and deltas to propagate back.
        dl = np.multiply(deltas.T, self.activation.derivative(self.net))
        deltas_prop = self.W @ dl
        delta_w = self.input @ dl
        
        # update weights and biases
        self.W -= (eta * delta_w)
        self.bias -= dl * eta

        return deltas_prop


    def __str__(self) -> str:
        return f"Weights matrix = {self.W} \n, biases = {self.bias}"


class InputLayer(Layer):
    def __init__(self, units: int) -> None:
        self.units = units
        self.output_shape: int = units

    def output(self, x):
        return x


