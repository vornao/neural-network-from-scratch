import numpy as np
from typing import List
from utils import Error
from layers import Layer, InputLayer, OutputLayer
from activation_functions import Activation


class Network:
    """
    Your dense neural network class.
    """
    layers: List[Layer] = []

    def __init__(self, input_shape: int) -> None:
        self.layers.append(InputLayer(input_shape))


    def add_layer(self, units, activation_function: Activation):
        """
        Just add a new hidden layer to the network, with requested 
        activation function and units.
        """
        self.layers.append(
            Layer(units, self.layers[-1].output_shape, activation_function)
        )


    def add_output_layer(self, units, activation_function: Activation):
        """
        Add an output layer with requested activation functions and units.
        """
        self.layers.append(
            OutputLayer(units, self.layers[-1].output_shape, activation_function)
        )


    def __forward_prop__(self, x):
        """
        Perform forward propagation and return network output
        """

        out = x
        for l in self.layers:
            out = l.output(out)

        return out

    def __backward_prop__(self, pred, target, eta=10e-3):

        """
        Perform backward propagation during training.
        """
        # compute difference among targets and actual prediction.
        deltas_prop = np.add(pred, -1*target)
        W_prop = np.zeros((1, 1))

        # now compute backward prop for every other layer, except for input layer.
        for l in reversed(self.layers[1 : len(self.layers)]):
            W, deltas = l.update_weights(deltas_prop, W_prop, eta)
            W_prop, deltas_prop = W, deltas


    def output(self, x):
        """
        Computes network output, given an input vector x.
        """
        return self.__forward_prop__(x)

    


    def train(self, train_data, train_labels, estimator: Error, epochs=25, eta=10e-3):
        """
        Train network with given data and labels for requested epoch.
        Print progress after each epoch.
        """
        
        l = len(train_data)

        for epoch in range(0, epochs):
            error_sum = 0

            for i in range(0, l):
                pred = self.output(train_data[i])
                error_sum += estimator.validate(train_labels[i], pred)
                self.__backward_prop__(pred, train_labels[i], eta=eta)

            print(f"> Epoch {epoch}. Training loss: {error_sum / l}")
