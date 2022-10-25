import numpy as np
from typing import List
from utils import Error, TRAIN_FMT
from layers import Layer, InputLayer, OutputLayer
from activation_functions import Activation


class Network:
    """
    Your dense neural network class.
    """

    def __init__(self, input_shape: int) -> None:
        self.layers: List[Layer] = []
        self.layers.append(InputLayer(input_shape))

    def add_layer(self, units, activation_function: Activation, bias=0.5):
        """
        Just add a new hidden layer to the network, with requested
        activation function and units.
        """
        self.layers.append(
            Layer(units, self.layers[-1].output_shape, activation_function, bias)
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

    # TODO: remove pred-target cost evaluation to exploit minibatch training
    def __backward_prop__(self, pred, target, eta=10e-3):

        """
        Perform backward propagation during training.
        """
        # compute difference among targets and actual prediction.
        deltas_prop = np.add(pred, -target)
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

    def compute_loss(self, x, y, estimator: Error):
        error = 0
        for x_t, y_t in zip(x, y):
            pred = self.output(x_t)
            error += estimator.validate(y_t, pred)

        return error / len(x)

    def train(
        self,
        train_data,
        train_labels,
        val_data,
        val_labels,
        estimator: Error,
        epochs=25,
        eta=10e-3,
        verbose=True
    ):
        """
        Train network with given data and labels for requested epoch.
        Print progress after each epoch.
        """
        tr_stats = []
        val_stats = []

        # TODO: 
        # - implement minibatch training computing error for b sized training labels and passing it to bacwkard prop function
        # - implement magnitude gradient descent algorithm
        for epoch in range(0, epochs):

            for train, label in zip(train_data, train_labels):
                self.__backward_prop__(
                    pred= self.output(train), 
                    target=label, 
                    eta=eta
                )

            tr_error = self.compute_loss(train_data, train_labels, estimator)
            val_error = self.compute_loss(val_data, val_labels, estimator)
            if verbose:
                print(TRAIN_FMT.format(epoch, round(tr_error, 4), round(val_error, 4)))

            tr_stats.append(tr_error)
            val_stats.append(val_error)
            
        print(TRAIN_FMT.format(epochs, round(tr_error, 4), round(val_error, 4)))
        return tr_stats, val_stats
