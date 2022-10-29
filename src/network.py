import numpy as np
from itertools import islice

from typing import List 
from utils import Error, TRAIN_FMT
from layers import Layer, InputLayer, OutputLayer
from losses import Loss
from activation_functions import Activation
from progress.bar import Bar


import time


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
    def __backward_prop__(self, pred, target, loss: Loss, eta=10e-3):

        """
        Perform backward propagation during training.
        """
        # compute difference among targets and actual prediction.
        # this in fact is the gradient
        deltas = loss.dloss(pred, target)
        # W_prop = np.zeros((1, 1))

        # now compute backward prop for every other layer, except for input layer.
        for l in reversed(self.layers[1 : len(self.layers)]):
            # TODO put only gradient :)
            #
            deltas = l.update_weights(deltas=deltas, eta=eta)

    def multiple_outputs(self, patterns):
        outputs = []

        for p in patterns:
            outputs.append(self.__forward_prop__(p))

        return np.array(outputs)

    def output(self, x):
        """
        Computes network output, given an input vector x.
        """
        return self.__forward_prop__(x)

    def train(
        self,
        train_data,
        train_labels,
        val_data,
        val_labels,
        metric: Error,
        loss: Loss,
        epochs=25,
        eta=10e-3,
        verbose=True,
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

            # Just some printing stuff
            bar = Bar(f"Epoch: {epoch}/{epochs}", max=len(train_labels))
            bar.bar_prefix = "["
            bar.bar_suffix = "]"

            # Forward propagation over single pattern and back propagation
            # for now it's just on-line learning.
            # remove that ugly predictions list.

            predictions = []

            for train, label in zip(train_data, train_labels):
                p = self.output(train)
                predictions.append(p)

                self.__backward_prop__(pred=p, target=label, eta=eta, loss=loss)

                if verbose:
                    bar.next()

            tr_error = loss.loss(predictions, train_labels)
            val_error = loss.loss(self.multiple_outputs(val_data), val_labels)

            if verbose:
                print(TRAIN_FMT.format(epoch, round(tr_error, 4), round(val_error, 4)))

            tr_stats.append(tr_error)
            val_stats.append(val_error)

            if verbose:
                bar.finish()

        print(TRAIN_FMT.format(epochs, round(tr_error, 4), round(val_error, 4)))
        return tr_stats, val_stats
