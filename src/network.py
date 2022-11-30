import numpy as np

from typing import List 
from utils import Error, TRAIN_FMT, chunker
from layers import Layer, InputLayer
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



    def __forward_prop__(self, x):
        """
        Perform forward propagation and return network output
        """

        out = x
        for l in self.layers:
            out = l.output(out)

        return out


    # TODO: remove pred-target cost evaluation to exploit minibatch training
    def __backward_prop__(self, deltas, loss: Loss, eta=10e-3, batch_size=1):

        """
        Perform backward propagation during training.
        """
        d = deltas
        # now compute backward prop for every other layer, except for input layer.
        for layer in reversed(self.layers[1 : len(self.layers)]):
            d = layer.update_weights(deltas=d, eta=eta)

    
    def multiple_outputs(self, patterns):
        outputs = []

        for p in patterns:
            outputs.append(self.__forward_prop__(np.expand_dims(p, 1)))

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
        batch_size=1,
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
            bar = Bar(f"Epoch: {epoch}/{epochs}", max=len(train_labels)/batch_size)
            bar.bar_prefix = "["
            bar.bar_suffix = "]"


            # make all pairs <input, label>
            zipped = zip(train_data, train_labels)

            # make batch_size sized tuples
            # ((x1, d1), (x2, d2), ... , (x_batch_size, d_batch_size))
            # batched = chunker(zipped, batch_size)

            # every iteration will return a tuple of tuples.
            # need to iterate over each tuple in batch and compute outputs. 
            # then compute cost gradient and backprop.

            outputs = self.multiple_outputs(train_data)
            deltas  = np.matrix([np.mean(outputs.flatten() - train_labels)])

        
            self.__backward_prop__(
                deltas=deltas,
                eta=eta, 
                loss=loss,
                batch_size=batch_size
            )

            if verbose:
                bar.next()

            
            tr_error = loss.loss(self.multiple_outputs(train_data), train_labels)
            val_error = loss.loss(self.multiple_outputs(val_data), val_labels)

            if verbose:
                print(TRAIN_FMT.format(epoch, round(tr_error, 4), round(val_error, 4)))

            tr_stats.append(tr_error)
            val_stats.append(val_error)

            if verbose:
                bar.finish()

        print(TRAIN_FMT.format(epochs, round(tr_error, 4), round(val_error, 4)))
        return tr_stats, val_stats
