import numpy as np

from typing import List
from src.utils import TRAIN_FMT
from src.layers import Layer, InputLayer
from src.activations import Activation
from src.metrics import Metric
from src.losses import Loss
from tqdm.auto import tqdm
from time import sleep

fmt = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}[{postfix}]'


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
        for layer in self.layers:
            out = layer.output(out)

        return out

    # TODO: remove pred-target cost evaluation to exploit minibatch training
    def __backward_prop__(self, deltas, eta, batch_size=1):

        """
        Perform backward propagation during training.
        """
        d = deltas
        # now compute backward prop for every other layer, except for input layer.
        for layer in reversed(self.layers[1: len(self.layers)]):
            d = layer.update_weights(deltas=d, eta=eta)

    def multiple_outputs(self, patterns):
        outputs = []

        for p in patterns:
            p.shape = (len(p), 1)
            out = self.__forward_prop__(p)
            outputs.append(out)

        outputs = np.array(outputs)
        outputs.shape = (outputs.shape[0], outputs.shape[2])
        return outputs

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
            metric: Metric,
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
        tr_err = []
        val_err = []

        bar = tqdm(total=epochs, desc="Training", leave=True, bar_format=fmt)

        # TODO:
        # - implement minibatch training computing error for b sized training
        #   labels and passing it to backward prop function
        # - implement magnitude gradient descent algorithm
        for epoch in range(0, epochs):

            # make batch_size sized tuples
            # ((x1, d1), (x2, d2), ... , (x_batch_size, d_batch_size))
            # batched = chunker(zipped, batch_size)

            for x, target in zip(train_data, train_labels):
                x.shape = (x.shape[0], 1)
                pred = self.__forward_prop__(x)
                deltas = pred - target
                deltas.shape = (deltas.shape[0], 1)

                self.__backward_prop__(deltas=deltas, eta=eta)

            tr_loss = loss.loss(self.multiple_outputs(train_data), train_labels)
            val_loss = loss.loss(self.multiple_outputs(val_data), val_labels)

            tr_error_stats = metric(self.multiple_outputs(train_data), train_labels)
            val_error_stats = metric(self.multiple_outputs(val_data), val_labels)

            tr_stats.append(tr_loss)
            val_stats.append(val_loss)
            tr_err.append(tr_error_stats)
            val_err.append(val_error_stats)

            if verbose:
                stats = {
                    "Loss": tr_loss,
                    "Val loss": val_loss,
                    "Val acc": val_error_stats,
                }
                bar.set_postfix(stats)
                bar.update(1)

        stats = {
            "epochs": range(epochs),
            "train_loss": tr_stats,
            "val_loss": val_stats,
            "train_error": tr_err,
            "val_error": val_err,
        }

        return stats
