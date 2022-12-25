import numpy as np

from typing import List
from src.layers import Layer, InputLayer
from src.activations import Activation
from src.metrics import Metric
from src.losses import Loss
from src.regularizers import Regularizer
from tqdm import tqdm
from src.callbacks import Callback

fmt = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}[{postfix}]'


def update_bar(bar, stats):
    bar.set_postfix(stats)
    bar.update(1)


class Network:
    """
    Your dense neural network class.
    """

    def __init__(self, input_shape: int, regularizer: Regularizer = None) -> None:
        self.layers: List[Layer] = []
        self.layers.append(InputLayer(input_shape))
        self.tr_stats = []
        self.val_stats = []
        self.tr_err = []
        self.val_err = []
        self.regularizer = regularizer
        self.bar = None
        self.training = True

    def add_layer(self, units, activation_function: Activation, bias=0.5):
        """
        Just add a new hidden layer to the network, with requested
        activation function and units.
        """
        self.layers.append(
            Layer(units, self.layers[-1].output_shape, activation_function, bias, regularizer=self.regularizer)
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
    def __backward_prop__(self, deltas, eta, nesterov, batch_size=1):
        """
        Perform backward propagation during training.
        """
        d = deltas
        # now compute backward prop for every other layer, except for input layer.
        for layer in reversed(self.layers[1: len(self.layers)]):
            d = layer.update_weights(deltas=d, eta=eta, nesterov=nesterov)

    def multiple_outputs(self, patterns):
        outputs = []

        for p in patterns:
            p.shape = (len(p), 1)
            out = self.__forward_prop__(p)
            outputs.append(out)

        outputs = np.array(outputs)
        return outputs

    def output(self, x):
        """
        Computes network output, given an input vector x.
        """
        return self.__forward_prop__(x)

    def train(
            self,
            train: tuple,
            validation: tuple,
            metric: Metric,
            loss: Loss,
            epochs=25,
            eta=10e-3,
            batch_size=1,
            verbose=True,
            callbacks: List[Callback] = [],
            nesterov=0 # parameter for momentum
    ):
        """
        Train network with given data and labels for requested epoch.
        Print progress after each epoch.
        """

        train_data, train_labels = train
        val_data, val_labels = validation

        bar = None
        if verbose:
            self.bar = tqdm(total=epochs, desc="Training", leave=True, bar_format=fmt)

        # TODO:
        # - implement minibatch training computing error for b sized training
        #   labels and passing it to backward prop function
        # - implement regularization
        # - implement momentum

        for epoch in range(0, epochs):
            if not self.training:
                break
            # make batch_size sized tuples
            # ((x1, d1), (x2, d2), ... , (x_batch_size, d_batch_size))
            # batched = chunker(zipped, batch_size)

            for x, target in zip(train_data, train_labels):

                pred = self.__forward_prop__(x)
                deltas = loss.backward(pred, target)
                self.__backward_prop__(deltas=deltas, eta=eta, nesterov=nesterov)


            # compute training error and accuracy for current epoch and append stats
            self.epoch_stats(epoch, train_data, train_labels, val_data, val_labels, metric, loss, verbose, self.bar)

            for callback in callbacks:
                callback(self)


        stats = {
            # "epochs": epochs,
            "train_loss": self.tr_stats,
            "val_loss": self.val_stats,
            "train_acc": self.tr_err,
            "val_acc": self.val_err,
        }

        return stats

    def epoch_stats(self, epoch, tr, tr_labels, val, val_labels, metric, loss, verbose, bar):
        """
        Compute network accuracy and loss given data and labels.
        """
        # compute training error and accuracy for current epoch and append stats

        val_loss = None
        val_metric = None

        tr_loss = loss.loss(self.multiple_outputs(tr), tr_labels)
        tr_metric = metric(self.multiple_outputs(tr), tr_labels)
        self.tr_stats.append(tr_loss)
        self.tr_err.append(tr_metric)

        if val is not None:
            val_loss = loss.loss(self.multiple_outputs(val), val_labels)
            val_metric = metric(self.multiple_outputs(val), val_labels)
            self.val_err.append(val_metric)
            self.val_stats.append(val_loss)

        epoch_stats = {
            "loss": tr_loss,
            "val_loss": val_loss,
            "val_acc": val_metric,
        }

        if verbose:
            update_bar(bar, epoch_stats)

    def get_loss_value (self):
        return self.val_stats[-1]


    def reset_model(self):
        self.tr_stats = []
        self.val_stats = []
        self.tr_err = []
        self.val_err = []

        for layer in self.layers[1:]:
            layer.reset()