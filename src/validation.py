from src.network import Network
from src.activations import ReLU, Tanh
from src.losses import MeanSquaredError, Loss
from src.metrics import BinaryAccuracy, Metric
from src.utils import load_monk1
from src.regularizers import L2
from src.callbacks import EarlyStopping, Callback

import numpy as np



def kfold_cv(model: Network, x, y, k=5, return_mean=True, **kwargs):
    """
    Perform k-fold cross validation on the given dataset.
    :param model: the network to train (instance of Network)
    :param k: the number of folds (default 5)
    :param x: the input dataset
    :param y: the output dataset
    :param kwargs: additional arguments to pass to the train method
    :return_mean: if True, return the mean of the k accuracies, otherwise return a list of k accuracies
    :return: the mean accuracy or a list of k accuracies
    :raises ValueError: if k is negative
    """
    if k < 1:
        raise ValueError("k must be greater than 1")

    x_folds = np.array_split(x, k)
    y_folds = np.array_split(y, k)

    metric = kwargs.get('metric', BinaryAccuracy())
    loss = kwargs.get('loss', MeanSquaredError())
    epochs = kwargs.get('epochs', 100)
    eta = kwargs.get('eta', 10e-3)
    callbacks = kwargs.get('callbacks', [])
    verbose = kwargs.get('verbose', False)

    accuracies = []
    for i in range(k):
        x_train = np.concatenate(x_folds[:i] + x_folds[i+1:])
        y_train = np.concatenate(y_folds[:i] + y_folds[i+1:])
        x_val = x_folds[i]
        y_val = y_folds[i]

        model.train((x_train, y_train), (x_val, y_val),
            metric=metric,
            loss=loss,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            eta=eta)

        # compute accuracy
        y_pred = model.multiple_outputs(x_val)
        acc = metric(y_pred, y_val)
        accuracies.append(acc)

    if return_mean:
        return np.mean(accuracies)

    return accuracies

