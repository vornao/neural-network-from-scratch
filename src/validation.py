from src.network import Network
from src.activations import ReLU, Tanh
from src.losses import MeanSquaredError, Loss
from src.metrics import BinaryAccuracy, Metric
from src.utils import load_monk1
from src.regularizers import L2
from src.callbacks import EarlyStopping, Callback

import numpy as np
import multiprocessing as mp
import os


def kfold_cv(model: Network, x, y, k=5, return_mean=True, workers=1, **kwargs):
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
    nesterov = kwargs.get('nestrov', 0)

    accuracies = []

    # use multiprocessing to speed up the process
    if workers > 1:
        with mp.Pool(processes=workers) as pool:
            results = [
                pool.apply_async(
                        _kfold_cv_worker, 
                        args=(model, x_folds, y_folds, i, metric, loss, epochs, eta, callbacks, verbose, nesterov)
                    ) for i in range(k)
                ]
            accuracies = [p.get() for p in results]
    else:
        for i in range(k):
            accuracies.append(_kfold_cv_worker(model, x_folds, y_folds, i, metric, loss, epochs, eta, callbacks, verbose, nesterov))
    
    if return_mean:
        return np.mean(accuracies)

    return accuracies


def _kfold_cv_worker(model: Network, x_folds, y_folds, i, metric, loss, epochs, eta, callbacks, verbose, nesterov):
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
    x_test = x_folds[i]
    y_test = y_folds[i]
    x_train = np.concatenate([x_folds[j] for j in range(len(x_folds)) if j != i])
    y_train = np.concatenate([y_folds[j] for j in range(len(y_folds)) if j != i])
    print(">", os.getpid(), "started training")
    model.reset_model()
    model.train(x_train, y_train, epochs=epochs, eta=eta, callbacks=callbacks, verbose=verbose, nesterov=nesterov)
    print(">", os.getpid(), "finished training")
    val_acc = metric(y_test, model.predict(x_test))
    print(">", os.getpid(), "finished validation", val_acc)

    return val_acc




    

