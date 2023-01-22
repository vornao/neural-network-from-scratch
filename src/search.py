# Author: Giacomo Lagomarsini - Luca Miglior - Leonardo Stoppani
# Date: 2023-01-23
# License: MIT

from src.network import Network
from src.losses import MeanSquaredError
from src.metrics import BinaryAccuracy
from itertools import product
from src.validation import kfold_cv
import multiprocessing as mp
from tqdm import tqdm


def grid_search_cv(
    model_shape: Network,
    x,
    y,
    n_folds=3,
    metric=BinaryAccuracy(),
    loss=MeanSquaredError(),
    eta=[1e-3],
    nesterov=[0.5],
    reg_type=[None],
    reg_val=[0],
    epochs=1000,
    verbose=False,
    scaler=None,
    workers=4,
):
    """
    Grid search cross validation.
    @param model_shape: Network object with the desired network architecture.
    @param x: training data.
    @param y: training labels.
    @param n_folds: number of folds for cross validation.
    @param metric: metric to use for evaluation.
    @param loss: loss function to use.
    @param eta: learning rate.
    @param nesterov: nesterov momentum strength.
    @param reg_type: regularization type.
    @param reg_val: regularization value.
    @param epochs: number of epochs to train.
    @param verbose: print progress bar.
    @param scaler: scaler to use for data normalization.
    @param workers: number of workers to use for parallelization.
    """

    parameters = None
    n_tries = 0
    if reg_val == 0 or reg_type is None:
        parameters = product(eta, nesterov)
        n_tries = len(eta) * len(nesterov)
    else:
        parameters = product(eta, nesterov, reg_type, reg_val)
        n_tries = len(eta) * len(nesterov) * len(reg_type) * len(reg_val)

    jobs = []
    manager = mp.Manager()
    return_dict = manager.dict()
    params = {}
    count = 0

    print("Gridsearch: exploring " + str(n_tries) + " combinations.")

    bar = tqdm(total=n_tries)

    for par in parameters:

        [eta, nesterov, reg_type, reg_val] = par
        if (reg_type is not None and reg_val != 0) or (
            reg_type is None and reg_val == 0
        ):
            # create new model for each grid search.
            if reg_type is not None:
                model = Network(model_shape.layers[0].units, reg_type(reg_val))
            else:
                model = Network(model_shape.layers[0].units)

            for layer in model_shape.layers[1:]:
                model.add_layer(layer.units, layer.activation)

            # spawn workers processes at a time
            if len(jobs) >= workers:
                for proc in jobs:
                    proc.join()
                    bar.update(1)
                jobs = []

            id = "proc-" + str(count)
            params[id] = {
                "eta": eta,
                "nesterov": nesterov,
                "reg_type": get_reg_as_string(reg_type),
                "reg_val": reg_val,
            }

            proc = mp.Process(
                target=kfold_cv,
                args=(model, x, y),
                kwargs={
                    "k": n_folds,
                    "eta": eta,
                    "nesterov": nesterov,
                    "epochs": epochs,
                    "metric": metric,
                    "loss": loss,
                    "scaler": scaler,
                    "return_dict": return_dict,
                    "pid": id,
                    "verbose": False,
                },
            )
            jobs.append(proc)

            count += 1
            proc.start()

    for proc in jobs:
        proc.join()
        bar.update(1)

    merged = {}
    for key in params.keys():
        merged[key] = (params[key], return_dict[key])

    bar.close()
    return merged


def get_reg_as_string(reg_type):
    s = str(reg_type)
    if reg_type is None:
        return "None"
    else:
        return "L1" if s.find("L1") > -1 else "L2"
