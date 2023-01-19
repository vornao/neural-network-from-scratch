from src.network import Network
from src.losses import MeanSquaredError
from src.metrics import BinaryAccuracy
from itertools import product
from heapq import nlargest, nsmallest
from src.validation import kfold_cv
from src.callbacks import EarlyStopping
import multiprocessing as mp


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
    scaler=None,
):

    parameters = None

    if reg_val == 0 or reg_type is None:
        parameters = product(eta, nesterov)
    else:
        parameters = product(eta, nesterov, reg_type, reg_val)

    jobs = []
    manager = mp.Manager()
    return_dict = manager.dict()
    params = {}
    count = 0


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

            print(
                "eta=",
                eta,
                "nesterov=",
                nesterov,
                "reg ",
                reg_type,
                "lambda =",
                reg_val,
            )
            # spawn 4 processes at a time
            if len(jobs) >= 4:
                for proc in jobs:
                    proc.join()
                jobs = []
            
            id = 'proc-'+str(count)
            params[id] = {'eta': eta, 'nesterov': nesterov, 'reg_type': reg_type, 'reg_val': reg_val}

            proc = mp.Process(target=kfold_cv, args=(model, x, y), kwargs={'k': n_folds, 'eta': eta, 'nesterov': nesterov, 'epochs': epochs, 'metric': metric, 'loss': loss, 'scaler': scaler, 'return_dict': return_dict, 'pid': id})
            jobs.append(proc)

            count+=1
            proc.start()

    
    for proc in jobs:
        proc.join()

    merged = {}
    for key in params.keys():
        merged[key] = (params[key], return_dict[key])


    return merged
