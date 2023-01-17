from src.network import Network
from src.losses import MeanSquaredError
from src.metrics import BinaryAccuracy
from itertools import product
from heapq import nlargest, nsmallest
from src.validation import kfold_cv
from src.callbacks import EarlyStopping

import multiprocessing as mp

def grid_search(
        model_shape: Network,
        x_train,
        y_train,
        x_val,
        y_val,
        k=1,
        metric=BinaryAccuracy(),
        loss=MeanSquaredError(),
        eta=(1e-3,),
        nesterov=(0.5,),
        reg_type=(None,),
        reg_val=(1e-10,),
        metric_decreasing=0,
        epochs=100
):
    parameters = product(eta, nesterov, reg_type, reg_val)
    param = {}  # dictionary accuracy : set of parameters
    for par in parameters:
        [eta, nesterov, reg_type, reg_val] = par
        if reg_type is not None:
            model = Network(model_shape.layers[0].units, reg_type(reg_val))
        else:
            model = Network(model_shape.layers[0].units)
        for layer in model_shape.layers[1:]:
            model.add_layer(layer.units, layer.activation)

        print("eta=", eta, "nesterov=", nesterov, "reg ", reg_type, "lambda =", reg_val)
        model.train(train=(x_train, y_train), validation=(x_val, y_val), metric=metric, loss=loss,
                    epochs=epochs, eta=eta, nesterov=nesterov, verbose=True)
        y_pred = model.multiple_outputs(x_val)
        acc = metric(y_pred, y_val)
        param.update({acc: par})
    if metric_decreasing:
        best_parameters = [param[i] for i in nsmallest(k, param)]
    else:
        best_parameters = [param[i] for i in nlargest(k, param)]
    return best_parameters


def grid_search_cv(
        model_shape: Network,
        x,
        y,
        metric=BinaryAccuracy(),
        loss=MeanSquaredError(),
        eta=(1e-3,),
        nesterov=(0.5,),
        reg_type=(None,),
        reg_val=(1e-10,),
        epochs=1000,
        scaler=None
):
    parameters = product(eta, nesterov, reg_type, reg_val)
    history = []

    for par in parameters:
        [eta, nesterov, reg_type, reg_val] = par
        
        # create new model for each grid search.
        if reg_type is not None:
            model = Network(model_shape.layers[0].units, reg_type(reg_val))
        else:
            model = Network(model_shape.layers[0].units)

        
        for layer in model_shape.layers[1:]:
            model.add_layer(layer.units, layer.activation)

        print("eta=", eta, "nesterov=", nesterov, "reg ", reg_type, "lambda =", reg_val)
        acc = kfold_cv(model=model, x=x, y=y, k=3,  eta=eta, nesterov=nesterov, epochs=epochs, metric=metric, loss=loss, scaler=scaler, callbacks=[EarlyStopping(int(epochs/10))],verbose=True)

        history.append({'parameters': par, 'metrics': acc})
    
    return history

