import numpy as np
from src.regularizers import L1, L2
from src.network import Network
from src.utils import load_moons, load_monk1, load_mnist
from src.network import Network
from src.activations import ReLU, Tanh, Sigmoid
from src.losses import MeanSquaredError
from src.metrics import BinaryAccuracy, MulticlassAccuracy, MeanEuclideanError
from itertools import product


def gridsearch(  # gridsearch per moons
        metric=BinaryAccuracy(),
        loss=MeanSquaredError(),
        eta=np.logspace(-2, -5, 4),
        nesterov=np.linspace(0.1, 0.8, 3),
        reg_type=np.array([L1, L2]),
        reg_val=np.logspace(-9, -11, 3)
):
    parameters = product(eta, nesterov, reg_type, reg_val)

    best_loss = np.Inf
    for par in parameters:
        [eta, nesterov, reg_type, reg_val] = par

        x_train, x_val, x_test, y_train, y_val, y_test = load_moons(validation=True, noise=0.2)
        if reg_type is not None:
            model = Network(2, reg_type(reg_val))
        else:
            model = Network(2)
        model.add_layer(8, ReLU())
        model.add_layer(1, Tanh())

        print("eta, nesterov, reg, lambda ", (eta, nesterov, reg_type, reg_val))
        stats = model.train(train=(x_train, y_train), validation=(x_val, y_val), metric=metric, loss=loss,
                            epochs=2, eta=eta, nesterov=nesterov, verbose=False)
        if model.val_stats[-1] <= best_loss:
            best_model = model
            best_par = par
    return best_model, best_par

best_model, best_par = gridsearch()
print(best_par)


