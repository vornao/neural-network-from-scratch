from src.search import grid_search
import numpy as np
from src.utils import load_moons,  load_cup
from src.network import Network
from src.activations import ReLU, Tanh, Sigmoid
from src.losses import MeanSquaredError
from src.metrics import BinaryAccuracy, MulticlassAccuracy, MeanEuclideanError

from sklearn.metrics import accuracy_score

from src.regularizers import L2, L1



x_train, x_val, y_train, y_val = load_cup()
model = Network(9)
model.add_layer(16, ReLU())
model.add_layer(8, ReLU())
model.add_layer(2, Sigmoid())

best_par = grid_search(model, x_train, y_train, x_val, y_val,
                       metric=MeanEuclideanError(), loss=MeanSquaredError(), epochs=200,
                       eta=[1e-1, 1e-2, 1e-3, 1e-4], nesterov=[0.5, 0.8], reg_type=(None,), reg_val=(0,), metric_decreasing=1)

print(best_par)


x_train, x_val, x_test, y_train, y_val, y_test = load_moons(validation=True, noise=0.2)
model = Network(2)
model.add_layer(8, ReLU())
model.add_layer(1, Tanh())
best_par = grid_search(model, x_train, y_train, x_val, y_val,
                       eta=(1e-1, 1e-2, 1e-3), nesterov=(0.5, 0.8), reg_type=(None,), reg_val=(0,))
print(best_par)

for par in best_par:
    eta, nesterov, regularizer, lamda = par
    model = Network(2)
    model.add_layer(8, ReLU())
    model.add_layer(1, Tanh())
    model.train(train=(x_train, y_train), validation=(x_val, y_val),
                eta=eta, nesterov=nesterov, epochs=200, verbose=True,
                loss=MeanSquaredError(), metric=BinaryAccuracy())

    predictions = model.multiple_outputs(x_test)

    print(accuracy_score(y_test.flatten(), np.round(predictions.flatten())))
