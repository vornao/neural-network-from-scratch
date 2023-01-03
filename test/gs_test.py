from src.search import grid_search, grid_search_cv
import numpy as np
from src.utils import load_moons, load_cup
from src.network import Network
from src.activations import ReLU, Tanh, Sigmoid
from src.losses import MeanSquaredError
from src.metrics import BinaryAccuracy,  MeanEuclideanError

from sklearn.metrics import accuracy_score

x_train, x_val, y_train, y_val = load_cup()
x = np.append(x_train, x_val, axis=0)
y = np.append(y_train, y_val, axis=0)

model = Network(9)
model.add_layer(16, ReLU())
model.add_layer(8, ReLU())
model.add_layer(2, Sigmoid())
print(x_train.shape)
print(x_val.shape)
print(x.shape)

best_par = grid_search_cv(model_shape=model, x=x, y=y,
                          metric=MeanEuclideanError(), loss=MeanSquaredError(), epochs=10,
                          eta=[1e-1, 1e-2, 1e-3, 1e-4], nesterov=[0.5, 0.8], reg_type=(None,), reg_val=(0,),
                          metric_decreasing=1)

print(best_par)

