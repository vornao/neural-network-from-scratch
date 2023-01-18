from src.search import grid_search_cv
import numpy as np
from src.utils import load_moons, load_cup
from src.network import Network
from src.activations import ReLU, Tanh, Sigmoid
from src.losses import MeanSquaredError
from src.metrics import BinaryAccuracy,  MeanEuclideanError
from src.regularizers import L1
from sklearn.metrics import accuracy_score

x,y,scaler = load_cup(validation=False, scale_outputs=True)



model = Network(9)
model.add_layer(16, ReLU())
model.add_layer(8, ReLU())
model.add_layer(2, Sigmoid())




best_par = grid_search_cv(model_shape=model, x=x, y=y,
                          metric=MeanEuclideanError(), loss=MeanSquaredError(), epochs=10,
                          eta=[1e-1, 1e-2, 1e-3, 1e-4], nesterov=[0.5, 0.8], reg_type=(None,L1), reg_val=(0,0.0001),
                          scaler=scaler)

print(best_par)

