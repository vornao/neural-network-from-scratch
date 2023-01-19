import sys




from src.search import grid_search_cv
from src.utils import load_cup
from src.network import Network
from src.activations import ReLU,  Sigmoid
from src.losses import MeanSquaredError
from src.metrics import MeanEuclideanError
from src.regularizers import L1

x, y, scaler = load_cup(validation=False, scale_outputs=True)


model = Network(9)
model.add_layer(8, ReLU())
model.add_layer(8, ReLU())
model.add_layer(2, Sigmoid())


best_par = grid_search_cv(
    model_shape=model,
    x=x,
    y=y,
    metric=MeanEuclideanError(),
    loss=MeanSquaredError(),
    epochs=25,
    eta=[1e-1],
    nesterov=[0.5, 0.9],
    reg_type=(L1),
    reg_val=(0, 0.0001),
    scaler=scaler,
)

print(best_par)
