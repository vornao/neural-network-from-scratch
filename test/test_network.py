from src.network import Network
from src.activations import ReLU, Tanh
from src.losses import MeanSquaredError
from src.metrics import BinaryAccuracy
from src.utils import load_monk1
from src.regularizers import L2
from src.callbacks import EarlyStopping


def test_network_monk1():

    # load monk1
    x_train, x_val, x_test, y_train, y_val, y_test = load_monk1()

    binary_accuracy = BinaryAccuracy()

    model = Network(17)
    model.add_layer(6, ReLU())
    model.add_layer(1, Tanh())

    model.train((x_train, y_train), (x_val, y_val),
        metric=binary_accuracy,
        loss=MeanSquaredError(),
        epochs=1000,
        verbose=False,
        callbacks=[EarlyStopping(patience=100)],
        nesterov=0.5
                )

    # compute accuracy
    y_pred = model.multiple_outputs(x_val)
    acc = binary_accuracy(y_pred, y_val)

    assert acc == 1

