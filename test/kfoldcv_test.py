from src.network import Network
from src.activations import ReLU, Sigmoid
from src.losses import MeanSquaredError
from src.metrics import MeanEuclideanError
from src.utils import load_cup
from src.regularizers import L2
from src.callbacks import EarlyStopping
from src.validation import kfold_cv



def test_kfold():
    x, x_test, y, y_test = load_cup(test_size=0.2)
    model = Network(9, regularizer=L2(0.1))
    model.add_layer(4, ReLU())
    model.add_layer(4, ReLU())
    model.add_layer(2, Sigmoid())

    accuracies = kfold_cv(
        model,
        x,
        y,
        k=5,
        metric=MeanEuclideanError(),
        loss=MeanSquaredError(),
        epochs=10,
        eta=0.01,
        verbose=False
    )
    assert accuracies is not None
    assert len(accuracies) == 5


def test_kfold_negative_k():
    x, x_test, y, y_test = load_cup(test_size=0.2)
    model = Network(9, regularizer=L2(0.1))
    model.add_layer(4, ReLU())
    model.add_layer(4, ReLU())
    model.add_layer(2, Sigmoid())

    try:
        kfold_cv(
            model,
            x,
            y,
            k=-1,
            metric=MeanEuclideanError(),
            loss=MeanSquaredError(),
            epochs=10,
            eta=0.01,
            verbose=False,
            callbacks=[EarlyStopping(patience=100)]
        )

    except ValueError as e:
        assert e is not None

