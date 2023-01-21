import numpy as np
from src.metrics import BinaryAccuracy, MulticlassAccuracy


def test_binary_accuracy():
    acc = BinaryAccuracy()
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    assert acc(y_pred, y_true) == 1

    y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    assert acc(y_pred, y_true) == 0

    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    assert acc(y_pred, y_true) == 0

    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1])

    assert acc(y_pred, y_true) == 0.5

    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.7, 0.6, 0.51])
    y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1])

    assert acc(y_pred, y_true) == 0.5


def test_multiclass_accuracy():

    # test multilcass accuracy
    acc = MulticlassAccuracy()
    y_pred = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    assert acc(y_pred, y_true) == 1

    y_pred = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    assert acc(y_pred, y_true) == 0

    y_pred = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    y_true = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    assert acc(y_pred, y_true) == 0.3333333333333333

    y_pred = np.array([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])
    assert acc(y_pred, y_true) == 0.3333333333333333

    y_pred = np.array([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])
    y_true = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1]])

    assert acc(y_pred, y_true) == 0.6666666666666666
