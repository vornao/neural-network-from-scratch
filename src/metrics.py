# define binary accuracy
from sklearn.metrics import accuracy_score
from numpy import round
from numpy import argmax


def _binary_accuracy(y_true, y_pred):
    y_true = round(y_true)
    y_pred = round(y_pred)
    return accuracy_score(y_true, y_pred)


def _multiclass_accuracy(y_true, y_pred):
    y_pred = argmax(y_pred, axis=1)
    y_true = argmax(y_true, axis=1)
    return accuracy_score(y_true, y_pred)


# define base class Metric
class Metric:
    def __init__(self, name=None):
        self.name = name

    def __call__(self, y_true, y_pred):
        raise NotImplementedError("I lay in the -- v o i d --")


# define class for binary accuracy
class BinaryAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.name = "binary_accuracy"

    def __call__(self, y_true, y_pred):
        return _binary_accuracy(y_true, y_pred)


# define class for multiclass accuracy
class MulticlassAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.name = "multiclass_accuracy"

    def __call__(self, y_true, y_pred):
        return _multiclass_accuracy(y_true, y_pred)

