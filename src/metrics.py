# define binary accuracy
from sklearn.metrics import accuracy_score
from numpy import round
from numpy import argmax
import numpy as np


def _binary_accuracy(y_pred, y_true):
    y_pred = round(y_pred)
    y_pred.shape = y_true.shape

    return accuracy_score(y_true, y_pred)


def _multiclass_accuracy(y_pred, y_true):
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

    def __call__(self, y_pred, y_true):
        return _binary_accuracy(y_pred, y_true)


# define class for multiclass accuracy
class MulticlassAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.name = "multiclass_accuracy"

    def __call__(self, y_pred, y_true):
        return _multiclass_accuracy(y_pred, y_true)

class MeanEuclideanError(Metric):
    def __init__(self):
        super().__init__()
        self.name = "mee"
        
    def __call__(self, y_pred, y_true):
        
        s = np.array([])
        for o, t in zip(y_pred, y_true):
            s = np.append(s, np.linalg.norm(o-t, 2))
            
        return np.mean(s)
