# Author: Giacomo Lagomarsini - Luca Miglior - Leonardo Stoppani
# Date: 2023-01-23
# License: MIT

import numpy as np

class Loss:
    def loss(self, pred, labels):
        raise NotImplementedError("I lay in the -- v o i d --")

    def backward(self, pred, labels):
        raise NotImplementedError("I lay in the -- v o i d --")


class MeanSquaredError(Loss):
    def loss(self, pred, labels):
        pred.shape = labels.shape
        return np.mean(np.square(pred - labels))

    def backward(self, pred, labels):
        return np.mean(pred - labels, keepdims=True, axis=1)


class BinaryCrossEntropy(Loss):
    """Binary crossentropy loss function"""

    def loss(self, pred, labels):
        return -(labels * np.log(pred) + (1 - labels) * np.log(1 - pred))

    def backward(self, pred, labels):
        return -(labels / pred - (1 - labels) / (1 - pred))
