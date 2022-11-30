import numpy as np


class Loss:
    def loss(self):
        raise NotImplementedError("I lay in the -- v o i d --")

    def dloss(self):
        raise NotImplementedError("I lay in the -- v o i d --")


class MeanSquaredError(Loss):
    def loss(self, pred, labels):
        return np.mean(np.power(pred - labels, 2))

    def dloss(self, pred, labels):
        return pred - labels
   
