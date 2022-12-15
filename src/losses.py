import numpy as np
import src.regularizer as reg

class Loss:
    """
    Base class for losses.
    regularizer: Regularizer - regularizer to be used
    """
    #define class constructor with regularization
    def __init__(self, regularizer: reg.Regularizer = None):
        self.regularizer = regularizer

    def loss(self, pred, labels):
        raise NotImplementedError("I lay in the -- v o i d --")

    def backward(self, pred, labels):
        raise NotImplementedError("I lay in the -- v o i d --")


class MeanSquaredError(Loss):
    """
    Mean squared error loss.
    """
    def loss(self, pred, labels):
        pred.shape = labels.shape
        if self.regularizer:
            return (np.mean(np.square(pred - labels)) + self.regularizer.reg()) #reshape to 1x1 matrix to avoid problems with np.dot
        else:
            return np.mean(np.square(pred - labels))

    def backward(self, pred, labels):
        if self.regularizer:
            return (np.mean(pred - labels) + self.regularizer.backward())
        else:
            return np.mean(pred - labels)
        
   

# define binary crossentropy
class BinaryCrossEntropy(Loss):

    def loss(self, pred, labels):
        return -(labels * np.log(pred) + (1 - labels) * np.log(1 - pred))

    def backward(self, pred, labels):
        # derivative binary crossentropy
        return -(labels / pred - (1 - labels) / (1 - pred))
