import numpy as np

class Regularizer:
    """
    Base class for regularizers.
    lamda: float - regularization strength
    network: Network - network to be regularized
    """
    def __init__(self, lamda: float, network):
        self.lamda = lamda
        self.network = network

    def reg(self):
        raise NotImplementedError("I lay in the -- v o i d --")

    def backward(self):
        raise NotImplementedError("I lay in the -- v o i d --")

class L1(Regularizer):
    """
    L1 regularization.
    """
    def reg(self):
        return self.lamda * np.sum(np.abs(self.network.layers[-1].W))

    def backward(self):
        #non funziona peche' la shape non e' corretta, ritorno 16x1 invece di 1x1, ma la derivata sarebbe questa 
        return self.lamda * np.sign(self.network.layers[-1].W) 
    
class L2(Regularizer):
    """ 
    L2 regularization.
    """
    def reg(self):
        return self.lamda * np.sum(np.square(self.network.layers[-1].W))

    def backward(self):
        #qui faccio la media per poter restituire una shape corretta, ma non so se è corretto
        return 2 * self.lamda * np.mean(self.network.layers[-1].W)

