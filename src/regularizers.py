# Author: Giacomo Lagomarsini - Luca Miglior - Leonardo Stoppani
# Date: 2023-01-23
# License: MIT

import numpy as np


class Regularizer:
    def __init__(self, lamda):
        self.lamda = lamda

    def __call__(self, weights):
        raise NotImplementedError

    def gradient(self, weights):
        raise NotImplementedError


class L1(Regularizer):
    def __call__(self, weights):
        return self.lamda * np.sum(np.abs(weights))

    def gradient(self, weights):
        return self.lamda * np.sign(weights)


class L2(Regularizer):
    def __call__(self, weights):
        return self.lamda * np.sum(weights**2)

    def gradient(self, weights):
        return 2 * self.lamda * weights
