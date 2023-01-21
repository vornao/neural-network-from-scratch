import numpy as np


def sigmoid(x, a=1):
    return 1 / (1 + np.exp(-1 * a * x))


def relu(x):
    return np.maximum(0, x)


def sigmoid_d(x, a=1):
    return sigmoid(x, a) * (1 - sigmoid(x, a))


def relu_d(x):
    return x > 0


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def sofmax_d(x):
    return softmax(x) * (1 - softmax(x))


def linear(x):
    return x


def linear_d(x):
    return 1


class Activation:
    def activation(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class Linear(Activation):
    def activation(self, x):
        return x

    def derivative(self, x):
        return 1


class Sigmoid(Activation):
    def __init__(self, a=1) -> None:
        self.a = a

    def activation(self, x):
        return sigmoid(x, self.a)

    def derivative(self, x):
        return sigmoid_d(x, self.a)


class ReLU(Activation):
    def activation(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return relu_d(x)


class Softmax(Activation):
    def activation(self, x):
        return np.exp(x) / np.exp(x).sum()

    def derivative(self, x):
        return np.exp(x) / np.exp(x).sum()


class Tanh(Activation):
    def activation(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x) ** 2
