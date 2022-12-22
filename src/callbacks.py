import numpy as np


class Callback:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented in the child class")


# TODO: restore best weights
class EarlyStopping(Callback):

    def __init__(self, func, patience, restore_best_weights=True, *args, **kwargs):
        super().__init__(func, *args, **kwargs)
        self.patience = patience
        self.best_loss = np.inf
        self.counter = 0

    def __call__(self, loss: float):

        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.func(*self.args, **self.kwargs)