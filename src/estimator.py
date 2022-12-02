
# custom estimator for sklearn

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

from src.network import Network
from src.activations import Tanh, ReLU
from src.losses import MeanSquaredError
from src.metrics import BinaryAccuracy

import numpy as np

class NeuralNetwork(BaseEstimator, ClassifierMixin):

    def __init__(self, input_shape=5, output_shape = 1, units_per_layer=None, activation=ReLU(), loss=MeanSquaredError(), metric=BinaryAccuracy(), epochs=100, eta=0.01, batch_size=1, verbose=False):

        self.units_per_layer = units_per_layer
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation
        self.loss = loss
        self.metric = metric
        self.epochs = epochs
        self.eta = eta
        self.batch_size = batch_size
        self.verbose = verbose


    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.model = Network(self.n_features_)

        for layer in self.units_per_layer:
            self.model.add_layer(layer, self.activation)

        self.model.add_layer(self.output_shape, Tanh())

        self.stats = self.model.train(
            X,
            y,
            val_labels=None,
            val_data=None,
            metric=self.metric,
            loss=self.loss,
            epochs=self.epochs,
            eta=self.eta,
            verbose=self.verbose,
            batch_size=self.batch_size
        )

        return self

    def predict(self, X):
        y_pred = self.model.multiple_outputs(X)
        y_pred = np.round(y_pred)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        return acc