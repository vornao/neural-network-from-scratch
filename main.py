import network as network
import pandas as pd
import numpy as np
from activation_functions import Sigmoid, ReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from utils import one_hot_encoder, MultiClassError, SingleClassError



def test_mnist():
    mnist_train = pd.read_csv('data/mnist/train.csv')

    """
    
    """

    y = mnist_train['label'].to_numpy()
    x = mnist_train.drop(['label'], axis=1).to_numpy()
    y = one_hot_encoder(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    x_train = (x_train / 255)*.1  # type: ignore
    x_test  = (x_test / 255)*.1   # type: ignore

    net = network.Network(28*28)
    net.add_layer(10, ReLU())
    net.add_output_layer(10, Sigmoid(a=0.05))
    net.train(x_train, y_train, MultiClassError(), epochs=10, eta=10e-2)

    predictions = []

    for p in x_test:
        pred = net.output(p)
        predictions.append(np.round(pred))
    
    predictions = np.array(predictions)
    
    accuracy = round(accuracy_score(y_test, predictions), 2)*100
    roc = round(roc_auc_score(y_test, predictions), 2)

    print(f'> Model accuracy: {accuracy}%. ROC AUC Score: {roc}')


def test_moons():
    moons = pd.read_csv("data/moons.csv")
    x = moons[["x", "y"]].to_numpy()
    y = moons["label"].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )

    net = network.Network(2)
    net.add_layer(16, "relu")
    net.add_layer(8, "relu")
    net.add_output_layer(1, "sigmoid")
    net.train(x_train, y_train, 100)

    predictions = []

    for p in x_test:
        predictions.append(round(net.output(p)[0]))

    predictions = np.array(predictions)

    print(f"ROC AUC SCORE :{roc_auc_score(y_test, predictions)}")



if __name__ == "__main__":

    # test_moons()
    test_mnist()

 
    
