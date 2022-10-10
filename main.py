import network as network
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from activation_functions import Sigmoid, ReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from utils import one_hot_encoder, MultiClassError, SingleClassError



def test_mnist():
    mnist_train = pd.read_csv('data/mnist/train.csv')

    y_train = one_hot_encoder(mnist_train[:30000]['label'].to_numpy())
    x_train = mnist_train[:30000].drop(['label'], axis=1).to_numpy()
    x_train = (x_train / 255)*.1  # type: ignore

    y_valid = one_hot_encoder(mnist_train[30000:36000]['label'].to_numpy())
    x_valid = mnist_train[30000:36000].drop(['label'], axis=1).to_numpy()
    x_valid = (x_valid / 255)*.1  # type: ignore

    y_test = one_hot_encoder(mnist_train[30000:36000]['label'].to_numpy())
    x_test = mnist_train[30000:36000].drop(['label'], axis=1).to_numpy()
    x_test = (x_test / 255)*.1  # type: ignore


    net = network.Network(28*28)
    net.add_layer(10, ReLU())
    net.add_output_layer(10, Sigmoid(a=0.05))
    net.train(x_train, y_train, x_valid, y_valid, MultiClassError(), epochs=10, eta=10e-2)

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

    x_train, x_test_val, y_train, y_test_val = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_test_val, y_test_val, test_size = .25
    )


    net = network.Network(2)
    net.add_layer(8, ReLU())
    net.add_output_layer(1, Sigmoid(a=1))

    stats = net.train(
        x_train, y_train, x_val, y_val, 
        estimator=SingleClassError(),
        eta=10e-2,
        epochs=25
    )

    predictions = []

    for p in x_test:
        predictions.append(round(net.output(p)[0]))

    predictions = np.array(predictions) 

    accuracy = round(accuracy_score(y_test, predictions), 2)*100
    roc = round(roc_auc_score(y_test, predictions), 2)

    print(f'> Model accuracy: {accuracy}%. ROC AUC Score: {roc}')

    sns.lineplot(stats[0])
    sns.lineplot(stats[1])
    plt.show()


if __name__ == "__main__":

    # test_moons()
    test_mnist()

 
    
