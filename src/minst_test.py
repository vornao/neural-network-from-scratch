# %%
import seaborn as sns
import pandas as pd
import numpy as np

import sys

sys.path.insert(0, "../")
sys.path.insert(0, "../src")

from utils import MultiClassError, one_hot_encoder
import network as network
from activation_functions import Sigmoid, ReLU
from losses import MeanSquaredError


from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)


mnist_train = pd.read_csv(
    "/Users/vornao/Desktop/unipi/ML/code/first_nn/data/mnist/train.csv"
)

y_train = one_hot_encoder(mnist_train[:30000]["label"].to_numpy())
x_train = mnist_train[:30000].drop(["label"], axis=1).to_numpy()
x_train = (x_train / 255) * 0.1  # type: ignore

y_valid = one_hot_encoder(mnist_train[30000:36000]["label"].to_numpy())
x_valid = mnist_train[30000:36000].drop(["label"], axis=1).to_numpy()
x_valid = (x_valid / 255) * 0.1  # type: ignore

y_test = one_hot_encoder(mnist_train[30000:36000]["label"].to_numpy())
x_test = mnist_train[30000:36000].drop(["label"], axis=1).to_numpy()
x_test = (x_test / 255) * 0.1  # type: ignore


# %%
net = network.Network(28 * 28)
net.add_layer(16, ReLU())
net.add_layer(8, ReLU())
net.add_output_layer(10, Sigmoid(a=0.05))

stats = net.train(
    x_train,
    y_train,
    x_valid,
    y_valid,
    metric=MultiClassError(),
    loss=MeanSquaredError(),
    epochs=30,
    eta=0.01,
    verbose=True,
)

# %%
predictions = []

for p in x_test:
    predictions.append(np.round(net.output(p)))

predictions = np.array(predictions)


accuracy = round(accuracy_score(y_test, predictions), 2) * 100
roc = round(roc_auc_score(y_test, predictions), 2)

print(f"> Model accuracy: {accuracy}%. ROC AUC Score: {roc}")

losses = pd.DataFrame({"Training Loss": stats[0], "Validation Loss": stats[1]})

palette = sns.color_palette("rocket_r", 2)
sns.set_style("darkgrid")
sns.lineplot(losses, palette=palette)


# %%
labels_pred = np.argmax(predictions, 1)
labels_true = np.argmax(y_test, 1)

C = confusion_matrix(labels_pred, labels_true)
sns.heatmap(C, annot=True).show()
