from src.utils import MultiClassError, SingleClassError, one_hot_encoder, load_monk1, load_moons
import src.network as network
from src.activations import Sigmoid, ReLU, Tanh, Softmax
from src.losses import MeanSquaredError, BinaryCrossEntropy
from src.metrics import BinaryAccuracy

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
)

import numpy as np


if __name__ == '__main__':

    # load monk1
    x_train, x_val, x_test, y_train, y_val, y_test = load_monk1()

    y_train.shape = (y_train.shape[0], 1)
    y_val.shape = (y_val.shape[0], 1)

    model = network.Network(17)
    model.add_layer(8, ReLU())
    model.add_layer(1, Tanh())

    stats = model.train(
        x_train,
        y_train,
        x_val,
        y_val,
        metric=BinaryAccuracy(),
        loss=MeanSquaredError(),
        epochs=500,
        eta=0.01,
        verbose=True,
        batch_size=1
    )


    # plot stats with seaborn
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.lineplot(data=stats['train_loss'])
    sns.lineplot(data=stats['val_loss'])
    sns.lineplot(data=stats['val_error'])
    plt.show()

    # compute accuracy
    y_pred = model.multiple_outputs(x_val)
    y_pred = np.round(y_pred)
    acc = accuracy_score(y_val, y_pred)

    # print acc e tacc
    print(f"Model trained. Validation Accuracy: {acc}")
