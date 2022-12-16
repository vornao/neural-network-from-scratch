from src.network import Network
from src.activations import ReLU, Tanh
from src.losses import MeanSquaredError
from src.regularizer import L1, L2
from src.metrics import BinaryAccuracy
from src.utils import load_monk3
import seaborn as sns
import matplotlib.pyplot as plt

def test_network_monk3():

    # load monk1
    x_train, x_val, x_test, y_train, y_val, y_test = load_monk3()

    binary_accuracy = BinaryAccuracy()

    model = Network(17)
    model.add_layer(10, ReLU())
    model.add_layer(1, Tanh())

    stats = model.train((x_train, y_train), (x_val, y_val),
        metric=binary_accuracy,
        loss=MeanSquaredError(L2(0.005,model)),
        epochs=1500,
        verbose=True,
        eta=0.0001)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    sns.lineplot(stats['train_loss'], ax=axs[0])
    sns.lineplot(stats['val_loss'], ax=axs[0])
    sns.lineplot(stats['train_acc'], ax=axs[1])
    sns.lineplot(stats['val_acc'], ax=axs[1])

    plt.show()

    # compute accuracy
    y_pred = model.multiple_outputs(x_val)
    acc = binary_accuracy(y_pred, y_val)

    assert acc == 1