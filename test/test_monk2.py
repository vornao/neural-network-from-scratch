from src.network import Network
from src.activations import ReLU, Tanh
from src.losses import MeanSquaredError
from src.metrics import BinaryAccuracy
from src.utils import load_monk2
from src.regularizers import L2
import seaborn as sns
import matplotlib.pyplot as plt

# test monk 3
def test_network_monk3():
    
    # load monk3
    x_train, x_val, x_test, y_train, y_val, y_test = load_monk2()

    binary_accuracy = BinaryAccuracy()

    model = Network(17)
    model.add_layer(6, ReLU(), initializer="xavier")
    model.add_layer(1, Tanh(), initializer="xavier")

    stats = model.train((x_train, y_train), (x_val, y_val),
        metric=binary_accuracy,
        loss=MeanSquaredError(),
        epochs=1000,
        verbose=False)

    # compute accuracy
    y_pred = model.multiple_outputs(x_val)
    acc = binary_accuracy(y_pred, y_val)

    # show graph
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    sns.lineplot(stats['train_loss'], ax=axs[0], label='train-loss')
    sns.lineplot(stats['val_loss'], ax=axs[0], label='val-loss')
    sns.lineplot(stats['train_acc'], ax=axs[1], label='train-acc')
    sns.lineplot(stats['val_acc'], ax=axs[1], label='val-acc')

    plt.show()

    assert acc == 1