from src.network import Network
from src.activations import ReLU, Sigmoid, Tanh
from src.losses import MeanSquaredError
from src.metrics import BinaryAccuracy
from src.utils import load_monk3, load_monk1, load_monk2
from src.regularizers import L2
import seaborn as sns
import matplotlib.pyplot as plt

# test monk 3
def test_network_monk3():

    # load monk3
    x_train, x_val, x_test, y_train, y_val, y_test = load_monk3(test_size=0.0001)

    binary_accuracy = BinaryAccuracy()

    model = Network(17, regularizer=L2(1e-8))
    model.add_layer(4, ReLU())
    model.add_layer(1, Sigmoid())

    stats = model.train(
        (x_train, y_train),
        (x_val, y_val),
        metric=binary_accuracy,
        loss=MeanSquaredError(),
        epochs=1000,
        eta=0.01,
        verbose=False,
    )

    # compute accuracy
    y_pred = model.multiple_outputs(x_test)
    acc = binary_accuracy(y_pred, y_test)
    assert acc > 0.8

# test monk 2
def test_network_monk2():
    
        # load monk2
        x_train, x_val, x_test, y_train, y_val, y_test = load_monk2(test_size=0.0001)
    
        binary_accuracy = BinaryAccuracy()

        model = Network(17)
        model.add_layer(3, ReLU())
        model.add_layer(1, Sigmoid())
    
        stats = model.train(
            (x_train, y_train),
            (x_val, y_val),
            metric=BinaryAccuracy(),
            loss=MeanSquaredError(),
            epochs=1000,
            eta=0.01,
            nesterov=0.7,
            verbose=False,
        )
    
        # compute accuracy
        y_pred = model.multiple_outputs(x_test)
        acc = binary_accuracy(y_pred, y_test)    
        assert acc == 1

# test monk 1
def test_network_monk1():

    # load monk1
    x_train, x_val, x_test, y_train, y_val, y_test = load_monk1(test_size=0.0001)

    binary_accuracy = BinaryAccuracy()

    model = Network(17)
    model.add_layer(4, ReLU())
    model.add_layer(1, Sigmoid(), initializer="xavier")

    stats = model.train(
        (x_train, y_train),
        (x_val, y_val),
        metric=binary_accuracy,
        loss=MeanSquaredError(),
        epochs=1000,
        eta=0.01,
        nesterov=0.8,
        verbose=False,
    )

    # compute accuracy
    y_pred = model.multiple_outputs(x_test)
    acc = binary_accuracy(y_pred, y_test)

    # show graph
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    sns.lineplot(stats["train_loss"], ax=axs[0], label="train-loss")
    sns.lineplot(stats["val_loss"], ax=axs[0], label="val-loss")
    sns.lineplot(stats["train_acc"], ax=axs[1], label="train-acc")
    sns.lineplot(stats["val_acc"], ax=axs[1], label="val-acc")

    plt.show()

    assert acc == 1
