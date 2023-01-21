from src.utils import load_mnist, load_moons, load_monk1


def test_load_monk1():
    X_train, X_val, X_test, y_train, y_val, y_test = load_monk1()
    assert X_train.shape[1] == 17
    assert X_train.shape[2] == 1

    assert X_val.shape[1] == 17
    assert X_val.shape[2] == 1

    assert X_test.shape[1] == 17
    assert X_test.shape[2] == 1

    assert y_train.shape[1] == 1
    assert y_val.shape[1] == 1
    assert y_test.shape[1] == 1


def test_make_moons():
    x_train, x_val, x_test, y_train, y_val, y_test = load_moons(1000, noise=0.1)

    assert x_train.shape[1] == 2
    assert x_train.shape[2] == 1

    assert x_val.shape[1] == 2
    assert x_val.shape[2] == 1

    assert x_test.shape[1] == 2
    assert x_test.shape[2] == 1

    assert y_train.shape[1] == 1
    assert y_val.shape[1] == 1


def test_load_mnist():

    x_train, x_val, x_test, y_train, y_val, y_test = load_mnist()

    assert x_train.shape[1] == 64
    assert x_train.shape[2] == 1

    assert x_val.shape[1] == 64
    assert x_val.shape[2] == 1

    assert x_test.shape[1] == 64
    assert x_test.shape[2] == 1

    assert y_train.shape[1] == 10
    assert y_val.shape[1] == 10
    assert y_train.shape[2] == 1
    assert y_val.shape[2] == 1
