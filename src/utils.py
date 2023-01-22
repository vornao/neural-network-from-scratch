# Author: Giacomo Lagomarsini - Luca Miglior - Leonardo Stoppani
# Date: 2023-01-23
# License: MIT

import numpy as np
import pandas as pd
from itertools import zip_longest

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_moons, load_digits


def one_hot_encoder(data):
    datas = []
    for i in range(0, len(data)):
        a = np.zeros(10)
        a[data[i]] = 1
        datas.append(a)
    return np.array(datas)


class Error:
    def validate(self, y_true, y_pred):
        return y_true - y_pred


class MultiClassError(Error):
    def validate(self, y_true, y_pred):
        y_true = np.argmax(y_true)
        y_pred = np.argmax(y_pred)

        if y_true == y_pred:
            return 0
        else:
            return 1


class SingleClassError(Error):
    def validate(self, y_true, y_pred):
        """
        Loss function for moons dataset classifier.
        """
        y_pred = round(y_pred[0])

        if y_true == y_pred:
            return 0
        else:
            return 1


# load monk dataset


def load_monk1(test_size=0.2, validation=False):
    train = pd.read_csv("../data/monk/monks-1.train", sep=" ").drop(["a8"], axis=1)
    test = pd.read_csv("../data/monk/monks-1.test", sep=" ").drop(["a8"], axis=1)

    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(train.drop("a1", axis=1))

    X_train = enc.transform(train.drop("a1", axis=1)).toarray()
    y_train = train["a1"].values

    X_train.shape = (len(X_train), 17, 1)
    y_train.shape = (len(y_train), 1)

    enc.fit(test.drop("a1", axis=1))
    X_test = enc.transform(test.drop("a1", axis=1)).toarray()
    y_test = test["a1"].values

    X_test.shape = (len(X_test), 17, 1)
    y_test.shape = (len(y_test), 1)

    if not validation:
        return X_train, X_test, y_train, y_test

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=test_size, random_state=12
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# load monk2 dataset
def load_monk2(test_size=0.2, validation=False):
    train = pd.read_csv("../data/monk/monks-2.train", sep=" ").drop(["a8"], axis=1)
    test = pd.read_csv("../data/monk/monks-2.test", sep=" ").drop(["a8"], axis=1)

    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(train.drop("a1", axis=1))

    X_train = enc.transform(train.drop("a1", axis=1)).toarray()
    y_train = train["a1"].values

    X_train.shape = (len(X_train), 17, 1)
    y_train.shape = (len(y_train), 1)

    enc.fit(test.drop("a1", axis=1))
    X_test = enc.transform(test.drop("a1", axis=1)).toarray()
    y_test = test["a1"].values

    X_test.shape = (len(X_test), 17, 1)
    y_test.shape = (len(y_test), 1)

    if not validation:
        return X_train, X_test, y_train, y_test

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=test_size, random_state=12
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# load monk3 dataset
def load_monk3(test_size=0.2, validation=False):
    train = pd.read_csv("../data/monk/monks-3.train", sep=" ").drop(["a8"], axis=1)
    test = pd.read_csv("../data/monk/monks-3.test", sep=" ").drop(["a8"], axis=1)

    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(train.drop("a1", axis=1))

    X_train = enc.transform(train.drop("a1", axis=1)).toarray()
    y_train = train["a1"].values

    X_train.shape = (len(X_train), 17, 1)
    y_train.shape = (len(y_train), 1)

    enc.fit(test.drop("a1", axis=1))
    X_test = enc.transform(test.drop("a1", axis=1)).toarray()
    y_test = test["a1"].values

    X_test.shape = (len(X_test), 17, 1)
    y_test.shape = (len(y_test), 1)

    # train validation test split
    if not validation:
        return X_train, X_test, y_train, y_test

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=test_size, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_moons(
    n_samples=1000, test_size=0.1, random_state=42, noise=0.2, validation=True
):
    """
    Load the moons dataset.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    X = np.expand_dims(X, 2)
    y = np.expand_dims(y, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if validation:
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    return X_train, X_test, y_train, y_test


def load_mnist(test_size=0.2, scale=1, random_state=42, validation=True):
    digits = load_digits()

    X = digits.data
    X = (X / 16.0) * scale
    X = np.expand_dims(X, 2)
    y = digits.target

    # one hot encoding on y with sklearn
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(y.reshape(-1, 1))
    y = enc.transform(y.reshape(-1, 1)).toarray()
    y = np.expand_dims(y, 2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # train validation test split
    if validation:
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    return X_train, X_test, y_train, y_test


def load_cup(test_size=0.2, validation=True, scale_outputs=True):
    """
    Load cup dataset
    :param test_size: test size
    """

    df = pd.read_csv(
        "../data/cup/cup.internal.train",
        comment="#",
        index_col="id",
        skipinitialspace=True,
    )
    if scale_outputs:
        scaler = MinMaxScaler()
        df[["tx", "ty"]] = scaler.fit_transform(df[["tx", "ty"]])

    X = np.expand_dims(df.drop(["ty", "tx"], axis=1).values, 2)
    y = np.expand_dims(df[["tx", "ty"]].values, 2)

    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    if not validation:
        if scale_outputs:
            return X, y, scaler
        else:
            return X, y
    if scale_outputs:
        return x_train, x_val, y_train, y_val, scaler

    return x_train, x_val, y_train, y_val


def load_cup_test(scaler: MinMaxScaler = None):
    """
    Load cup dataset (internal test sed)
    """

    df = pd.read_csv(
        "../data/cup/cup.internal.test",
        comment="#",
        index_col="id",
        skipinitialspace=True,
    )

    if scaler is not None:
        df[["tx", "ty"]] = scaler.transform(df[["tx", "ty"]])

    X = np.expand_dims(df.drop(["ty", "tx"], axis=1).values, 2)
    y = np.expand_dims(df[["tx", "ty"]].values, 2)

    return X, y


def load_cup_blind_test(scaler: MinMaxScaler = None):
    """
    Load cup dataset
    """

    df = pd.read_csv(
        "../data/cup/cup.blind.test", comment="#", index_col=0, skipinitialspace=True
    )
    X = np.expand_dims(df.values, 2)
    return X


def parse_results(results: dict) -> pd.DataFrame:

    res = [val for val in results.values()]
    results = pd.DataFrame({}, columns=[])
    results["eta"] = [k[0]["eta"] for k in res]
    results["nesterov"] = [k[0]["nesterov"] for k in res]
    results["reg_type"] = [k[0]["reg_type"] for k in res]
    results["reg_val"] = [k[0]["reg_val"] for k in res]
    results["tr_mee"] = [k[1]["tr_mee"] for k in res]
    results["val_mee"] = [k[1]["val_mee"] for k in res]
    results["loss"] = [k[1]["losses"] for k in res]
    results["val_loss"] = [k[1]["val_losses"] for k in res]
    return results
