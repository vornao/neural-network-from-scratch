import numpy as np
import pandas as pd
from itertools import zip_longest

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


from sklearn.datasets import make_moons, load_digits

def chunker(iterable, n, fillvalue=None):
    """
    Return iterator for iterate over chunked iterable
    @params iterable: iterable to be chunked
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


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

def load_monk1(test_size=0.2):
    train = pd.read_csv('../data/monk/monks-1.train', sep=' ').drop(['a8'], axis=1)
    test = pd.read_csv('../data/monk/monks-1.test', sep=' ').drop(['a8'], axis=1)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(train.drop('a1', axis=1))

    X_train = enc.transform(train.drop('a1', axis=1)).toarray()
    y_train = train['a1'].values

    X_train.shape = (len(X_train), 17, 1)
    y_train.shape = (len(y_train), 1)

    enc.fit(test.drop('a1', axis=1))
    X_test = enc.transform(test.drop('a1', axis=1)).toarray()
    y_test = test['a1'].values

    X_test.shape = (len(X_test), 17, 1)
    y_test.shape = (len(y_test), 1)

    # train validation test split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test
    
#load monk3 dataset
def load_monk3(test_size=0.2):
    train = pd.read_csv('../data/monk/monks-3.train', sep=' ').drop(['a8'], axis=1)
    test = pd.read_csv('../data/monk/monks-3.test', sep=' ').drop(['a8'], axis=1)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(train.drop('a1', axis=1))

    X_train = enc.transform(train.drop('a1', axis=1)).toarray()
    y_train = train['a1'].values

    X_train.shape = (len(X_train), 17, 1)
    y_train.shape = (len(y_train), 1)

    enc.fit(test.drop('a1', axis=1))
    X_test = enc.transform(test.drop('a1', axis=1)).toarray()
    y_test = test['a1'].values

    X_test.shape = (len(X_test), 17, 1)
    y_test.shape = (len(y_test), 1)

    # train validation test split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

# make moons dataset

def load_moons(n_samples=1000, test_size=0.1, random_state=42, noise=0.2, validation=True):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    X = np.expand_dims(X, 2)
    y = np.expand_dims(y, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if validation:
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test


    return X_train, X_test, y_train, y_test


def load_mnist(test_size=0.2, scale=1, random_state=42, validation=True):
    digits = load_digits()

    X = digits.data
    X = (X / 16.0) * scale
    X = np.expand_dims(X, 2)
    y = digits.target

    # one hot encoding on y with sklearn
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(y.reshape(-1, 1))
    y = enc.transform(y.reshape(-1, 1)).toarray()
    y = np.expand_dims(y, 2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # train validation test split
    if validation:
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test

    return X_train, X_test, y_train, y_test




def load_cup(test_size=0.2):
    df = pd.read_csv("../data/cup/cup.train", comment="#", index_col='id', skipinitialspace=True)
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    x_train, x_val, y_train, y_val = train_test_split(scaled.drop(["ty", 'tx'], axis=1).values, scaled[['tx','ty']].values, test_size=0.25, random_state=42)
    x_train = np.expand_dims(x_train, 2) 
    print(x_train)