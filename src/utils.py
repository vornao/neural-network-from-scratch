import numpy as np
import pandas as pd
from itertools import zip_longest

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

TRAIN_FMT = """\nEpoch : {} -  Training Loss: {} - Validaition Loss {}\n------------------------------------------------------------"""

def chunker(iterable, n, fillvalue=None):
    """
    Return iterator for iterate over chunked iterable
    @paramd
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
    train = pd.read_csv('../data/monks-1.train', sep=' ').drop(['a8'], axis=1)
    test = pd.read_csv('../data/monks-1.test', sep=' ').drop(['a8'], axis=1)

    # perform one-hot encoding with sklearn 

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(train.drop('a1', axis=1))

    X_train = enc.transform(train.drop('a1', axis=1)).toarray()
    y_train = train['a1'].values

    enc.fit(test.drop('a1', axis=1))
    X_test = enc.transform(test.drop('a1', axis=1)).toarray()
    y_test = test['a1'].values

    # train validation test split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test
    






