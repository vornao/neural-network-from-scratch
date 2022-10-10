import numpy as np


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


TRAIN_FMT = """Epoch : {} -  Training Loss: {} - Validaition Loss {}\n------------------------------------------------------------"""
