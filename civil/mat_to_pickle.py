import math

import numpy as np
import scipy.io
from keras.utils import np_utils


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


x = scipy.io.loadmat('matlab.mat')
sensors = np.transpose(x['InputData'], (3, 0, 1, 2))
print(sensors.shape)
sensors = sensors.reshape(1440, 5001, 28)

X_train = sensors
Y_train = np.array([math.ceil(i / 40) for i in range(1, 1441)])
# X_train, Y_train = unison_shuffled_copies(X_train, Y_train)

Y_train = np_utils.to_categorical(Y_train)
X_train, Y_train = unison_shuffled_copies(X_train, Y_train)
np.savez("civil-36.pickle", X_train, Y_train)