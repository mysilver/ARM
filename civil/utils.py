import numpy as np
import scipy.io
import math

from keras.utils import np_utils


def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def convert_matlab_file(samples, number_of_samples_per_class,signal_length, number_of_sensors, matlab_file, save_path):
    x = scipy.io.loadmat(matlab_file)

    sensors = np.transpose(x['InputData'], (3, 0, 1, 2)) # reorder the file
    print(sensors.shape)
    sensors = sensors.reshape(samples, signal_length, number_of_sensors)
    X_train = sensors
    Y_train = np.array([math.ceil(i / number_of_samples_per_class) -1 for i in range(1, 241)])
    Y_train = np_utils.to_categorical(Y_train)
    X_train, Y_train = shuffle(X_train, Y_train)
    np.savez(save_path, X_train, Y_train)
    return X_train, Y_train


def load_pickle(path):
    t = np.load(path + ".npz")
    return t['arr_0'], t['arr_1']
