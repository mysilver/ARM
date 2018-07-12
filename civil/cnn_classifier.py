import math

import numpy as np
from keras import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten

from civil.utils import convert_matlab_file, load_dataset


def reduce_dimension(sensor, sensor_size):
    size = sensor.shape[0]
    step_size = math.ceil(size / sensor_size)
    ret = []
    start = 0
    end = 0
    while end <= size:
        end = end + step_size
        max = np.max(sensor[start:end])
        ret.append(max)
        start = end

    return ret


def reduce_last_dimension(sensors, sensor_size):
    sensors = np.transpose(sensors, (0, 2, 1))
    print("reducing dimension for ", sensors.shape)
    ret = []
    dimension_after_reduction = 0
    for sample_index in range(sensors.shape[0]):
        sample_array = []
        for sensor_index in range(sensors[sample_index].shape[0]):
            sensor = sensors[sample_index][sensor_index]
            sensor = reduce_dimension(sensor, sensor_size)
            dimension_after_reduction = len(sensor)
            sample_array.append(sensor)
        ret.append(sample_array)
    return dimension_after_reduction, np.transpose(np.array(ret), (0, 1, 2))


if __name__ == "__main__":

    try:
        print("Loading the data")
        path = "civil.pickle"
        X_train, Y_train = load_dataset(path)
        print(X_train.shape, "==>", Y_train.shape)
    except:
        print("Converting Matlab file to python compatible file")
        X_train, Y_train = convert_matlab_file('Bridge_01.mat', path)

    sensor_size = 200
    sensor_size, X_train = reduce_last_dimension(X_train, sensor_size)
    model = Sequential()
    print("Sensor Shape (after reduction):", X_train.shape)
    model.add(Conv1D(filters=1, kernel_size=2, input_shape=(28, sensor_size)))
    model.add(MaxPool1D())
    model.add(Conv1D(filters=1, kernel_size=2, input_shape=(28, sensor_size)))
    model.add(MaxPool1D())
    model.add(Flatten())
    model.add(Dense(6, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    model.summary()

    model.fit(X_train, Y_train, epochs=50000, batch_size=256)
