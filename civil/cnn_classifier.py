import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten
# from keras.utils.vis_utils import plot_model
from sklearn.cross_validation import StratifiedKFold
from civil.utils import convert_matlab_file, load_pickle
import numpy


def create_model():
    model = Sequential()
    print("Sensor Shape (after reduction):", X_train.shape)
    model.add(Conv1D(filters=20, kernel_size=2, input_shape=(signal_length, number_of_sensors)))
    model.add(MaxPool1D())
    model.add(Conv1D(filters=20, kernel_size=2, input_shape=(signal_length, number_of_sensors)))
    model.add(MaxPool1D())
    model.add(Flatten())
    model.add(Dense(number_of_classes, activation="sigmoid"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    model.summary()
    return model


if __name__ == "__main__":

    number_of_samples = 240
    number_of_samples_per_class = 40
    number_of_classes = int(number_of_samples / number_of_samples_per_class)
    signal_length = 5001  # the length of signal
    number_of_sensors = 28
    epochs = 500
    batch_size = 64
    path = "civil.pickle"

    try:
        print("Loading the data")
        X_train, Y_train = load_pickle(path)
        print(X_train.shape, "==>", Y_train.shape)
    except:
        print("Converting Matlab file to python compatible file")
        X_train, Y_train = convert_matlab_file(number_of_samples, number_of_samples_per_class, signal_length,
                                               number_of_sensors, 'Bridge_01.mat', path)

    # sensor_size, X_train = reduce_last_dimension(X_train, sensor_size)
    with tf.device("/gpu:0"):
        print("Training is started")
        kfold = StratifiedKFold(numpy.argmax(Y_train, axis=-1), n_folds=5)
        for train, test in kfold:
            model = create_model()
            model.fit(X_train[train], Y_train[train], epochs=epochs, batch_size=batch_size)
            scores = model.evaluate(X_train[test], Y_train[test], verbose=1)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
