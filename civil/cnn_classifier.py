import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten
from keras.utils.vis_utils import plot_model

from civil.utils import convert_matlab_file, load_dataset

if __name__ == "__main__":

    try:
        print("Loading the data")
        path = "civil.pickle"
        X_train, Y_train = load_dataset(path)
        print(X_train.shape, "==>", Y_train.shape)
    except:
        print("Converting Matlab file to python compatible file")
        X_train, Y_train = convert_matlab_file(240, 'Bridge_01.mat', path)

    with tf.device("/gpu:1"):
        sensor_size = 5001
        # sensor_size, X_train = reduce_last_dimension(X_train, sensor_size)
        model = Sequential()
        print("Sensor Shape (after reduction):", X_train.shape)
        model.add(Conv1D(filters=20, kernel_size=2, input_shape=(sensor_size, 28)))
        model.add(MaxPool1D())
        model.add(Conv1D(filters=20, kernel_size=2, input_shape=(sensor_size, 28)))
        model.add(MaxPool1D())
        model.add(Flatten())
        model.add(Dense(6, activation="sigmoid"))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        model.summary()

        model.fit(X_train, Y_train, epochs=500, batch_size=64)
