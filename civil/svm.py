from keras.engine.saving import model_from_yaml
import numpy
from keras import backend as K
import tensorflow as tf
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier


def load_cnn_model(yaml_path, weights_path):
    yaml_file = open(yaml_path, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights(weights_path)
    # print("Loaded model from disk")
    return loaded_model


def load_test_set(x_path, y_path):
    x = numpy.load(x_path)
    y = numpy.argmax(numpy.load(y_path), 1)  # Convert back from to_categorical
    return x, y


if __name__ == "__main__":

    with tf.device("/cpu:0"):
        for fold in range(1, 6):
            yaml_path = "./trained_models/model_fold_{}.yaml".format(fold)
            weights_path = "./trained_models/model_fold_{}.h5".format(fold)

            model = load_cnn_model(yaml_path, weights_path)

            x_path = "./trained_models/test_x_{}.npy".format(fold)
            y_path = "./trained_models/test_y_{}.npy".format(fold)
            x_test, y_test = load_test_set(x_path, y_path)

            x_path = "./trained_models/train_x_{}.npy".format(fold)
            y_path = "./trained_models/train_y_{}.npy".format(fold)
            x_train, y_train = load_test_set(x_path, y_path)

            outputs = [layer.output for layer in model.layers]
            functor = K.function([model.input, K.learning_phase()], outputs)

            last_layer_index = 4
            lastlayer_output = functor([x_train, 1.])[last_layer_index]

            test_lastlayer = functor([x_test.tolist(), 1.])[last_layer_index]

            classifiers = {
                "SVM": svm.SVC(kernel='poly', decision_function_shape='ovo'),
                "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0),
                "Decision Tree": tree.DecisionTreeClassifier(),
                "KNN": KNeighborsClassifier(n_neighbors=100)
            }
            print("*****  Fold {} *****".format(fold))
            for name in classifiers:
                clf = classifiers[name]
                clf.fit(lastlayer_output.tolist(), y_train.tolist())
                test_results = clf.predict(test_lastlayer) == y_test
                accuracy = numpy.average(clf.predict(test_lastlayer) == y_test) * 100
                print("Accuracy{}:".format(name), accuracy)
