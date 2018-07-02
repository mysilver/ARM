import os
import pickle

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix

from utils.preprocess import normalize


def baseline_model(features=21):
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=features))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


batch_size = 64
epochs = 10000
callbacks = []
features_per_tweet = 20
dataset_feature_path = "data/tweet-features.pickle"

if os.path.isfile(dataset_feature_path):
    with open(dataset_feature_path, 'rb') as f:
        X, Y = pickle.load(f)
        X = np.array(X)
        Y = np.array(Y)

else:
    dataset = read_paraphrased_tsv_files(
        "/media/may/Data/LinuxFiles/PycharmProjects/PhD/paraphrasing-data/crowdsourced",
        processor=normalize)
    X, Y = extract_features(dataset, save=dataset_feature_path)

print("Input :", X.shape)
print("Output :", Y.shape)


with tf.device('/gpu:0'):

    kfold = StratifiedKFold(Y, n_folds=5)
    cvscores = []
    for train, test in kfold:
        model, _ = baseline_model()
        model.fit(X[train], Y[train], batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=0)
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        # print(model.metrics)
        # print(scores)
        cvscores.append(scores[1] * 100)
        y_pred = model.predict(X[test])
        y_pred = (y_pred > 0.5)
        cm = confusion_matrix(Y[test], y_pred)
        print("confusion_matrix:\n", cm)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))