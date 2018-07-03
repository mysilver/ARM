import os
import pickle

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix

from utils.dataset import read_and_marge


def create_model(features):
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=features))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


def load_training_data(tweet_xml, tweet_tsv, dataset_feature_path):
    if os.path.isfile(dataset_feature_path):
        with open(dataset_feature_path, 'rb') as f:
            X, Y = pickle.load(f)

    else:
        dataset = read_and_marge(tweet_xml, tweet_tsv)
        X, Y = [], []
        for tweetid in dataset:
            tweet = dataset[tweetid]
            x, y = tweet.features(feature_extractor)
            X.append(x)
            Y.append(y)
        with open(dataset_feature_path, 'wb') as f:
            pickle.dump((X, Y), f)

    X = np.array(X)
    Y = np.array(Y)
    feature_length = X.shape[1]
    print("Input :", X.shape)
    print("Output :", Y.shape)
    return X, Y, feature_length


def train(tweet_xml, tweet_tsv, batch_size, epochs, dataset_feature_path, best_model):
    X, Y, feature_length = load_training_data(tweet_xml, tweet_tsv, dataset_feature_path)

    callbacks = [
        ModelCheckpoint("../data/check-points/epoch-{epoch:02d}.hdf5", monitor='val_loss', verbose=1,
                        save_best_only=False,
                        save_weights_only=False,
                        mode='auto', period=1),
        ModelCheckpoint(best_model, monitor='val_loss', verbose=1, save_best_only=True,
                        save_weights_only=False, mode='auto', period=1)]

    with tf.device('/gpu:0'):
        kfold = StratifiedKFold(Y, n_folds=5)
        cvscores = []
        for train, test in kfold:
            model, _ = create_model(feature_length)
            model.fit(X[train], Y[train], batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=0)
            scores = model.evaluate(X[test], Y[test], verbose=1)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            # print(model.metrics)
            # print(scores)
            cvscores.append(scores[1] * 100)
            y_pred = model.predict(X[test])
            y_pred = (y_pred > 0.5)
            cm = confusion_matrix(Y[test], y_pred)
            print("confusion_matrix:\n", cm)

        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


def test(weights_path):
    feature_length = 100
    model, _ = create_model(feature_length)
    model.load_weights(weights_path)
    # TODO : we need to see what is the testing data

if __name__ == '__main__':
    train_mode = True
    best_model = "../data/check-points/best-epoch.hdf5"
    if train_mode:
        tweet_xml = "../data/tweets-sample.xml"
        tweet_tsv = "../data/tweets-sample.tsv"
        dataset_feature_path = "../data/tweet-features.pickle"
        batch_size = 64
        epochs = 10000
        train(tweet_xml, tweet_tsv, batch_size, epochs, dataset_feature_path, best_model)
    else:
        test(best_model)
