import os
import pickle

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold

from utils.dataset import read_and_marge
from utils.preprocess import text2vec


def create_model(features):
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=features))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mean_squared_error'])
    return model


def load_training_data(tweet_xml, tweet_tsv, dataset_feature_path, feature_extractor):
    if os.path.isfile(dataset_feature_path):
        with open(dataset_feature_path, 'rb') as f:
            X, Y = pickle.load(f)

    else:
        dataset = read_and_marge(tweet_xml, tweet_tsv)
        X, Y = [], []
        for tweetid in dataset:
            tweet = dataset[tweetid]
            x, y = tweet.features(feature_extractor, numeric=True)
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


def train(tweet_xml, tweet_tsv, batch_size, epochs, dataset_feature_path, best_model, feature_extractor):
    X, Y, feature_length = load_training_data(tweet_xml, tweet_tsv, dataset_feature_path, feature_extractor)

    callbacks = [
        ModelCheckpoint("../data/check-points/epoch-{epoch:02d}.hdf5", monitor='mean_squared_error', verbose=1,
                        save_best_only=False,
                        save_weights_only=False,
                        mode='auto', period=1),
        ModelCheckpoint(best_model, monitor='mean_squared_error', verbose=1, save_best_only=True,
                        save_weights_only=False, mode='auto', period=1)]

    with tf.device('/gpu:0'):
        kfold = StratifiedKFold(n_splits=2)

        for index, (train, test) in enumerate(kfold.split(X, np.reshape([Y > 0.5], (Y.shape[0], 1)))):
            model = create_model(feature_length)
            model.fit(X[train], Y[train], batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=0)
            scores = model.evaluate(X[test], Y[test], verbose=1)
            print(scores)


def evaluation(weights_path):
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
        epochs = 1
        feature_extractor = text2vec
        train(tweet_xml, tweet_tsv, batch_size, epochs, dataset_feature_path, best_model, feature_extractor)
    else:
        evaluation(best_model)
