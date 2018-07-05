import xml.etree.ElementTree

import numpy
import pickle

from empath import Empath
from nltk.corpus import stopwords

from tweet import Tweet
from utils.preprocess import text2vec, pos_tag, sentiment, tokenize
from utils.word2vec import word_vector

stop_words = set(stopwords.words('english'))
EmpathCat = Empath()


def read_tsv(file):
    """
    Reads a TSV file 
    :param file: a TSV file
    :return: list of processed lines
    """
    ret = []
    with open(file, 'rt') as f:
        for line in f.readlines():
            temp = []
            for item in line.split('\t'):
                temp.append(item)
            ret.append(temp)
    return ret


def tweet_xml_reader(file, filter, dictionary=False):
    """
    :param file: Tweet XML file path 
    :param dictionary: returns a dictionary (TweetID, Tweet) if set to True; and a list of Tweets if it is false
    :return: a dictionary or list based on the dictionary parameter
    """
    e = xml.etree.ElementTree.parse(file).getroot()

    if dictionary:
        ret = {}
    else:
        ret = []
    for atype in e.findall('Tweet'):
        id = atype.find('TweetID').text.strip()
        text = atype.find('TweetText').text.strip()
        time = atype.find('TweetTimeStamp').text.strip()
        location = atype.find('TweetLocation').text.strip()
        userid = atype.find('UserID').text.strip()
        friends = int(atype.find('UserFriendsCount').text.strip())
        followers = int(atype.find('UserFollowersCount').text.strip())
        favorites = int(atype.find('UserFavoritesCount').text.strip())
        tweet = Tweet(id=id, user_id=userid, timestamp=time, location=location, text=text, friends=friends,
                      followers=followers, favorites=favorites)

        if tweet.id in filter:
            if dictionary:
                ret[tweet.id] = tweet
            else:
                ret.append(tweet)

    return ret


def merge_by_tweet_id(tweets_dictionary, tsv_scores):
    """
    updates tweet scores based on the values presented in tsv_scores
    :param tweets_dictionary: {'tweetid': tweet}
    :param tsv_scores: [[tweetid, score], ...]
    :return: 
    """
    for item in tsv_scores:
        tweetid = item[0]
        score = item[1]
        tweet = tweets_dictionary[tweetid]
        tweet.score = float(score)
        tweets_dictionary[tweetid] = tweet

    return tweets_dictionary


def read_and_marge(tweet_xml, tweet_tsv):
    tweet_dict = tweet_xml_reader(tweet_xml, dictionary=True, filter={})
    tweet_tsv = read_tsv(tweet_tsv)

    return merge_by_tweet_id(tweet_dict, tweet_tsv)


def create_weka_arff(tweets_dictionary, path):
    """
    Creates input for Weka
    :param tweets_dictionary: 
    :param path: 
    :return: 
    """
    attributes = -1
    with open(path, 'wt') as f:
        for t in tweets_dictionary:
            tweet = tweets_dictionary[t]
            x, y = tweet.features(text2vec)
            if attributes == -1:
                # First sample
                attributes = len(x)
                f.write("@relation ARM\n")
                for i in range(attributes):
                    f.write("@attribute attr_" + str(i) + " numeric\n")
                f.write("@attribute score numeric\n\n@data\n")
            x.append(y)
            f.write(",".join([str(i) for i in x]) + "\n")


def load_empath(empath_features_path):
    tsv = read_tsv(empath_features_path)
    ret = {}
    for item in tsv:
        ret[item[0]] = item[1:]
    return ret


def create_tweet_vec(tweets_xml_path, empath_features_path):
    empath_feature_dict = load_empath(empath_features_path)
    tweet_dict = tweet_xml_reader(tweets_xml_path, set(empath_feature_dict.keys()), True)
    # output = {}
    counter = 0
    with open("../data/tweet_vec.tsv", 'wt') as f:
        for tweet_id, tweet in tweet_dict.items():
            if tweet_id in empath_feature_dict:
                vec = tweet_vector(tweet.text, tweet_id, empath_feature_dict)
                # output[tweet_id] = vec
                if counter % 100 == 0:
                    print("Tweet processed:", counter * 100 / len(empath_feature_dict))
                counter += 1
                f.write(tweet_id + "\t" + "\t".join([str(i) for i in vec]))

    print("Successfully stored the output in the data directory")


def load_tweet_vec(path):
    tsv = read_tsv(path)
    pass


def tweet_vector(tweet, tweet_id, empath_features,
                 tags={'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RBR', 'RBS', 'RB', 'VB', 'VBD',
                       'VBG', 'VBN', 'VBP', 'VBZ'}):
    vec = numpy.zeros(300)
    # tagged_tokens = pos_tag(tweet)
    # counter = 0
    # for item in tagged_tokens:
    #     if item[1] in tags:
    #         wv = word_vector(item[0])
    #         if wv is not None:
    #             vec = vec + wv
    #             counter += 1
    # vec = vec / counter
    tokens = tokenize(tweet, stop_words)
    counter = 0
    for item in tokens:
        wv = word_vector(item)
        if wv is not None:
            vec = vec + wv
            counter += 1
    vec = vec / counter

    # sent = sentiment(tweet)
    if tweet_id:
        empath = empath_features[tweet_id]
    else:
        empath = EmpathCat.analyze(tweet.text, categories=[
            "medical_emergency",
            "hate",
            "aggression",
            "envy",
            "crime",
            "masculine",
            "prison",
            "dispute",
            "nervousness",
            "weakness",
            "horror",
            "suffering",
            "kill",
            "redicule",
            "sexual",
            "fear",
            "violence",
            "neglect",
            "war",
            "disgust",
            "ugliness",
            "torment",
            "lust",
            "shame",
            "terrorism",
            "poor",
            "timidity",
            "alcohol",
            "monster",
            "health",
            "disappointment",
            "rage",
            "pain",
            "swearing_terms",
            "negative_emotional",
            "cold_war",
            "weapon",
            "children",
            "injury",
            "irritability",
        ])
    vec = numpy.concatenate((vec, empath, []), axis=0).tolist()
    return vec

# if __name__ == '__main__':
# This is only for testing

# tweet_dict = read_and_marge("../data/tweets-sample.xml", "../data/tweets-sample.tsv")
# create_weka_arff(tweet_dict, "../data/tweets.arff")
# print(tweet_dict)
