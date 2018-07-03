import xml.etree.ElementTree

from tweet import Tweet
from utils.preprocess import text2vec


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


def tweet_xml_reader(file, dictionary=False):
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
    tweet_dict = tweet_xml_reader(tweet_xml, True)
    tweet_tsv = read_tsv(tweet_tsv)

    return merge_by_tweet_id(tweet_dict, tweet_tsv)


def create_weka_arff(tweets_dictionary, path):
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


if __name__ == '__main__':
    # This is only for testing

    tweet_dict = read_and_marge("../data/tweets-sample.xml", "../data/tweets-sample.tsv")
    create_weka_arff(tweet_dict, "../data/tweets.arff")
    print(tweet_dict)
