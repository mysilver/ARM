import xml.etree.ElementTree

from tweet import Tweet


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


if __name__ == '__main__':
    # This is only for testing
    tweet_dict = tweet_xml_reader("../data/tweets-sample.xml", True)
    tweet_tsv = read_tsv("../data/tweets-sample.tsv")

    tweet_dict = merge_by_tweet_id(tweet_dict, tweet_tsv)
    print(tweet_dict)
