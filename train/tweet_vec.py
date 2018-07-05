from utils.dataset import load_empath, tweet_xml_reader, create_tweet_vec

# empath_features = load_empath("../data/empath-features.tsv")
# tweet_dict = tweet_xml_reader("../data/1000000 tweets.xml", empath_features.keys(), True)

# Create Tweet-VSM
if True:
    create_tweet_vec("../data/1000000 tweets.xml", "../data/empath-features.tsv")

