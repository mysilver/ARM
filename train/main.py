import numpy

from utils.dataset import load_empath, tweet_xml_reader, create_tweet_vec, tweet_vector

# empath_features = load_empath("../data/empath-features.tsv")
# tweet_dict = tweet_xml_reader("../data/1000000 tweets.xml", empath_features.keys(), True)

# Create Tweet-VSM
from utils.word2vec import find_most_similar

if False:
    create_tweet_vec("../data/1000000 tweets.xml", "../data/empath-features.tsv")

with open("../data/seed-words", 'rt') as f:
    resultant_vector = numpy.zeros(340)
    counter = 0
    for tweet in f.readlines():
        vec = tweet_vector(tweet,tweet_id=None, empath_features=None)
        resultant_vector += numpy.array(vec)
        counter+=1

    resultant_vector /=counter
    # print(resultant_vector)

    # tweet_vec = load_empath("../data/tweet_vec.tsv")
    most_similarity = find_most_similar(resultant_vector, vectors_dictionary="../data/tweet_vec.tsv")
    print(most_similarity)
    tweet_dict = tweet_xml_reader("../data/1000000 tweets.xml", dictionary=True, filter=None)
    for t in most_similarity:
        print(t[0], tweet_dict[t[1]].text)
