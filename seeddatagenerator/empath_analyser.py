from empath import Empath
from elasticparser.DataSearcher import TweetLocator
import utils.dataset as ds
from seeddatagenerator.DocParser import WikiParser


class GetEmpathCategory():
    def __init__(self, tweetid='', tweet='', feature={}):
        self.tweetid = tweetid
        self.tweet = tweet
        self.feature = feature

    def get_data(self):
        # tl = TweetLocator()
        # es = Elasticsearch([{'host': 'localhost', 'port': '9200', 'timeout': 300}])
        # query = {"query": {"match_all": {}}}
        lst_tweet = ds.tweet_xml_reader("F:\\Twitter Dataset\\1000000 tweets.xml", dictionary=False)
        # lst_returned_value = es.search(index='mtweet', body=query, search_type="scan", scroll="1m")['hits']['hits']
        return lst_tweet

    def get_category(self, dict_tweet):
        get_emapth_lexicon = Empath()
        lst_tweet_category = list()
        counter = 0
        all = 0
        for tweet in dict_tweet:
            # counter +=1
            # if counter >1000:
            #     break
            dict_empath_cat = get_emapth_lexicon.analyze(tweet.text, categories=[
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
            all += 1
            if sum([val for _, val in dict_empath_cat.items()]) > 0.1:
                counter += 1
                if counter > 1000:
                    print('Number of iterated items, and the number of accepted item', all, counter)
                    counter = 0
                lst_tweet_category.append(
                    GetEmpathCategory(tweetid=tweet.id, tweet=tweet.text, feature=dict_empath_cat))
        return lst_tweet_category


if __name__ == '__main__':
    gec = GetEmpathCategory()
    lst_tweet = gec.get_data()
    wk = WikiParser()
    lst_get_user_values = gec.get_category(lst_tweet)
    wk.generate_tsv_file(lst_get_user_values)
