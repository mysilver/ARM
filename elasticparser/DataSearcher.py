from elasticsearch import Elasticsearch
from utils import dataset as ds
import os
from seeddatagenerator.DocParser import WikiParser
class TweetLocator():
    def __init__(self):
        pass

    def define_mapping(self):
        es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
        mapping = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "my_analyzer": {
                            "tokenizer": "standard",
                            "filter": [
                                "standard", "lowercase", "my_stemmer"
                            ]
                        }
                    },
                    "filter": {
                        "my_stemmer": {
                            "type": "stemmer",
                            "name": "english"
                        }
                    }
                }
            },
            "mappings": {
                "tweet": {
                    "properties": {
                        "text": {
                            "type": "text", "analyzer": "my_analyzer"
                            },
                        "id": {
                                "type": "text", "analyzer": "my_analyzer"
                            },
                        "user_id":{
                            "type": "text", "analyzer": "my_analyzer"
                        },
                        "location":{
                            "type": "text", "analyzer": "my_analyzer"
                        },

                    }
                }
            }
        }
        if not es.indices.exists("tweethack"):
            es.indices.create(index = "tweethack", body=mapping)

    def index_data(self, lst_tweet):
        es = Elasticsearch([{"host": "localhost", "port": "9200", 'timeout': 50}])
        counter = 0
        indexed_Items = 0
        for tweet in lst_tweet:
            try:
                es.index(index='mtweet', doc_type='tweet', body={'text': tweet.text,'id':tweet.id, 'user_id':tweet.user_id,'location':tweet.location})
                indexed_Items += 1
            except:
                counter += 1
            if indexed_Items > 100000:
                print('100000 items indexed')
                indexed_Items=0
        print("Number of error:", str(counter))

    def search_data_and(self, query):
        es = Elasticsearch([{'host': 'localhost', 'port': '9200', 'timeout': 300}])
        lst_returned_value = es.search(index='mtweet', body={
            'size': '999999',
            'query': {
                'match': {
                    'text': {
                        'query': query,

                    }
                }
            }
        })['hits']['hits']
        return lst_returned_value

    def increase_window_size(self, index_name):
        es = Elasticsearch([{'host':'localhost', 'port':'9200', 'timeout':300}])
        es.indices.put_settings(index=index_name,
                                body={"index": {
                                    "max_result_window": 1000000
                                }})

if __name__=='__main__':
    WP = WikiParser()
    #dirname = os.path.dirname(__file__)
    #filename = os.path.join(dirname,"tweets-sample.xml")
    # lst_tweet = ds.tweet_xml_reader("F:\\Twitter Dataset\\1000000 tweets.xml")
    tl = TweetLocator()
    # tl.define_mapping()
    # tl.index_data(lst_tweet)
    #tl.increase_window_size('mtweet')
    lst_freq = WP.doc_extractor('https://en.wikipedia.org/wiki/Mental_disorder')
    lst_token = WP.token_enrichment(lst_freq)
    set_result= list()
    for token in lst_token:
        set_result.extend(tl.search_data_and(token))
    print(len(set_result))