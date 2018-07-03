import requests
from gensim.models.keyedvectors import KeyedVectors
from nltk import Text
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
from gensim.models.keyedvectors import KeyedVectors
import json
class WikiParser():

    def __init__(self, token ='', frequency = ''):
        super().__init__()
        self.token = token
        self.frequency = frequency

    def doc_extractor(self, path):
        #stemmer = SnowballStemmer('english')
        stop_words = set(stopwords.words('english'))
        wiki_data = requests.post(
            'http://d2dcrc.cse.unsw.edu.au:9091/ExtractionAPI-0.0.1-SNAPSHOT/rest/url/paragraph?url=' + path).json()
        json_dumps = json.dumps(wiki_data)
        json_loads = json.loads(json_dumps)
        wiki_data = json_loads["value"]
        tokens = nltk.word_tokenize(wiki_data)
        text = Text(tokens)
        set_tokens = set()
        lst_tokens = list()
        lst_token_freq= list()
        for token in tokens:
            if len(token) >3 and token not in stop_words:
                set_tokens.add(token.lower())
                lst_tokens.append(token.lower())
        for set_token in set_tokens:
            try:
                frequency = [token for token in lst_tokens if token==set_token]
                lst_token_freq.append(WikiParser(set_token, len(frequency)))
            except:
                continue
        lst_token_freq.sort(key=lambda x:x.frequency, reverse=True)
        return lst_token_freq[:40]

    def token_enrichment(self, lst_token):
        print("Loading word2vec model please wait... ")
        model = KeyedVectors.load_word2vec_format(
            'F:\GoogleNews-vectors-negative300.bin\GoogleNews-vectors-negative300.bin', binary=True, limit=200000)
        lst_user_enriched_data = list()
        for token in lst_token:
            print(token.token)
            user_feedback = input('Do you think Following token relevant to mental health disorder (Yes (y)| No (n)?')
            if user_feedback == 'y':
                lst_token_vecs = model.similar_by_word(token.token, topn=5)
                for item in lst_token_vecs:
                    lst_user_enriched_data.append(item[0])
        return lst_user_enriched_data


if __name__=='__main__':
    WP = WikiParser()
    lst_freq = WP.doc_extractor('https://en.wikipedia.org/wiki/Mental_disorder')
    token = WP.token_enrichment(lst_freq)
    for t in token:
        print(t)

        # json_keyword = requests.post(
        #     'http://d2dcrc.cse.unsw.edu.au:9091/ExtractionAPI-0.0.1-SNAPSHOT/rest/keyword?sentence='
        #     + wiki_data).json()
        # json_dumps = json.dumps(json_keyword)
        # json_loads = json.loads(json_dumps)
        # tokenization = json_loads["keyword"]
        # set_tokenization =set()
        # lst_text_token = list()
        # for token in tokenization.split(','):
        #     set_tokenization.add(token)
        #     lst_text_token.append(token)
        # for single_token in set_tokenization:
        #         [for token in lst_text_token].count()