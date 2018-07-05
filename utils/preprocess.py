import re
import unicodedata
from nltk import ngrams
from nltk.tokenize import TweetTokenizer
from stanfordcorenlp import StanfordCoreNLP
import json

tknzr = TweetTokenizer()
# nlp = StanfordCoreNLP('D:\TEMP\stanford-corenlp-full-2018-02-27')


def tokenize(text, dictionary=None, ngrams_sizes=(3, 2), normilize_text=True, stop_words={}):
    if normilize_text:
        text = normalize(text)

    if dictionary and ngrams_sizes:
        for i in ngrams_sizes:
            # join ngrams with '_'
            tokens = tknzr.tokenize(text)
            ngs = ngrams(tokens, i)
            for ng in ngs:
                phrs = "_".join(ng)
                if phrs in dictionary:
                    text = text.replace(" ".join(ng), phrs)

    tokens = [w for w in tknzr.tokenize(text) if w not in stop_words]

    return tokens


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = s.strip()
    return s


def sentiment(tweet, aggregation_method="avg"):
    """
    Neutral (2) and Negative (1), the range is from VeryNegative (0) to VeryPositive (4)
    :param aggregation_method:  "min", "avg"
    :param tweet:
    :return: The overall sentiment value for the sentence
    """
    res = nlp.annotate(tweet,
                       properties={
                           'annotators': 'sentiment'
                       })
    res = json.loads(res)

    min = 5
    sum = 0
    for s in res["sentences"]:

        if "min" == aggregation_method:
            if int(s["sentimentValue"]) < min:
                min = int(s["sentimentValue"])
        else:
            sum += int(s["sentimentValue"])
            # print("%d: '%s': %s %s" % (
            #     s["index"],
            #     " ".join([t["word"] for t in s["tokens"]]),
            #     s["sentimentValue"], s["sentiment"]))
    # print(min)
    if "min" == aggregation_method:
        return min
    else:
        return sum / len(res["sentences"])


def pos_tag(tweet):
    return nlp.pos_tag(tweet)


def text2vec(text):
    return [len(text)]


# print(pos_tag("I love you. I hate you"))
