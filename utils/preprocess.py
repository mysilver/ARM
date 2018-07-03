import re
import unicodedata

from nltk import ngrams
from nltk.tokenize import TweetTokenizer

tknzr = TweetTokenizer()


def tokenize(text, dictionary=None, ngrams_sizes=(3, 2), normilize_text=True):
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

    tokens = tknzr.tokenize(text)
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


def text2vec(text):
    return [len(text)]